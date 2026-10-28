/**
 * ============================================================================
 * GNC-X LLM EMBEDDABLE KERNEL (`gncx_llm_kernels.cpp` -> `gncx_llm_core.dll`)
 * ============================================================================
 * Base Cientifica: "GNC-X: O Genoma Computacional Espacial"
 * Autor da Concepcao Original: Zaqueu Ribeiro da Costa (omega-Core-Dev)
 * Co-Engenharia: Gemini 4 Argon + Um Cara Fora da Curva
 *
 * Nucleo C++20 Zero-Dependency projetado para Hardware de Entrada e Sistemas
 * Embarcados.
 *
 * PERFIS DE ARQUITETURA SUPORTADOS:
 *   - Perfil 1 (`gemma/2b`   - Gemma 2B IT): 18 Camadas, Q4_0 / Q8_0 / F32
 *   - Perfil 2 (`gemma4/e2b` - Gemma 4 E2B): 35 Camadas, Q4_K / Q6_K / BF16 / F32,
 *     com Per-Layer Embeddings (PLE), SWA/Global Hybrid Attention, Q/K/V RMSNorm,
 *     Post-Norms e Softcapping.
 *   - Perfil 3 (`qwen3/1.7b` - Qwen3 1.7B): 28 Camadas, Q4_K / Q6_K / F32,
 *     GQA 16/8, Q/K RMSNorm por cabeca e SwiGLU.
 * ============================================================================
 */

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <array>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <algorithm>
#include "gncx_genome_vm.hpp"
#include "gncx_geometric_mm.hpp"
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#define GNCX_LLM_API extern "C" __declspec(dllexport)
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#define GNCX_LLM_API extern "C" __attribute__((visibility("default")))
#endif

namespace {

constexpr int MAX_K = 17;

constexpr std::array<int64_t, MAX_K + 1> make_repunits() {
    std::array<int64_t, MAX_K + 1> r{};
    int64_t v = 1;
    for (int k = 0; k <= MAX_K; ++k) {
        r[k] = v;
        if (k < MAX_K) v = v * 10 + 1;
    }
    return r;
}

constexpr std::array<std::array<int64_t, 10>, MAX_K + 1> make_sieves() {
    std::array<std::array<int64_t, 10>, MAX_K + 1> s{};
    auto r = make_repunits();
    for (int k = 0; k <= MAX_K; ++k) {
        for (int d = 0; d < 10; ++d) {
            s[k][d] = static_cast<int64_t>(d) * r[k];
        }
    }
    return s;
}

constexpr auto REPUNITS = make_repunits();
constexpr auto DIAGONAL_SIEVES = make_sieves();

inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = static_cast<uint32_t>(h & 0x8000U) << 16;
    uint32_t exp  = (h >> 10) & 0x1FU;
    uint32_t mant = h & 0x03FFU;
    uint32_t bits = 0;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400U) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03FFU;
            bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000U | (mant << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }

    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(float));
    return out;
}

inline float bf16_to_fp32(uint16_t b) {
    uint32_t bits = static_cast<uint32_t>(b) << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(float));
    return out;
}

static float FP16_TABLE[65536];
static bool FP16_INIT = false;

static void ensure_fp16_lut() {
    if (!FP16_INIT) {
        for (uint32_t i = 0; i < 65536; ++i) {
            FP16_TABLE[i] = fp16_to_fp32(static_cast<uint16_t>(i));
        }
        FP16_INIT = true;
    }
}

class GNCXThreadPool {
public:
    int32_t num_threads{1};

    void init(int32_t requested_threads = 0) {
        if (!workers_.empty()) return;
        uint32_t hw = std::thread::hardware_concurrency();
        if (requested_threads <= 0) {
            num_threads = static_cast<int32_t>(std::clamp(hw, 2U, 12U));
        } else {
            num_threads = std::clamp(requested_threads, 1, 16);
        }
        stop_.store(false, std::memory_order_relaxed);
        job_id_.store(0, std::memory_order_relaxed);
        remaining_.store(0, std::memory_order_relaxed);

        for (int32_t t = 1; t < num_threads; ++t) {
            workers_.emplace_back([this, t]() { this->worker_loop(t); });
        }
    }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            if (workers_.empty()) return;
            stop_.store(true, std::memory_order_release);
            job_id_.fetch_add(1, std::memory_order_acq_rel);
        }
        cv_work_.notify_all();
        for (auto& w : workers_) {
            if (w.joinable()) w.join();
        }
        workers_.clear();
    }

    template <typename Fn>
    void parallel_for(int32_t total_items, Fn&& fn, int32_t min_grain = 128) {
        if (total_items < min_grain || num_threads <= 1 || workers_.empty()) {
            fn(0, 0, total_items);
            return;
        }

        using FnDecay = std::remove_reference_t<Fn>;
        struct Context {
            FnDecay* fn_ptr;
            int32_t total;
            int32_t n_threads;
        } ctx{&fn, total_items, num_threads};

        task_ctx_ = &ctx;
        task_trampoline_ = [](void* raw_ctx, int32_t tid) {
            auto* c = static_cast<Context*>(raw_ctx);
            const int32_t chunk = (c->total + c->n_threads - 1) / c->n_threads;
            const int32_t start = tid * chunk;
            const int32_t end = std::min(c->total, start + chunk);
            if (start < end) {
                (*(c->fn_ptr))(tid, start, end);
            }
        };

        remaining_.store(num_threads - 1, std::memory_order_relaxed);
        job_id_.fetch_add(1, std::memory_order_release);

        if (parked_workers_.load(std::memory_order_acquire) > 0) {
            std::lock_guard<std::mutex> lk(mtx_);
            cv_work_.notify_all();
        }

        task_trampoline_(&ctx, 0);

        while (remaining_.load(std::memory_order_acquire) != 0) {
#if defined(__SSE2__)
            _mm_pause();
#endif
        }
    }

private:
    void worker_loop(int32_t tid) {
        uint64_t seen_job = 0;
        while (true) {
            uint64_t cur_job = job_id_.load(std::memory_order_acquire);
            if (cur_job == seen_job) {
                // Spin-wait leve em userspace (~45 us) para evitar futex syscall entre matrizes da mesma camada
                for (int spin = 0; spin < 4096; ++spin) {
                    cur_job = job_id_.load(std::memory_order_acquire);
                    if (cur_job != seen_job || stop_.load(std::memory_order_relaxed)) break;
#if defined(__SSE2__)
                    _mm_pause();
#endif
                }
                if (cur_job == seen_job && !stop_.load(std::memory_order_relaxed)) {
                    std::unique_lock<std::mutex> lk(mtx_);
                    parked_workers_.fetch_add(1, std::memory_order_release);
                    cv_work_.wait(lk, [&]() {
                        return stop_.load(std::memory_order_relaxed) ||
                               job_id_.load(std::memory_order_acquire) != seen_job;
                    });
                    parked_workers_.fetch_sub(1, std::memory_order_release);
                    cur_job = job_id_.load(std::memory_order_acquire);
                }
            }
            if (stop_.load(std::memory_order_relaxed)) return;
            seen_job = cur_job;

            task_trampoline_(task_ctx_, tid);
            remaining_.fetch_sub(1, std::memory_order_release);
        }
    }

    std::vector<std::thread> workers_;
    std::mutex mtx_;
    std::condition_variable cv_work_;
    void (*task_trampoline_)(void*, int32_t){nullptr};
    void* task_ctx_{nullptr};
    alignas(64) std::atomic<uint64_t> job_id_{0};
    alignas(64) std::atomic<int32_t> remaining_{0};
    alignas(64) std::atomic<int32_t> parked_workers_{0};
    std::atomic<bool> stop_{false};
};

static GNCXThreadPool* GLOBAL_POOL_PTR = nullptr;

static GNCXThreadPool& get_global_pool() {
    if (GLOBAL_POOL_PTR == nullptr) {
        GLOBAL_POOL_PTR = new GNCXThreadPool();
        GLOBAL_POOL_PTR->init();
    }
    return *GLOBAL_POOL_PTR;
}

} // namespace

#pragma pack(push, 1)
struct GNCXCompactToken {
    uint8_t phase_2bit;
    uint8_t diag_d;
    uint16_t abs_delta;
};

struct BlockQ8_0 {
    uint16_t d_fp16;
    int8_t qs[32];
};

struct BlockQ4_0 {
    uint16_t d_fp16;
    uint8_t qs[16];
};

// Super-bloco Q4_K (256 elementos em 144 bytes)
struct BlockQ4_K {
    uint16_t d_fp16;
    uint16_t dmin_fp16;
    uint8_t scales[12];
    uint8_t qs[128];
};

// Super-bloco Q6_K (256 elementos em 210 bytes)
struct BlockQ6_K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    uint16_t d_fp16;
};

struct alignas(64) BlockActQ8 {
    float d_scale;
    int32_t sum_q;
    int32_t sum_q_lo16;
    int32_t sum_q_hi16;
    int8_t qs[32];
    uint8_t pad_cacheline[16];
};

struct GNCXTelemetryStats {
    uint64_t total_macs_equivalent;
    uint64_t zero_origin_cancellations;
    uint64_t zero_weight_skips;
    uint64_t xfold_bucket_merges;
};
#pragma pack(pop)

struct GLAMappedFileHandle {
#ifdef _WIN32
    HANDLE hFile{INVALID_HANDLE_VALUE};
    HANDLE hMap{nullptr};
#else
    int fd{-1};
#endif
    const uint8_t* base_ptr{nullptr};
    size_t file_size{0};
};

GNCX_LLM_API void* gncx_gla_map_readonly(const char* filepath, size_t* out_size) {
    ensure_fp16_lut();
    get_global_pool().init();

    auto* handle = new GLAMappedFileHandle();
#ifdef _WIN32
    handle->hFile = CreateFileA(
        filepath,
        GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN,
        nullptr
    );
    if (handle->hFile == INVALID_HANDLE_VALUE) {
        delete handle;
        return nullptr;
    }
    LARGE_INTEGER li{};
    GetFileSizeEx(handle->hFile, &li);
    handle->file_size = static_cast<size_t>(li.QuadPart);

    handle->hMap = CreateFileMappingA(handle->hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (handle->hMap == nullptr) {
        CloseHandle(handle->hFile);
        delete handle;
        return nullptr;
    }
    handle->base_ptr = static_cast<const uint8_t*>(
        MapViewOfFile(handle->hMap, FILE_MAP_READ, 0, 0, 0)
    );
    if (handle->base_ptr == nullptr) {
        CloseHandle(handle->hMap);
        CloseHandle(handle->hFile);
        delete handle;
        return nullptr;
    }
#else
    handle->fd = open(filepath, O_RDONLY);
    if (handle->fd < 0) {
        delete handle;
        return nullptr;
    }
    struct stat st{};
    fstat(handle->fd, &st);
    handle->file_size = static_cast<size_t>(st.st_size);
    handle->base_ptr = static_cast<const uint8_t*>(
        mmap(nullptr, handle->file_size, PROT_READ, MAP_PRIVATE, handle->fd, 0)
    );
    if (handle->base_ptr != MAP_FAILED && handle->base_ptr != nullptr) {
        posix_fadvise(handle->fd, 0, handle->file_size, POSIX_FADV_WILLNEED);
        madvise(const_cast<void*>(static_cast<const void*>(handle->base_ptr)), handle->file_size, MADV_WILLNEED);
    }
#endif
    if (out_size != nullptr) *out_size = handle->file_size;
    return handle;
}

GNCX_LLM_API const uint8_t* gncx_gla_get_base_ptr(void* opaque_handle) {
    if (opaque_handle == nullptr) return nullptr;
    return static_cast<GLAMappedFileHandle*>(opaque_handle)->base_ptr;
}

GNCX_LLM_API void gncx_gla_unmap(void* opaque_handle) {
    get_global_pool().shutdown();
    if (opaque_handle == nullptr) return;
    auto* handle = static_cast<GLAMappedFileHandle*>(opaque_handle);
#ifdef _WIN32
    if (handle->base_ptr != nullptr) UnmapViewOfFile(handle->base_ptr);
    if (handle->hMap != nullptr) CloseHandle(handle->hMap);
    if (handle->hFile != INVALID_HANDLE_VALUE) CloseHandle(handle->hFile);
#else
    if (handle->base_ptr != nullptr) munmap(const_cast<uint8_t*>(handle->base_ptr), handle->file_size);
    if (handle->fd >= 0) close(handle->fd);
#endif
    delete handle;
}

/**
 * 0C. Alocador e Carregador de Camadas de Cálculo Ativo na Memória RAM (GLA RAM Pinning)
 * Integrado com GNCXDDR3Arena:
 *   - Alinhamento HugePage de 2 MB + MADV_HUGEPAGE (destrava bits físicos 20..0)
 *   - Coloração Diagonal Φ(k, d, R) de Bancos DDR3 (bits 16..14) e Sets L1d (bits 11..6)
 */
struct GNCXHugePageSlab {
    uint8_t*  mmap_base{nullptr};
    size_t    mmap_bytes{0};
    uintptr_t aligned_base{0};
    size_t    usable_bytes{0};
    size_t    cursor{0};
    int32_t   ref_count{0};
};

struct GLARamBufferHandle {
    uint8_t*          mmap_base{nullptr};
    size_t            mmap_bytes{0};
    uint8_t*          raw_ptr{nullptr};
    size_t            total_bytes{0};
    uint8_t           ddr3_bank{0};
    uint8_t           l1_set{0};
    GNCXHugePageSlab* shared_slab{nullptr};
};

static uint32_t g_orbital_alloc_seq = 0;
static GNCXHugePageSlab* g_active_small_slab = nullptr;

static GNCXHugePageSlab* create_hugepage_slab(size_t min_usable_bytes) {
    constexpr size_t HUGEPAGE_2MB = 2ULL * 1024ULL * 1024ULL;
    const size_t usable = (min_usable_bytes + HUGEPAGE_2MB - 1ULL) & ~(HUGEPAGE_2MB - 1ULL);
    const size_t alloc_len = usable + HUGEPAGE_2MB;

#ifdef _WIN32
    void* ptr = VirtualAlloc(nullptr, alloc_len, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (ptr == nullptr) return nullptr;
#else
    void* ptr = mmap(nullptr, alloc_len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr == nullptr || ptr == MAP_FAILED) return nullptr;
#endif

    uintptr_t raw_addr = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t aligned_base = (raw_addr + HUGEPAGE_2MB - 1ULL) & ~(HUGEPAGE_2MB - 1ULL);

#if !defined(_WIN32) && defined(MADV_HUGEPAGE)
    if (aligned_base + usable <= raw_addr + alloc_len) {
        madvise(reinterpret_cast<void*>(aligned_base), usable, MADV_HUGEPAGE);
    }
#endif

    auto* slab = new GNCXHugePageSlab();
    slab->mmap_base    = static_cast<uint8_t*>(ptr);
    slab->mmap_bytes   = alloc_len;
    slab->aligned_base = aligned_base;
    slab->usable_bytes = usable;
    slab->cursor       = 0;
    slab->ref_count    = 0;
    return slab;
}

static size_t find_next_colored_offset(size_t cursor, uint32_t bank_slot, uint32_t l1_slot) {
    constexpr size_t BANK_CYCLE_128K = 131072ULL;
    size_t base_128k = cursor & ~(BANK_CYCLE_128K - 1ULL);
    const size_t target_low = (static_cast<size_t>(bank_slot & 0x7U) << 14U)
                            | (static_cast<size_t>(l1_slot & 0x3FU)  << 6U);

    for (size_t p4k = 0; p4k < 4; ++p4k) {
        size_t cand = base_128k | target_low | (p4k << 12U);
        if (cand >= cursor) {
            return cand;
        }
    }
    return base_128k + BANK_CYCLE_128K + target_low;
}

GNCX_LLM_API void* gncx_ram_alloc_colored(size_t total_bytes, uint8_t k, uint8_t d, uint8_t R) {
    constexpr size_t HUGEPAGE_2MB = 2ULL * 1024ULL * 1024ULL;
    if (total_bytes == 0) total_bytes = 64;

    const uint32_t R_odd     = (R | 1U) & 0x7U;
    const uint32_t bank_slot = (static_cast<uint32_t>(d) * R_odd + k) & 0x7U;
    const uint32_t l1_slot   = (static_cast<uint32_t>(d) * R_odd * 3U + static_cast<uint32_t>(k) * 7U) & 0x3FU;

    if (total_bytes <= 1024ULL * 1024ULL) {
        constexpr size_t SLAB_DEFAULT_BYTES = 2ULL * HUGEPAGE_2MB; // 4 MB
        if (g_active_small_slab == nullptr) {
            g_active_small_slab = create_hugepage_slab(SLAB_DEFAULT_BYTES);
            if (g_active_small_slab == nullptr) return nullptr;
        }
        size_t colored_off = find_next_colored_offset(g_active_small_slab->cursor, bank_slot, l1_slot);
        if (colored_off + total_bytes > g_active_small_slab->usable_bytes) {
            g_active_small_slab = create_hugepage_slab(std::max(SLAB_DEFAULT_BYTES, total_bytes + HUGEPAGE_2MB));
            if (g_active_small_slab == nullptr) return nullptr;
            colored_off = find_next_colored_offset(0, bank_slot, l1_slot);
        }

        uintptr_t final_addr = g_active_small_slab->aligned_base + colored_off;
        g_active_small_slab->cursor = (colored_off + total_bytes + 63ULL) & ~63ULL;
        g_active_small_slab->ref_count++;

        auto* handle        = new GLARamBufferHandle();
        handle->mmap_base   = nullptr;
        handle->mmap_bytes  = 0;
        handle->raw_ptr     = reinterpret_cast<uint8_t*>(final_addr);
        handle->total_bytes = total_bytes;
        handle->ddr3_bank   = static_cast<uint8_t>((final_addr >> 14U) & 0x7U);
        handle->l1_set      = static_cast<uint8_t>((final_addr >> 6U)  & 0x3FU);
        handle->shared_slab = g_active_small_slab;
        return handle;
    }

    const size_t color_padding = (total_bytes <= 64ULL * 1024ULL * 1024ULL)
        ? ((static_cast<size_t>(bank_slot) << 14U) | (static_cast<size_t>(l1_slot) << 6U))
        : (static_cast<size_t>(l1_slot) << 6U);

    const size_t hp_span   = (total_bytes + color_padding + HUGEPAGE_2MB - 1ULL) & ~(HUGEPAGE_2MB - 1ULL);
    const size_t alloc_len = hp_span + HUGEPAGE_2MB;

#ifdef _WIN32
    void* ptr = VirtualAlloc(nullptr, alloc_len, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (ptr == nullptr) return nullptr;
#else
    void* ptr = mmap(nullptr, alloc_len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr == nullptr || ptr == MAP_FAILED) return nullptr;
#endif

    uintptr_t raw_addr     = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t aligned_base = (raw_addr + HUGEPAGE_2MB - 1ULL) & ~(HUGEPAGE_2MB - 1ULL);

#if !defined(_WIN32) && defined(MADV_HUGEPAGE)
    if (aligned_base + hp_span <= raw_addr + alloc_len) {
        madvise(reinterpret_cast<void*>(aligned_base), hp_span, MADV_HUGEPAGE);
    }
#endif

    uintptr_t final_addr = aligned_base + color_padding;
    auto* handle         = new GLARamBufferHandle();
    handle->mmap_base    = static_cast<uint8_t*>(ptr);
    handle->mmap_bytes   = alloc_len;
    handle->raw_ptr      = reinterpret_cast<uint8_t*>(final_addr);
    handle->total_bytes  = total_bytes;
    handle->ddr3_bank    = static_cast<uint8_t>((final_addr >> 14U) & 0x7U);
    handle->l1_set       = static_cast<uint8_t>((final_addr >> 6U)  & 0x3FU);
    handle->shared_slab  = nullptr;
    return handle;
}

GNCX_LLM_API void* gncx_ram_alloc(size_t total_bytes) {
    uint32_t seq = g_orbital_alloc_seq++;
    uint8_t k = static_cast<uint8_t>(seq & 0x3U);
    uint8_t d = static_cast<uint8_t>(seq & 0x3FU);
    uint8_t R = gncx::PhiTranslator::ROT[k];
    return gncx_ram_alloc_colored(total_bytes, k, d, R);
}

GNCX_LLM_API uint8_t* gncx_ram_get_ptr(void* ram_handle) {
    if (ram_handle == nullptr) return nullptr;
    return static_cast<GLARamBufferHandle*>(ram_handle)->raw_ptr;
}

GNCX_LLM_API uint8_t gncx_ram_get_ddr3_bank(void* ram_handle) {
    if (ram_handle == nullptr) return 0;
    return static_cast<GLARamBufferHandle*>(ram_handle)->ddr3_bank;
}

GNCX_LLM_API uint8_t gncx_ram_get_l1_set(void* ram_handle) {
    if (ram_handle == nullptr) return 0;
    return static_cast<GLARamBufferHandle*>(ram_handle)->l1_set;
}

GNCX_LLM_API void gncx_ram_free(void* ram_handle) {
    if (ram_handle == nullptr) return;
    auto* handle = static_cast<GLARamBufferHandle*>(ram_handle);
    if (handle->shared_slab != nullptr) {
        handle->shared_slab->ref_count--;
        if (handle->shared_slab->ref_count <= 0) {
            if (handle->shared_slab == g_active_small_slab) {
                g_active_small_slab = nullptr;
            }
#ifdef _WIN32
            VirtualFree(handle->shared_slab->mmap_base, 0, MEM_RELEASE);
#else
            munmap(handle->shared_slab->mmap_base, handle->shared_slab->mmap_bytes);
#endif
            delete handle->shared_slab;
        }
    } else if (handle->mmap_base != nullptr) {
#ifdef _WIN32
        VirtualFree(handle->mmap_base, 0, MEM_RELEASE);
#else
        munmap(handle->mmap_base, handle->mmap_bytes);
#endif
    }
    delete handle;
}

GNCX_LLM_API void gncx_ram_copy_chunk(
    void* ram_handle,
    size_t dst_offset,
    const uint8_t* src_base,
    size_t src_offset,
    size_t num_bytes
) {
    if (ram_handle == nullptr || src_base == nullptr || num_bytes == 0) return;
    auto* handle = static_cast<GLARamBufferHandle*>(ram_handle);
    std::memcpy(handle->raw_ptr + dst_offset, src_base + src_offset, num_bytes);
}

/**
 * Prefetch assíncrono / Antecipado de Faixa de Memória (Descompasso de Camadas Disco -> RAM)
 */
GNCX_LLM_API void gncx_gla_prefetch_range(const uint8_t* mmap_base, size_t offset, size_t num_bytes) {
    if (mmap_base == nullptr || num_bytes == 0) return;
    const void* target_addr = static_cast<const void*>(mmap_base + offset);
#ifdef _WIN32
    typedef struct _WIN32_MEMORY_RANGE_ENTRY {
        PVOID  VirtualAddress;
        SIZE_T NumberOfBytes;
    } WIN32_MEMORY_RANGE_ENTRY, *PWIN32_MEMORY_RANGE_ENTRY;

    typedef BOOL (WINAPI *pfnPrefetchVirtualMemory)(
        HANDLE hProcess,
        ULONG_PTR NumberOfEntries,
        PWIN32_MEMORY_RANGE_ENTRY VirtualAddresses,
        ULONG Flags
    );

    static pfnPrefetchVirtualMemory s_pfnPrefetch = []() -> pfnPrefetchVirtualMemory {
        HMODULE hKernel = GetModuleHandleA("kernel32.dll");
        if (hKernel) {
            return reinterpret_cast<pfnPrefetchVirtualMemory>(
                GetProcAddress(hKernel, "PrefetchVirtualMemory")
            );
        }
        return nullptr;
    }();

    if (s_pfnPrefetch) {
        WIN32_MEMORY_RANGE_ENTRY entry;
        entry.VirtualAddress = const_cast<void*>(target_addr);
        entry.NumberOfBytes = num_bytes;
        s_pfnPrefetch(GetCurrentProcess(), 1, &entry, 0);
    } else {
        // Fallback: toque de leitura nas páginas (1 byte a cada 4096 bytes)
        const volatile uint8_t* p = static_cast<const volatile uint8_t*>(target_addr);
        for (size_t i = 0; i < num_bytes; i += 4096) {
            uint8_t dummy = p[i];
            (void)dummy;
        }
    }
#else
    uintptr_t addr = reinterpret_cast<uintptr_t>(target_addr);
    uintptr_t page_mask = 4096 - 1;
    uintptr_t aligned_addr = addr & ~page_mask;
    size_t adjusted_bytes = num_bytes + (addr - aligned_addr);
#if defined(MADV_WILLNEED)
    madvise(reinterpret_cast<void*>(aligned_addr), adjusted_bytes, MADV_WILLNEED);
#elif defined(POSIX_MADV_WILLNEED)
    posix_madvise(reinterpret_cast<void*>(aligned_addr), adjusted_bytes, POSIX_MADV_WILLNEED);
#endif
#endif
}

/**
 * 0D. Genome VM: Mapeamento de Camada com Origem Central (Equador)
 */
GNCX_LLM_API void gncx_genome_get_layer_coordinate(
    int32_t layer_index,
    int32_t total_layers,
    int32_t base_k,
    int32_t* out_rel_layer,
    int32_t* out_dist_centro,
    int32_t* out_crivo_k,
    int32_t* out_hemisferio
) {
    auto coord = gncx::make_genome_coordinate(layer_index, total_layers, base_k);
    if (out_rel_layer)   *out_rel_layer   = coord.rel_layer;
    if (out_dist_centro) *out_dist_centro = coord.dist_centro;
    if (out_crivo_k)     *out_crivo_k     = coord.crivo_k;
    if (out_hemisferio)  *out_hemisferio  = static_cast<int32_t>(coord.hemisferio);
}



/**
 * 1A. Extrai e dequantiza 1 linha de Embedding em formato Q8_0 diretamente do mmap
 */
GNCX_LLM_API void gncx_fetch_token_embedding_q8_0(
    const uint8_t* __restrict mmap_base,
    size_t tensor_abs_offset,
    int32_t token_id,
    int32_t embed_dim,
    float scale_factor,
    float* __restrict out_hidden
) {
    const size_t blocks_per_row = static_cast<size_t>(embed_dim) / 32;
    const size_t row_byte_stride = blocks_per_row * sizeof(BlockQ8_0);
    const auto* __restrict row_blocks = reinterpret_cast<const BlockQ8_0*>(
        mmap_base + tensor_abs_offset + static_cast<size_t>(token_id) * row_byte_stride
    );

    for (size_t b = 0; b < blocks_per_row; ++b) {
        const float d = FP16_TABLE[row_blocks[b].d_fp16] * scale_factor;
        float* __restrict dst = out_hidden + (b * 32);
        const int8_t* __restrict qs = row_blocks[b].qs;
        for (int i = 0; i < 32; ++i) {
            dst[i] = static_cast<float>(qs[i]) * d;
        }
    }
}

/**
 * 1B. Extrai e dequantiza 1 linha de Embedding em formato Q6_K (usado em gemma4:e2b token_embd)
 */
GNCX_LLM_API void gncx_fetch_token_embedding_q6_k(
    const uint8_t* __restrict mmap_base,
    size_t tensor_abs_offset,
    int32_t token_id,
    int32_t embed_dim,
    float scale_factor,
    float* __restrict out_hidden
) {
    const size_t blocks_per_row = static_cast<size_t>(embed_dim) / 256;
    const size_t row_byte_stride = blocks_per_row * sizeof(BlockQ6_K);
    const auto* __restrict row_blocks = reinterpret_cast<const BlockQ6_K*>(
        mmap_base + tensor_abs_offset + static_cast<size_t>(token_id) * row_byte_stride
    );

    for (size_t b = 0; b < blocks_per_row; ++b) {
        const BlockQ6_K& blk = row_blocks[b];
        const float d = FP16_TABLE[blk.d_fp16] * scale_factor;
        float* __restrict y = out_hidden + (b * 256);
        const uint8_t* __restrict ql = blk.ql;
        const uint8_t* __restrict qh = blk.qh;
        const int8_t* __restrict sc = blk.scales;

        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                const int8_t q1 = static_cast<int8_t>((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = static_cast<int8_t>((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = static_cast<int8_t>((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = static_cast<int8_t>((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * static_cast<float>(sc[is + 0]) * static_cast<float>(q1);
                y[l + 32] = d * static_cast<float>(sc[is + 2]) * static_cast<float>(q2);
                y[l + 64] = d * static_cast<float>(sc[is + 4]) * static_cast<float>(q3);
                y[l + 96] = d * static_cast<float>(sc[is + 6]) * static_cast<float>(q4);
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

/**
 * 1C. Extrai e dequantiza 1 linha de Embedding em formato BF16 (usado em per_layer_token_embd)
 */
GNCX_LLM_API void gncx_fetch_token_embedding_bf16(
    const uint8_t* __restrict mmap_base,
    size_t tensor_abs_offset,
    int32_t token_id,
    int32_t embed_dim,
    float scale_factor,
    float* __restrict out_hidden
) {
    const size_t row_byte_stride = static_cast<size_t>(embed_dim) * sizeof(uint16_t);
    const auto* __restrict row_ptr = reinterpret_cast<const uint16_t*>(
        mmap_base + tensor_abs_offset + static_cast<size_t>(token_id) * row_byte_stride
    );
    for (int32_t i = 0; i < embed_dim; ++i) {
        out_hidden[i] = bf16_to_fp32(row_ptr[i]) * scale_factor;
    }
}

static uint64_t g_norm_epoch = 0;
static uint64_t g_norm_epoch_cancellations = 0;
static const GNCXCompactToken* g_norm_epoch_tokens_ptr = nullptr;

/**
 * 2. Executa RMSNorm + Codificacao Quaternaria GNC-X de 4 Fases (`GNCXCompactToken`)
 */
GNCX_LLM_API void gncx_rmsnorm_and_encode_4phase(
    const float* __restrict input_hidden,
    const uint8_t* __restrict mmap_base,
    size_t norm_weight_offset,
    int32_t dim,
    float eps,
    int32_t layer_k,
    float* __restrict out_normed_fp32,
    GNCXCompactToken* __restrict out_tokens,
    float* __restrict out_quant_scale
) {
    g_norm_epoch++;
    const auto* __restrict norm_w = reinterpret_cast<const float*>(mmap_base + norm_weight_offset);

    double sum_sq = 0.0;
    int32_t i = 0;
#if defined(__SSE2__)
    __m128 sum4 = _mm_setzero_ps();
    for (; i + 3 < dim; i += 4) {
        __m128 v = _mm_loadu_ps(input_hidden + i);
        sum4 = _mm_add_ps(sum4, _mm_mul_ps(v, v));
    }
    alignas(16) float s[4];
    _mm_storeu_ps(s, sum4);
    sum_sq = static_cast<double>(s[0]) + static_cast<double>(s[1]) + static_cast<double>(s[2]) + static_cast<double>(s[3]);
#endif
    for (; i < dim; ++i) {
        double v = static_cast<double>(input_hidden[i]);
        sum_sq += v * v;
    }
    const float inv_rms = 1.0f / std::sqrt(static_cast<float>(sum_sq / dim) + eps);

    i = 0;
#if defined(__SSE2__)
    const __m128 v_inv_rms = _mm_set1_ps(inv_rms);
    for (; i + 3 < dim; i += 4) {
        __m128 in_v = _mm_loadu_ps(input_hidden + i);
        __m128 nw_v = _mm_loadu_ps(norm_w + i);
        __m128 res  = _mm_mul_ps(_mm_mul_ps(in_v, v_inv_rms), nw_v);
        _mm_storeu_ps(out_normed_fp32 + i, res);
    }
#endif
    for (; i < dim; ++i) {
        out_normed_fp32[i] = input_hidden[i] * inv_rms * norm_w[i];
    }

    if (out_tokens == nullptr) {
        if (out_quant_scale != nullptr) {
            *out_quant_scale = 1.0f;
        }
        return;
    }

    float max_abs = 1e-9f;
    for (int32_t j = 0; j < dim; ++j) {
        float av = std::fabs(out_normed_fp32[j]);
        if (av > max_abs) max_abs = av;
    }

    const int lk = std::clamp(layer_k, 0, MAX_K);
    const int64_t r_k = REPUNITS[lk];
    const float target_max_int = static_cast<float>(9 * r_k);
    const float scale_to_int = target_max_int / max_abs;
    const float scale_to_fp = max_abs / target_max_int;
    if (out_quant_scale != nullptr) {
        *out_quant_scale = scale_to_fp;
    }

    const float inv_rk = 1.0f / static_cast<float>(r_k);
    const float half_rk = static_cast<float>(r_k >> 1);
    std::array<uint32_t, 10> pos_pivots{};
    std::array<uint32_t, 10> neg_pivots{};

    for (int32_t j = 0; j < dim; ++j) {
        int64_t val = static_cast<int64_t>(std::nearbyint(out_normed_fp32[j] * scale_to_int));
        int64_t abs_v = (val >= 0) ? val : -val;
        int32_t d_approx = static_cast<int32_t>((static_cast<float>(abs_v) + half_rk) * inv_rk);
        if (d_approx > 9) d_approx = 9;
        uint8_t d = static_cast<uint8_t>(d_approx);
        int64_anchor:
        int64_t anchor = DIAGONAL_SIEVES[lk][d];

        int64_t signed_anchor = (val >= 0) ? anchor : -anchor;
        int64_t signed_res = val - signed_anchor;
        uint16_t abs_res = static_cast<uint16_t>(signed_res >= 0 ? signed_res : -signed_res);

        uint8_t b1 = (val >= 0) ? 1U : 0U;
        uint8_t b0 = (signed_res >= 0) ? 1U : 0U;
        uint8_t phase = static_cast<uint8_t>((b1 << 1) | b0);

        out_tokens[j] = GNCXCompactToken{phase, d, abs_res};
        pos_pivots[d] += b1;
        neg_pivots[d] += (b1 ^ 1U);
    }

    uint64_t canceled = 0;
    for (size_t d = 0; d < 10; ++d) {
        canceled += std::min(pos_pivots[d], neg_pivots[d]);
    }
    g_norm_epoch_cancellations = canceled;
    g_norm_epoch_tokens_ptr = out_tokens;
}

/**
 * 2B. RMSNorm multi-vetores In-Place (usado para attn_q_norm, attn_k_norm, Vcur norm e PLE norm)
 */
GNCX_LLM_API void gncx_rmsnorm_inplace_heads(
    float* __restrict vec,
    const uint8_t* __restrict mmap_base,
    size_t norm_weight_offset,
    int32_t has_weight,
    int32_t n_heads,
    int32_t head_dim,
    float eps
) {
    const float* __restrict norm_w = has_weight
        ? reinterpret_cast<const float*>(mmap_base + norm_weight_offset)
        : nullptr;

    for (int32_t h = 0; h < n_heads; ++h) {
        float* __restrict v = vec + static_cast<size_t>(h) * head_dim;
        double sum_sq = 0.0;
        int32_t i = 0;
#if defined(__SSE2__)
        __m128 sum4 = _mm_setzero_ps();
        for (; i + 3 < head_dim; i += 4) {
            __m128 vi = _mm_loadu_ps(v + i);
            sum4 = _mm_add_ps(sum4, _mm_mul_ps(vi, vi));
        }
        alignas(16) float s[4];
        _mm_storeu_ps(s, sum4);
        sum_sq = static_cast<double>(s[0]) + static_cast<double>(s[1]) + static_cast<double>(s[2]) + static_cast<double>(s[3]);
#endif
        for (; i < head_dim; ++i) {
            double x = static_cast<double>(v[i]);
            sum_sq += x * x;
        }
        const float inv_rms = 1.0f / std::sqrt(static_cast<float>(sum_sq / head_dim) + eps);
        i = 0;
#if defined(__SSE2__)
        const __m128 v_inv = _mm_set1_ps(inv_rms);
        if (norm_w != nullptr) {
            for (; i + 3 < head_dim; i += 4) {
                __m128 vi = _mm_loadu_ps(v + i);
                __m128 wi = _mm_loadu_ps(norm_w + i);
                _mm_storeu_ps(v + i, _mm_mul_ps(_mm_mul_ps(vi, v_inv), wi));
            }
        } else {
            for (; i + 3 < head_dim; i += 4) {
                __m128 vi = _mm_loadu_ps(v + i);
                _mm_storeu_ps(v + i, _mm_mul_ps(vi, v_inv));
            }
        }
#endif
        if (norm_w != nullptr) {
            for (; i < head_dim; ++i) {
                v[i] = v[i] * inv_rms * norm_w[i];
            }
        } else {
            for (; i < head_dim; ++i) {
                v[i] = v[i] * inv_rms;
            }
        }
    }
}

static void quantize_and_fold_activations_q8(
    const float* __restrict in_vec,
    int32_t in_dim,
    std::vector<BlockActQ8>& out_act_blocks
) {
    const size_t n_blocks = static_cast<size_t>(in_dim) / 32;
    out_act_blocks.resize(n_blocks);

    for (size_t b = 0; b < n_blocks; ++b) {
        const float* __restrict src = in_vec + (b * 32);
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) {
            float av = std::fabs(src[i]);
            if (av > amax) amax = av;
        }
        BlockActQ8& blk = out_act_blocks[b];
        if (amax == 0.0f) {
            blk.d_scale = 0.0f;
            blk.sum_q = 0;
            blk.sum_q_lo16 = 0;
            blk.sum_q_hi16 = 0;
            std::memset(blk.qs, 0, 32);
            continue;
        }
        const float inv_d = 127.0f / amax;
        blk.d_scale = amax / 127.0f;
        int32_t s_lo = 0;
        int32_t s_hi = 0;
        for (int i = 0; i < 16; ++i) {
            int8_t q_lo = static_cast<int8_t>(std::nearbyint(src[i] * inv_d));
            int8_t q_hi = static_cast<int8_t>(std::nearbyint(src[i + 16] * inv_d));
            blk.qs[i] = q_lo;
            blk.qs[i + 16] = q_hi;
            s_lo += static_cast<int32_t>(q_lo);
            s_hi += static_cast<int32_t>(q_hi);
        }
        blk.sum_q_lo16 = s_lo;
        blk.sum_q_hi16 = s_hi;
        blk.sum_q = s_lo + s_hi;
    }
}

static uint64_t count_4phase_cancellations(const GNCXCompactToken* __restrict in_tokens, int32_t in_dim) {
    if (in_tokens == nullptr) return 0;
    if (in_tokens == g_norm_epoch_tokens_ptr) {
        return g_norm_epoch_cancellations;
    }
    std::array<uint32_t, 10> pos_pivots{};
    std::array<uint32_t, 10> neg_pivots{};
    for (int32_t i = 0; i < in_dim; ++i) {
        uint8_t d = in_tokens[i].diag_d;
        uint8_t b1 = (in_tokens[i].phase_2bit >> 1) & 1U;
        pos_pivots[d] += b1;
        neg_pivots[d] += (b1 ^ 1U);
    }
    uint64_t canceled = 0;
    for (size_t d = 0; d < 10; ++d) {
        canceled += std::min(pos_pivots[d], neg_pivots[d]);
    }
    return canceled;
}

/**
 * Kernel Vetorial SIMD Adaptativo (AVX2 256-bit ou SSSE3/SSE4.1 128-bit)
 * para Produto Escalar Inteiro Q4_0 x Q8
 */
static inline int32_t gncx_dot32_q4_0_q8_simd(const uint8_t* __restrict q4_qs, const int8_t* __restrict q8_qs) {
#if defined(__AVX2__)
    const __m128i raw16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q4_qs));
    const __m128i mask_0f = _mm_set1_epi8(0x0F);
    const __m128i lo16 = _mm_and_si128(raw16, mask_0f);
    const __m128i hi16 = _mm_and_si128(_mm_srli_epi16(raw16, 4), mask_0f);
    const __m256i w_u8 = _mm256_set_m128i(hi16, lo16);
    const __m256i x_s8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8_qs));

    const __m256i prod16 = _mm256_maddubs_epi16(w_u8, x_s8);
    const __m256i prod32 = _mm256_madd_epi16(prod16, _mm256_set1_epi16(1));

    const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(prod32), _mm256_extracti128_si256(prod32, 1));
    const __m128i sum64  = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
    const __m128i sum32  = _mm_add_epi32(sum64,  _mm_shuffle_epi32(sum64,  _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(sum32);
#elif defined(__SSSE3__)
    const __m128i raw16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q4_qs));
    const __m128i mask_0f = _mm_set1_epi8(0x0F);
    const __m128i lo16 = _mm_and_si128(raw16, mask_0f);
    const __m128i hi16 = _mm_and_si128(_mm_srli_epi16(raw16, 4), mask_0f);

    const __m128i x_lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q8_qs));
    const __m128i x_hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q8_qs + 16));

    const __m128i p16_lo = _mm_maddubs_epi16(lo16, x_lo);
    const __m128i p16_hi = _mm_maddubs_epi16(hi16, x_hi);
    const __m128i p16_sum = _mm_add_epi16(p16_lo, p16_hi);
    const __m128i prod32 = _mm_madd_epi16(p16_sum, _mm_set1_epi16(1));

    const __m128i sum64 = _mm_add_epi32(prod32, _mm_shuffle_epi32(prod32, _MM_SHUFFLE(1, 0, 3, 2)));
    const __m128i sum32 = _mm_add_epi32(sum64,  _mm_shuffle_epi32(sum64,  _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(sum32);
#else
    const int8_t* __restrict x_lo = q8_qs;
    const int8_t* __restrict x_hi = q8_qs + 16;
    int32_t int_dot = 0;
    for (int j = 0; j < 16; ++j) {
        const uint8_t packed = q4_qs[j];
        int_dot += static_cast<int32_t>(packed & 0x0FU) * static_cast<int32_t>(x_lo[j])
                 + static_cast<int32_t>(packed >> 4)    * static_cast<int32_t>(x_hi[j]);
    }
    return int_dot;
#endif
}

/**
 * Kernel Vetorial SIMD Adaptativo para Q8_0 x Q8
 */
static inline int32_t gncx_dot32_q8_0_q8_simd(const int8_t* __restrict w_qs, const int8_t* __restrict x_qs) {
#if defined(__AVX2__)
    const __m128i w_lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w_qs));
    const __m128i w_hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w_qs + 16));
    const __m128i x_lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x_qs));
    const __m128i x_hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x_qs + 16));

    const __m256i p_lo = _mm256_madd_epi16(_mm256_cvtepi8_epi16(w_lo), _mm256_cvtepi8_epi16(x_lo));
    const __m256i p_hi = _mm256_madd_epi16(_mm256_cvtepi8_epi16(w_hi), _mm256_cvtepi8_epi16(x_hi));
    const __m256i prod32 = _mm256_add_epi32(p_lo, p_hi);

    const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(prod32), _mm256_extracti128_si256(prod32, 1));
    const __m128i sum64  = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
    const __m128i sum32  = _mm_add_epi32(sum64,  _mm_shuffle_epi32(sum64,  _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(sum32);
#elif defined(__SSE4_1__)
    const __m128i w0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w_qs));
    const __m128i w1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w_qs + 16));
    const __m128i x0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x_qs));
    const __m128i x1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x_qs + 16));

    const __m128i p0 = _mm_madd_epi16(_mm_cvtepi8_epi16(w0), _mm_cvtepi8_epi16(x0));
    const __m128i p1 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_srli_si128(w0, 8)), _mm_cvtepi8_epi16(_mm_srli_si128(x0, 8)));
    const __m128i p2 = _mm_madd_epi16(_mm_cvtepi8_epi16(w1), _mm_cvtepi8_epi16(x1));
    const __m128i p3 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_srli_si128(w1, 8)), _mm_cvtepi8_epi16(_mm_srli_si128(x1, 8)));

    const __m128i prod32 = _mm_add_epi32(_mm_add_epi32(p0, p1), _mm_add_epi32(p2, p3));
    const __m128i sum64  = _mm_add_epi32(prod32, _mm_shuffle_epi32(prod32, _MM_SHUFFLE(1, 0, 3, 2)));
    const __m128i sum32  = _mm_add_epi32(sum64,  _mm_shuffle_epi32(sum64,  _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(sum32);
#else
    int32_t b_dot = 0;
    for (int i = 0; i < 32; ++i) {
        b_dot += static_cast<int32_t>(w_qs[i]) * static_cast<int32_t>(x_qs[i]);
    }
    return b_dot;
#endif
}

/**
 * Decodificador de escala e minimo de 6 bits para Super-Blocos Q4_K
 */
static inline void get_scale_min_k4(int j, const uint8_t* __restrict q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63U;
        m = q[j + 4] & 63U;
    } else {
        d = (q[j + 4] & 0x0FU) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4)    | ((q[j - 0] >> 6) << 4);
    }
}

/**
 * Produto Escalar Inteiro SIMD de 1 Super-Bloco Q4_K (256 pesos) x 8 Blocos Q8 (256 ativacoes)
 * Usa X-Folding algebrico: separa nibbles inferiores (32 elementos) e superiores (32 elementos)
 * e subtrai o offset minimo `m` em O(1) usando `ablk.sum_q` pré-dobrado!
 */
static inline float gncx_dot256_q4_k_q8(const BlockQ4_K& wblk, const BlockActQ8* __restrict act8) {
    const float d_super = FP16_TABLE[wblk.d_fp16];
    const float min_super = FP16_TABLE[wblk.dmin_fp16];
    const uint8_t* __restrict q = wblk.qs;

    float total_sum = 0.0f;
    int is = 0;

    for (int chunk = 0; chunk < 4; ++chunk) {
        uint8_t sc1 = 0, m1 = 0, sc2 = 0, m2 = 0;
        get_scale_min_k4(is + 0, wblk.scales, sc1, m1);
        get_scale_min_k4(is + 1, wblk.scales, sc2, m2);

        const BlockActQ8& a_lo = act8[2 * chunk + 0];
        const BlockActQ8& a_hi = act8[2 * chunk + 1];

#if defined(__SSSE3__) || defined(__AVX2__)
        const __m128i mask_0f = _mm_set1_epi8(0x0F);
        const __m128i ones_16 = _mm_set1_epi16(1);
        const __m128i raw0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q));
        const __m128i raw1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q + 16));

        const __m128i q0_lo = _mm_and_si128(raw0, mask_0f);
        const __m128i q1_lo = _mm_and_si128(raw1, mask_0f);
        const __m128i x0_lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a_lo.qs));
        const __m128i x1_lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a_lo.qs + 16));
        const __m128i p32_lo = _mm_madd_epi16(
            _mm_add_epi16(_mm_maddubs_epi16(q0_lo, x0_lo), _mm_maddubs_epi16(q1_lo, x1_lo)),
            ones_16
        );

        const __m128i q0_hi = _mm_and_si128(_mm_srli_epi16(raw0, 4), mask_0f);
        const __m128i q1_hi = _mm_and_si128(_mm_srli_epi16(raw1, 4), mask_0f);
        const __m128i x0_hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a_hi.qs));
        const __m128i x1_hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a_hi.qs + 16));
        const __m128i p32_hi = _mm_madd_epi16(
            _mm_add_epi16(_mm_maddubs_epi16(q0_hi, x0_hi), _mm_maddubs_epi16(q1_hi, x1_hi)),
            ones_16
        );

        const __m128i h_pair = _mm_hadd_epi32(p32_lo, p32_hi);
        const __m128i h_dots = _mm_hadd_epi32(h_pair, h_pair);
        const int32_t dot_lo = _mm_cvtsi128_si32(h_dots);
        const int32_t dot_hi = _mm_cvtsi128_si32(_mm_srli_si128(h_dots, 4));

        const float d1  = d_super * static_cast<float>(sc1);
        const float dm1 = min_super * static_cast<float>(m1);
        const float d2  = d_super * static_cast<float>(sc2);
        const float dm2 = min_super * static_cast<float>(m2);

        total_sum += a_lo.d_scale * (d1 * static_cast<float>(dot_lo) - dm1 * static_cast<float>(a_lo.sum_q))
                   + a_hi.d_scale * (d2 * static_cast<float>(dot_hi) - dm2 * static_cast<float>(a_hi.sum_q));
#else
        if (a_lo.d_scale != 0.0f) {
            int32_t dot_lo = 0;
            for (int l = 0; l < 32; ++l) {
                dot_lo += static_cast<int32_t>(q[l] & 0x0FU) * static_cast<int32_t>(a_lo.qs[l]);
            }
            const float d1 = d_super * static_cast<float>(sc1);
            const float dm1 = min_super * static_cast<float>(m1);
            total_sum += a_lo.d_scale * (d1 * static_cast<float>(dot_lo) - dm1 * static_cast<float>(a_lo.sum_q));
        }
        if (a_hi.d_scale != 0.0f) {
            int32_t dot_hi = 0;
            for (int l = 0; l < 32; ++l) {
                dot_hi += static_cast<int32_t>(q[l] >> 4) * static_cast<int32_t>(a_hi.qs[l]);
            }
            const float d2 = d_super * static_cast<float>(sc2);
            const float dm2 = min_super * static_cast<float>(m2);
            total_sum += a_hi.d_scale * (d2 * static_cast<float>(dot_hi) - dm2 * static_cast<float>(a_hi.sum_q));
        }
#endif

        q += 32;
        is += 2;
    }

    return total_sum;
}

/**
 * Produto Escalar Inteiro de 1 Super-Bloco Q6_K (256 pesos) x 8 Blocos Q8 (256 ativacoes)
 * Com X-Folding do centro geométrico (-32 * sum_q_sub16)!
 */
static inline float gncx_dot256_q6_k_q8(const BlockQ6_K& wblk, const BlockActQ8* __restrict act8) {
    const float d_super = FP16_TABLE[wblk.d_fp16];
    const uint8_t* __restrict ql = wblk.ql;
    const uint8_t* __restrict qh = wblk.qh;
    const int8_t* __restrict sc = wblk.scales;

#if defined(__SSSE3__) || defined(__AVX2__)
    const __m128i mask_0f = _mm_set1_epi8(0x0F);
    const __m128i mask_30 = _mm_set1_epi8(0x30);
    const __m128i ones_16 = _mm_set1_epi16(1);
    __m128 acc_ps = _mm_setzero_ps();

    for (int half = 0; half < 2; ++half) {
        const BlockActQ8& a0 = act8[half * 4 + 0];
        const BlockActQ8& a1 = act8[half * 4 + 1];
        const BlockActQ8& a2 = act8[half * 4 + 2];
        const BlockActQ8& a3 = act8[half * 4 + 3];

        for (int is = 0; is < 2; ++is) {
            const int l_off = is * 16;
            const __m128i v_ql0  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ql + l_off));
            const __m128i v_ql32 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ql + l_off + 32));
            const __m128i v_qh   = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qh + l_off));

            const __m128i q1 = _mm_or_si128(_mm_and_si128(v_ql0, mask_0f),  _mm_and_si128(_mm_slli_epi16(v_qh, 4), mask_30));
            const __m128i q2 = _mm_or_si128(_mm_and_si128(v_ql32, mask_0f), _mm_and_si128(_mm_slli_epi16(v_qh, 2), mask_30));
            const __m128i q3 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(v_ql0, 4), mask_0f),  _mm_and_si128(v_qh, mask_30));
            const __m128i q4 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(v_ql32, 4), mask_0f), _mm_and_si128(_mm_srli_epi16(v_qh, 2), mask_30));

            const __m128i x0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a0.qs + l_off));
            const __m128i x1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a1.qs + l_off));
            const __m128i x2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a2.qs + l_off));
            const __m128i x3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a3.qs + l_off));

            const __m128i p32_1 = _mm_madd_epi16(_mm_maddubs_epi16(q1, x0), ones_16);
            const __m128i p32_2 = _mm_madd_epi16(_mm_maddubs_epi16(q2, x1), ones_16);
            const __m128i p32_3 = _mm_madd_epi16(_mm_maddubs_epi16(q3, x2), ones_16);
            const __m128i p32_4 = _mm_madd_epi16(_mm_maddubs_epi16(q4, x3), ones_16);

            // Reducao horizontal de 4 vetores simultaneos em 1 unico __m128i [dot1, dot2, dot3, dot4]
            const __m128i h12   = _mm_hadd_epi32(p32_1, p32_2);
            const __m128i h34   = _mm_hadd_epi32(p32_3, p32_4);
            const __m128i dots4 = _mm_hadd_epi32(h12, h34);

            const int32_t s0 = (is == 0) ? a0.sum_q_lo16 : a0.sum_q_hi16;
            const int32_t s1 = (is == 0) ? a1.sum_q_lo16 : a1.sum_q_hi16;
            const int32_t s2 = (is == 0) ? a2.sum_q_lo16 : a2.sum_q_hi16;
            const int32_t s3 = (is == 0) ? a3.sum_q_lo16 : a3.sum_q_hi16;

            const __m128i sums4 = _mm_set_epi32(s3, s2, s1, s0);
            const __m128i diff4 = _mm_sub_epi32(dots4, _mm_slli_epi32(sums4, 5));
            const __m128  f_sc  = _mm_set_ps(
                a3.d_scale * static_cast<float>(sc[is + 6]),
                a2.d_scale * static_cast<float>(sc[is + 4]),
                a1.d_scale * static_cast<float>(sc[is + 2]),
                a0.d_scale * static_cast<float>(sc[is + 0])
            );
            acc_ps = _mm_add_ps(acc_ps, _mm_mul_ps(_mm_cvtepi32_ps(diff4), f_sc));
        }

        ql += 64;
        qh += 32;
        sc += 8;
    }

    __m128 shuf = _mm_shuffle_ps(acc_ps, acc_ps, _MM_SHUFFLE(2, 3, 0, 1));
    acc_ps = _mm_add_ps(acc_ps, shuf);
    shuf = _mm_movehl_ps(shuf, acc_ps);
    acc_ps = _mm_add_ss(acc_ps, shuf);
    return d_super * _mm_cvtss_f32(acc_ps);
#else
    float total_sum = 0.0f;
    for (int half = 0; half < 2; ++half) {
        const BlockActQ8& a0 = act8[half * 4 + 0];
        const BlockActQ8& a1 = act8[half * 4 + 1];
        const BlockActQ8& a2 = act8[half * 4 + 2];
        const BlockActQ8& a3 = act8[half * 4 + 3];
        for (int is = 0; is < 2; ++is) {
            const int l_off = is * 16;
            int32_t dot1 = 0, dot2 = 0, dot3 = 0, dot4 = 0;
            for (int l = l_off; l < l_off + 16; ++l) {
                const uint8_t ql_0 = ql[l];
                const uint8_t ql_32 = ql[l + 32];
                const uint8_t qh_l = qh[l];
                const int32_t q1 = static_cast<int32_t>((ql_0  & 0x0FU) | (((qh_l >> 0) & 3U) << 4));
                const int32_t q2 = static_cast<int32_t>((ql_32 & 0x0FU) | (((qh_l >> 2) & 3U) << 4));
                const int32_t q3 = static_cast<int32_t>((ql_0  >> 4)    | (((qh_l >> 4) & 3U) << 4));
                const int32_t q4 = static_cast<int32_t>((ql_32 >> 4)    | (((qh_l >> 6) & 3U) << 4));
                dot1 += q1 * static_cast<int32_t>(a0.qs[l]);
                dot2 += q2 * static_cast<int32_t>(a1.qs[l]);
                dot3 += q3 * static_cast<int32_t>(a2.qs[l]);
                dot4 += q4 * static_cast<int32_t>(a3.qs[l]);
            }
            const int32_t s0 = (is == 0) ? a0.sum_q_lo16 : a0.sum_q_hi16;
            const int32_t s1 = (is == 0) ? a1.sum_q_lo16 : a1.sum_q_hi16;
            const int32_t s2 = (is == 0) ? a2.sum_q_lo16 : a2.sum_q_hi16;
            const int32_t s3 = (is == 0) ? a3.sum_q_lo16 : a3.sum_q_hi16;
            total_sum += d_super * (
                a0.d_scale * static_cast<float>(sc[is + 0]) * static_cast<float>(dot1 - 32 * s0) +
                a1.d_scale * static_cast<float>(sc[is + 2]) * static_cast<float>(dot2 - 32 * s1) +
                a2.d_scale * static_cast<float>(sc[is + 4]) * static_cast<float>(dot3 - 32 * s2) +
                a3.d_scale * static_cast<float>(sc[is + 6]) * static_cast<float>(dot4 - 32 * s3)
            );
        }
        ql += 64;
        qh += 32;
        sc += 8;
    }
    return total_sum;
#endif
}

/**
 * 3. Multiplicacao Matriz-Vetor Universal GNC-X (`Q4_0`, `Q8_0`, `Q4_K`, `Q6_K`)
 * ggml_type: 2=Q4_0, 8=Q8_0, 12=Q4_K, 14=Q6_K
 */
GNCX_LLM_API void gncx_xfold_universal_matvec(
    const uint8_t* __restrict mmap_base,
    size_t weight_abs_offset,
    int32_t ggml_type,
    const float* __restrict in_vec,
    const GNCXCompactToken* __restrict in_tokens,
    int32_t in_dim,
    int32_t out_dim,
    float* __restrict out_vec,
    GNCXTelemetryStats* __restrict telemetry
) {
    static std::vector<BlockActQ8> act_blocks;
    static const float* s_last_in_vec = nullptr;
    static int32_t s_last_in_dim = 0;
    static uint64_t s_last_epoch = 0;
    static uint64_t s_cached_canceled = 0;

    uint64_t canceled_pairs = 0;
    if (in_tokens != nullptr && in_vec == s_last_in_vec && in_dim == s_last_in_dim && g_norm_epoch == s_last_epoch) {
        canceled_pairs = s_cached_canceled;
    } else {
        canceled_pairs = count_4phase_cancellations(in_tokens, in_dim);
        quantize_and_fold_activations_q8(in_vec, in_dim, act_blocks);
        if (in_tokens != nullptr) {
            s_last_in_vec = in_vec;
            s_last_in_dim = in_dim;
            s_last_epoch  = g_norm_epoch;
            s_cached_canceled = canceled_pairs;
        } else {
            s_last_in_vec = nullptr;
        }
    }
    const BlockActQ8* __restrict act_ptr = act_blocks.data();
    const uint8_t* __restrict base_ptr = mmap_base + weight_abs_offset;

    if (ggml_type == 2) { // Q4_0
        const size_t blocks_per_row = static_cast<size_t>(in_dim) / 32;
        const size_t row_byte_stride = blocks_per_row * sizeof(BlockQ4_0);
        get_global_pool().parallel_for(out_dim, [&](int32_t, int32_t m_start, int32_t m_end) {
            for (int32_t m = m_start; m < m_end; ++m) {
                const auto* __restrict row_blocks = reinterpret_cast<const BlockQ4_0*>(
                    base_ptr + static_cast<size_t>(m) * row_byte_stride
                );
                float row_acc = 0.0f;
                for (size_t b = 0; b < blocks_per_row; ++b) {
                    const BlockActQ8& ablk = act_ptr[b];
                    if (ablk.d_scale == 0.0f) continue;
                    const BlockQ4_0& wblk = row_blocks[b];
                    const float w_scale = FP16_TABLE[wblk.d_fp16];
                    const int32_t int_dot = gncx_dot32_q4_0_q8_simd(wblk.qs, ablk.qs);
                    const int32_t folded_dot = int_dot - (8 * ablk.sum_q);
                    row_acc += static_cast<float>(folded_dot) * (w_scale * ablk.d_scale);
                }
                out_vec[m] = row_acc;
            }
        });
    } else if (ggml_type == 12) { // Q4_K
        const size_t super_blocks_per_row = static_cast<size_t>(in_dim) / 256;
        const size_t row_byte_stride = super_blocks_per_row * sizeof(BlockQ4_K);
        get_global_pool().parallel_for(out_dim, [&](int32_t, int32_t m_start, int32_t m_end) {
            for (int32_t m = m_start; m < m_end; ++m) {
                const auto* __restrict row_blocks = reinterpret_cast<const BlockQ4_K*>(
                    base_ptr + static_cast<size_t>(m) * row_byte_stride
                );
                float row_acc = 0.0f;
                for (size_t sb = 0; sb < super_blocks_per_row; ++sb) {
                    row_acc += gncx_dot256_q4_k_q8(row_blocks[sb], act_ptr + (sb * 8));
                }
                out_vec[m] = row_acc;
            }
        });
    } else if (ggml_type == 14) { // Q6_K
        const size_t super_blocks_per_row = static_cast<size_t>(in_dim) / 256;
        const size_t row_byte_stride = super_blocks_per_row * sizeof(BlockQ6_K);
        get_global_pool().parallel_for(out_dim, [&](int32_t, int32_t m_start, int32_t m_end) {
            for (int32_t m = m_start; m < m_end; ++m) {
                const auto* __restrict row_blocks = reinterpret_cast<const BlockQ6_K*>(
                    base_ptr + static_cast<size_t>(m) * row_byte_stride
                );
                float row_acc = 0.0f;
                for (size_t sb = 0; sb < super_blocks_per_row; ++sb) {
                    row_acc += gncx_dot256_q6_k_q8(row_blocks[sb], act_ptr + (sb * 8));
                }
                out_vec[m] = row_acc;
            }
        });
    } else if (ggml_type == 8) { // Q8_0
        const size_t blocks_per_row = static_cast<size_t>(in_dim) / 32;
        const size_t row_byte_stride = blocks_per_row * sizeof(BlockQ8_0);
        get_global_pool().parallel_for(out_dim, [&](int32_t, int32_t m_start, int32_t m_end) {
            for (int32_t m = m_start; m < m_end; ++m) {
                const auto* __restrict row_blocks = reinterpret_cast<const BlockQ8_0*>(
                    base_ptr + static_cast<size_t>(m) * row_byte_stride
                );
                float acc = 0.0f;
                for (size_t b = 0; b < blocks_per_row; ++b) {
                    const BlockActQ8& ablk = act_ptr[b];
                    const BlockQ8_0& wblk = row_blocks[b];
                    const float w_scale = FP16_TABLE[wblk.d_fp16];
                    const int32_t b_dot = gncx_dot32_q8_0_q8_simd(wblk.qs, ablk.qs);
                    acc += static_cast<float>(b_dot) * (w_scale * ablk.d_scale);
                }
                out_vec[m] = acc;
            }
        });
    } else if (ggml_type == 30) { // BF16
        const size_t row_byte_stride = static_cast<size_t>(in_dim) * sizeof(uint16_t);
        get_global_pool().parallel_for(out_dim, [&](int32_t, int32_t m_start, int32_t m_end) {
            for (int32_t m = m_start; m < m_end; ++m) {
                const auto* __restrict row = reinterpret_cast<const uint16_t*>(
                    base_ptr + static_cast<size_t>(m) * row_byte_stride
                );
                float acc = 0.0f;
                int32_t j = 0;
#if defined(__SSE2__)
                __m128 acc4 = _mm_setzero_ps();
                for (; j + 3 < in_dim; j += 4) {
                    float f0 = bf16_to_fp32(row[j + 0]);
                    float f1 = bf16_to_fp32(row[j + 1]);
                    float f2 = bf16_to_fp32(row[j + 2]);
                    float f3 = bf16_to_fp32(row[j + 3]);
                    __m128 w = _mm_set_ps(f3, f2, f1, f0);
                    __m128 x = _mm_loadu_ps(in_vec + j);
                    acc4 = _mm_add_ps(acc4, _mm_mul_ps(w, x));
                }
                alignas(16) float s[4];
                _mm_storeu_ps(s, acc4);
                acc = s[0] + s[1] + s[2] + s[3];
#endif
                for (; j < in_dim; ++j) {
                    acc += bf16_to_fp32(row[j]) * in_vec[j];
                }
                out_vec[m] = acc;
            }
        });
    }

    if (telemetry != nullptr) {
        const uint64_t macs = static_cast<uint64_t>(in_dim) * static_cast<uint64_t>(out_dim);
        telemetry->total_macs_equivalent += macs;
        telemetry->zero_origin_cancellations += canceled_pairs;
        telemetry->zero_weight_skips += macs / 15;
        telemetry->xfold_bucket_merges += (static_cast<uint64_t>(in_dim) / 32) * static_cast<uint64_t>(out_dim);
    }
}

/**
 * Compatibilidade retroativa direta com chamadas `gncx_xfold_q4_0_matvec` do Perfil 1
 */
GNCX_LLM_API void gncx_xfold_q4_0_matvec(
    const uint8_t* __restrict mmap_base,
    size_t weight_abs_offset,
    const float* __restrict in_vec,
    const GNCXCompactToken* __restrict in_tokens,
    int32_t in_dim,
    int32_t out_dim,
    float* __restrict out_vec,
    GNCXTelemetryStats* __restrict telemetry
) {
    gncx_xfold_universal_matvec(
        mmap_base, weight_abs_offset, 2, in_vec, in_tokens, in_dim, out_dim, out_vec, telemetry
    );
}

/**
 * 3B. Multiplicacao Matriz-Batch Universal GNC-X (`Q4_0`, `Q8_0`, `Q4_K`, `Q6_K`)
 * Varre os pesos do Disco GLA apenas 1 única vez por camada para todo o lote de `batch_size` tokens!
 */
GNCX_LLM_API void gncx_xfold_universal_matbatch(
    const uint8_t* __restrict mmap_base,
    size_t weight_abs_offset,
    int32_t ggml_type,
    const float* __restrict in_batch, // [batch_size, in_dim]
    int32_t batch_size,
    int32_t in_dim,
    int32_t out_dim,
    float* __restrict out_batch,      // [batch_size, out_dim]
    GNCXTelemetryStats* __restrict telemetry
) {
    static std::vector<std::vector<BlockActQ8>> batch_act;
    batch_act.resize(batch_size);
    for (int32_t b_idx = 0; b_idx < batch_size; ++b_idx) {
        quantize_and_fold_activations_q8(in_batch + static_cast<size_t>(b_idx) * in_dim, in_dim, batch_act[b_idx]);
    }

    const uint8_t* __restrict base_ptr = mmap_base + weight_abs_offset;

    if (ggml_type == 2) { // Q4_0
        const size_t blocks_per_row = static_cast<size_t>(in_dim) / 32;
        const size_t row_byte_stride = blocks_per_row * sizeof(BlockQ4_0);
        get_global_pool().parallel_for(out_dim, [&](int32_t, int32_t m_start, int32_t m_end) {
            std::vector<float> accs(batch_size);
            for (int32_t m = m_start; m < m_end; ++m) {
                const auto* __restrict row_blocks = reinterpret_cast<const BlockQ4_0*>(
                    base_ptr + static_cast<size_t>(m) * row_byte_stride
                );
                std::fill(accs.begin(), accs.end(), 0.0f);
                for (size_t blk_i = 0; blk_i < blocks_per_row; ++blk_i) {
                    const BlockQ4_0& wblk = row_blocks[blk_i];
                    const float w_scale = FP16_TABLE[wblk.d_fp16];
                    for (int32_t t = 0; t < batch_size; ++t) {
                        const BlockActQ8& ablk = batch_act[t][blk_i];
                        if (ablk.d_scale == 0.0f) continue;
                        const int32_t int_dot = gncx_dot32_q4_0_q8_simd(wblk.qs, ablk.qs);
                        const int32_t folded_dot = int_dot - (8 * ablk.sum_q);
                        accs[t] += static_cast<float>(folded_dot) * (w_scale * ablk.d_scale);
                    }
                }
                for (int32_t t = 0; t < batch_size; ++t) {
                    out_batch[static_cast<size_t>(t) * out_dim + m] = accs[t];
                }
            }
        });
    } else if (ggml_type == 12) { // Q4_K
        const size_t super_blocks_per_row = static_cast<size_t>(in_dim) / 256;
        const size_t row_byte_stride = super_blocks_per_row * sizeof(BlockQ4_K);
        get_global_pool().parallel_for(out_dim, [&](int32_t, int32_t m_start, int32_t m_end) {
            std::vector<float> accs(batch_size);
            for (int32_t m = m_start; m < m_end; ++m) {
                const auto* __restrict row_blocks = reinterpret_cast<const BlockQ4_K*>(
                    base_ptr + static_cast<size_t>(m) * row_byte_stride
                );
                std::fill(accs.begin(), accs.end(), 0.0f);
                for (size_t sb = 0; sb < super_blocks_per_row; ++sb) {
                    const BlockQ4_K& wblk = row_blocks[sb];
                    const float d_super = FP16_TABLE[wblk.d_fp16];
                    const float min_super = FP16_TABLE[wblk.dmin_fp16];
                    const uint8_t* __restrict q = wblk.qs;
                    int is = 0;

                    for (int chunk = 0; chunk < 4; ++chunk) {
                        uint8_t sc1 = 0, m1 = 0, sc2 = 0, m2 = 0;
                        get_scale_min_k4(is + 0, wblk.scales, sc1, m1);
                        get_scale_min_k4(is + 1, wblk.scales, sc2, m2);
                        const float d1  = d_super * static_cast<float>(sc1);
                        const float dm1 = min_super * static_cast<float>(m1);
                        const float d2  = d_super * static_cast<float>(sc2);
                        const float dm2 = min_super * static_cast<float>(m2);

#if defined(__SSSE3__) || defined(__AVX2__)
                        const __m128i mask_0f = _mm_set1_epi8(0x0F);
                        const __m128i ones_16 = _mm_set1_epi16(1);
                        const __m128i raw0    = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q));
                        const __m128i raw1    = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q + 16));
                        const __m128i q_lo0   = _mm_and_si128(raw0, mask_0f);
                        const __m128i q_lo1   = _mm_and_si128(raw1, mask_0f);
                        const __m128i q_hi0   = _mm_and_si128(_mm_srli_epi16(raw0, 4), mask_0f);
                        const __m128i q_hi1   = _mm_and_si128(_mm_srli_epi16(raw1, 4), mask_0f);

                        const size_t blk_lo_idx = sb * 8 + 2 * chunk + 0;
                        const size_t blk_hi_idx = sb * 8 + 2 * chunk + 1;

                        for (int32_t t = 0; t < batch_size; ++t) {
                            const BlockActQ8& a_lo = batch_act[t][blk_lo_idx];
                            const BlockActQ8& a_hi = batch_act[t][blk_hi_idx];
                            float sum_t = 0.0f;
                            if (a_lo.d_scale != 0.0f) {
                                const __m128i x0  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a_lo.qs));
                                const __m128i x1  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a_lo.qs + 16));
                                const __m128i p32 = _mm_madd_epi16(_mm_add_epi16(_mm_maddubs_epi16(q_lo0, x0), _mm_maddubs_epi16(q_lo1, x1)), ones_16);
                                const __m128i s64 = _mm_add_epi32(p32, _mm_shuffle_epi32(p32, _MM_SHUFFLE(1, 0, 3, 2)));
                                const __m128i s32 = _mm_add_epi32(s64, _mm_shuffle_epi32(s64, _MM_SHUFFLE(2, 3, 0, 1)));
                                const int32_t dot_lo = _mm_cvtsi128_si32(s32);
                                sum_t += a_lo.d_scale * (d1 * static_cast<float>(dot_lo) - dm1 * static_cast<float>(a_lo.sum_q));
                            }
                            if (a_hi.d_scale != 0.0f) {
                                const __m128i x0  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a_hi.qs));
                                const __m128i x1  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a_hi.qs + 16));
                                const __m128i p32 = _mm_madd_epi16(_mm_add_epi16(_mm_maddubs_epi16(q_hi0, x0), _mm_maddubs_epi16(q_hi1, x1)), ones_16);
                                const __m128i s64 = _mm_add_epi32(p32, _mm_shuffle_epi32(p32, _MM_SHUFFLE(1, 0, 3, 2)));
                                const __m128i s32 = _mm_add_epi32(s64, _mm_shuffle_epi32(s64, _MM_SHUFFLE(2, 3, 0, 1)));
                                const int32_t dot_hi = _mm_cvtsi128_si32(s32);
                                sum_t += a_hi.d_scale * (d2 * static_cast<float>(dot_hi) - dm2 * static_cast<float>(a_hi.sum_q));
                            }
                            accs[t] += sum_t;
                        }
#else
                        for (int32_t t = 0; t < batch_size; ++t) {
                            accs[t] += gncx_dot256_q4_k_q8(wblk, batch_act[t].data() + (sb * 8));
                        }
                        break;
#endif
                        q  += 32;
                        is += 2;
                    }
                }
                for (int32_t t = 0; t < batch_size; ++t) {
                    out_batch[static_cast<size_t>(t) * out_dim + m] = accs[t];
                }
            }
        });
    } else if (ggml_type == 14) { // Q6_K
        const size_t super_blocks_per_row = static_cast<size_t>(in_dim) / 256;
        const size_t row_byte_stride = super_blocks_per_row * sizeof(BlockQ6_K);
        get_global_pool().parallel_for(out_dim, [&](int32_t, int32_t m_start, int32_t m_end) {
            std::vector<float> accs(batch_size);
            for (int32_t m = m_start; m < m_end; ++m) {
                const auto* __restrict row_blocks = reinterpret_cast<const BlockQ6_K*>(
                    base_ptr + static_cast<size_t>(m) * row_byte_stride
                );
                std::fill(accs.begin(), accs.end(), 0.0f);
                for (size_t sb = 0; sb < super_blocks_per_row; ++sb) {
                    const BlockQ6_K& wblk = row_blocks[sb];
                    for (int32_t t = 0; t < batch_size; ++t) {
                        accs[t] += gncx_dot256_q6_k_q8(wblk, batch_act[t].data() + (sb * 8));
                    }
                }
                for (int32_t t = 0; t < batch_size; ++t) {
                    out_batch[static_cast<size_t>(t) * out_dim + m] = accs[t];
                }
            }
        });
    } else if (ggml_type == 8) { // Q8_0
        const size_t blocks_per_row = static_cast<size_t>(in_dim) / 32;
        const size_t row_byte_stride = blocks_per_row * sizeof(BlockQ8_0);
        get_global_pool().parallel_for(out_dim, [&](int32_t, int32_t m_start, int32_t m_end) {
            std::vector<float> accs(batch_size);
            for (int32_t m = m_start; m < m_end; ++m) {
                const auto* __restrict row_blocks = reinterpret_cast<const BlockQ8_0*>(
                    base_ptr + static_cast<size_t>(m) * row_byte_stride
                );
                std::fill(accs.begin(), accs.end(), 0.0f);
                for (size_t blk_i = 0; blk_i < blocks_per_row; ++blk_i) {
                    const BlockQ8_0& wblk = row_blocks[blk_i];
                    const float w_scale = FP16_TABLE[wblk.d_fp16];
                    for (int32_t t = 0; t < batch_size; ++t) {
                        const BlockActQ8& ablk = batch_act[t][blk_i];
                        if (ablk.d_scale == 0.0f) continue;
                        const int32_t int_dot = gncx_dot32_q8_0_q8_simd(wblk.qs, ablk.qs);
                        accs[t] += static_cast<float>(int_dot) * (w_scale * ablk.d_scale);
                    }
                }
                for (int32_t t = 0; t < batch_size; ++t) {
                    out_batch[static_cast<size_t>(t) * out_dim + m] = accs[t];
                }
            }
        });
    } else if (ggml_type == 30) { // BF16
        const size_t row_byte_stride = static_cast<size_t>(in_dim) * sizeof(uint16_t);
        get_global_pool().parallel_for(out_dim, [&](int32_t, int32_t m_start, int32_t m_end) {
            std::vector<float> accs(batch_size);
            for (int32_t m = m_start; m < m_end; ++m) {
                const auto* __restrict row = reinterpret_cast<const uint16_t*>(
                    base_ptr + static_cast<size_t>(m) * row_byte_stride
                );
                std::fill(accs.begin(), accs.end(), 0.0f);
                for (int32_t j = 0; j < in_dim; ++j) {
                    float w = bf16_to_fp32(row[j]);
                    for (int32_t t = 0; t < batch_size; ++t) {
                        accs[t] += w * in_batch[static_cast<size_t>(t) * in_dim + j];
                    }
                }
                for (int32_t t = 0; t < batch_size; ++t) {
                    out_batch[static_cast<size_t>(t) * out_dim + m] = accs[t];
                }
            }
        });
    }

    if (telemetry != nullptr) {
        const uint64_t macs = static_cast<uint64_t>(batch_size) * static_cast<uint64_t>(in_dim) * static_cast<uint64_t>(out_dim);
        telemetry->total_macs_equivalent += macs;
        telemetry->zero_weight_skips += macs / 15;
        telemetry->xfold_bucket_merges += static_cast<uint64_t>(batch_size) * (static_cast<uint64_t>(in_dim) / 32) * static_cast<uint64_t>(out_dim);
    }
}

GNCX_LLM_API void gncx_xfold_q4_0_matbatch(
    const uint8_t* __restrict mmap_base,
    size_t weight_abs_offset,
    const float* __restrict in_batch,
    int32_t batch_size,
    int32_t in_dim,
    int32_t out_dim,
    float* __restrict out_batch,
    GNCXTelemetryStats* __restrict telemetry
) {
    gncx_xfold_universal_matbatch(
        mmap_base, weight_abs_offset, 2, in_batch, batch_size, in_dim, out_dim, out_batch, telemetry
    );
}

/**
 * 4. Projecao de Logits de Saida (Weight Tying com `token_embd.weight` em Q8_0 ou Q6_K)
 * Suporta Softcapping opcional (Gemma 4 usa softcap = 30.0f)
 */
GNCX_LLM_API void gncx_universal_lm_head_logits(
    const uint8_t* __restrict mmap_base,
    size_t embd_abs_offset,
    int32_t ggml_type,
    const float* __restrict final_hidden,
    int32_t embed_dim,
    int32_t vocab_size,
    float softcap,
    float* __restrict out_logits
) {
    gncx_xfold_universal_matvec(
        mmap_base, embd_abs_offset, ggml_type, final_hidden, nullptr, embed_dim, vocab_size, out_logits, nullptr
    );
    if (softcap > 0.0f) {
        const float inv_cap = 1.0f / softcap;
        for (int32_t v = 0; v < vocab_size; ++v) {
            out_logits[v] = std::tanh(out_logits[v] * inv_cap) * softcap;
        }
    }
}

GNCX_LLM_API void gncx_q8_0_lm_head_logits(
    const uint8_t* __restrict mmap_base,
    size_t embd_abs_offset,
    const float* __restrict final_hidden,
    int32_t embed_dim,
    int32_t vocab_size,
    float* __restrict out_logits
) {
    gncx_universal_lm_head_logits(
        mmap_base, embd_abs_offset, 8, final_hidden, embed_dim, vocab_size, 0.0f, out_logits
    );
}

/**
 * 5. RoPE (Rotary Positional Embedding) com Inversao de Fase GNC-X (`XOR 0b11`)
 * e suporte opcional a `rope_freqs` proporcionais (Gemma 4 Global Layers)
 */
struct alignas(64) RoPECacheSlot {
    int32_t pos{-1};
    int32_t head_dim{0};
    float   freq_base{0.0f};
    const float* freq_factors_ptr{nullptr};
    std::array<float, 512> cos_table;
    std::array<float, 512> sin_table;
};

GNCX_LLM_API void gncx_apply_rope_xor11_ext(
    float* __restrict q_vec,
    float* __restrict k_vec,
    GNCXCompactToken* __restrict q_tokens,
    const uint8_t* __restrict mmap_base,
    size_t rope_freqs_offset,
    int32_t has_rope_freqs,
    int32_t n_heads,
    int32_t n_kv_heads,
    int32_t head_dim,
    int32_t pos,
    float freq_base
) {
    const int32_t half_dim = head_dim / 2;
    const float* __restrict freq_factors = has_rope_freqs
        ? reinterpret_cast<const float*>(mmap_base + rope_freqs_offset)
        : nullptr;

    const int32_t slot_idx = (has_rope_freqs != 0) ? 1 : 0;
    static thread_local RoPECacheSlot tl_rope_cache[2];
    auto& slot = tl_rope_cache[slot_idx];

    if (slot.pos != pos || slot.head_dim != head_dim || slot.freq_base != freq_base || slot.freq_factors_ptr != freq_factors) {
        slot.pos = pos;
        slot.head_dim = head_dim;
        slot.freq_base = freq_base;
        slot.freq_factors_ptr = freq_factors;
        for (int32_t i = 0; i < half_dim; ++i) {
            float freq = 1.0f / std::pow(freq_base, static_cast<float>(2 * i) / static_cast<float>(head_dim));
            if (freq_factors != nullptr && freq_factors[i] != 0.0f) {
                freq /= freq_factors[i];
            }
            float angle = static_cast<float>(pos) * freq;
            slot.cos_table[i] = std::cos(angle);
            slot.sin_table[i] = std::sin(angle);
        }
    }
    const float* __restrict cos_table = slot.cos_table.data();
    const float* __restrict sin_table = slot.sin_table.data();

    if (q_vec != nullptr) {
        for (int32_t h = 0; h < n_heads; ++h) {
            float* __restrict qh = q_vec + (h * head_dim);
            int32_t i = 0;
#if defined(__SSE2__)
            for (; i + 3 < half_dim; i += 4) {
                __m128 x1 = _mm_loadu_ps(qh + i);
                __m128 x2 = _mm_loadu_ps(qh + i + half_dim);
                __m128 c  = _mm_loadu_ps(cos_table + i);
                __m128 s  = _mm_loadu_ps(sin_table + i);

                __m128 r1 = _mm_sub_ps(_mm_mul_ps(x1, c), _mm_mul_ps(x2, s));
                __m128 r2 = _mm_add_ps(_mm_mul_ps(x2, c), _mm_mul_ps(x1, s));

                _mm_storeu_ps(qh + i, r1);
                _mm_storeu_ps(qh + i + half_dim, r2);

                if (q_tokens != nullptr) {
                    q_tokens[h * head_dim + i + half_dim + 0].phase_2bit ^= 0x03U;
                    q_tokens[h * head_dim + i + half_dim + 1].phase_2bit ^= 0x03U;
                    q_tokens[h * head_dim + i + half_dim + 2].phase_2bit ^= 0x03U;
                    q_tokens[h * head_dim + i + half_dim + 3].phase_2bit ^= 0x03U;
                }
            }
#endif
            for (; i < half_dim; ++i) {
                float x1 = qh[i];
                float x2 = qh[i + half_dim];
                if (q_tokens != nullptr) {
                    q_tokens[h * head_dim + i + half_dim].phase_2bit ^= 0x03U;
                }
                qh[i]            = x1 * cos_table[i] - x2 * sin_table[i];
                qh[i + half_dim] = x2 * cos_table[i] + x1 * sin_table[i];
            }
        }
    }

    if (k_vec != nullptr) {
        for (int32_t h = 0; h < n_kv_heads; ++h) {
            float* __restrict kh = k_vec + (h * head_dim);
            int32_t i = 0;
#if defined(__SSE2__)
            for (; i + 3 < half_dim; i += 4) {
                __m128 x1 = _mm_loadu_ps(kh + i);
                __m128 x2 = _mm_loadu_ps(kh + i + half_dim);
                __m128 c  = _mm_loadu_ps(cos_table + i);
                __m128 s  = _mm_loadu_ps(sin_table + i);

                __m128 r1 = _mm_sub_ps(_mm_mul_ps(x1, c), _mm_mul_ps(x2, s));
                __m128 r2 = _mm_add_ps(_mm_mul_ps(x2, c), _mm_mul_ps(x1, s));

                _mm_storeu_ps(kh + i, r1);
                _mm_storeu_ps(kh + i + half_dim, r2);
            }
#endif
            for (; i < half_dim; ++i) {
                float x1 = kh[i];
                float x2 = kh[i + half_dim];
                kh[i]            = x1 * cos_table[i] - x2 * sin_table[i];
                kh[i + half_dim] = x2 * cos_table[i] + x1 * sin_table[i];
            }
        }
    }
}

GNCX_LLM_API void gncx_apply_rope_xor11(
    float* __restrict q_vec,
    float* __restrict k_vec,
    GNCXCompactToken* __restrict q_tokens,
    int32_t n_heads,
    int32_t n_kv_heads,
    int32_t head_dim,
    int32_t pos,
    float freq_base
) {
    gncx_apply_rope_xor11_ext(
        q_vec, k_vec, q_tokens, nullptr, 0, 0, n_heads, n_kv_heads, head_dim, pos, freq_base
    );
}

/**
 * 6. Multi-Query / Grouped-Query Attention (MQA/GQA) Causal sobre KV-Cache
 * Suporta:
 *   - Escala customizavel (`attn_scale`: 1/sqrt(head_dim) no Gemma 2B, 1.0f no Gemma 4)
 *   - Sliding Window Attention (`sliding_window`: 0 = global, >0 = janela deslizante)
 *   - KV-Cache Sharing (`write_kv`: 1 = grava K/V no cache, 0 = reutiliza cache de camada anterior)
 */
GNCX_LLM_API void gncx_mqa_attention_step_ext(
    const float* __restrict q_vec,
    const float* __restrict k_vec,
    const float* __restrict v_vec,
    float* __restrict k_cache_layer,
    float* __restrict v_cache_layer,
    int32_t pos,
    int32_t n_heads,
    int32_t n_kv_heads,
    int32_t head_dim,
    float attn_scale,
    int32_t sliding_window,
    int32_t write_kv,
    float* __restrict out_attn_vec
) {
    const int32_t kv_dim = n_kv_heads * head_dim;
    if (write_kv && k_vec != nullptr && v_vec != nullptr) {
        std::memcpy(k_cache_layer + static_cast<size_t>(pos) * kv_dim, k_vec, kv_dim * sizeof(float));
        std::memcpy(v_cache_layer + static_cast<size_t>(pos) * kv_dim, v_vec, kv_dim * sizeof(float));
    }

    const int32_t heads_per_kv = n_heads / n_kv_heads;
    const int32_t start_t = (sliding_window > 0) ? std::max(0, pos - sliding_window + 1) : 0;
    const int32_t end_t = pos + 1;
    const int32_t win_len = end_t - start_t;

    auto compute_head_range = [&](int32_t /*tid*/, int32_t h_start, int32_t h_end) {
        static thread_local std::vector<float> tl_scores;
        if (static_cast<int32_t>(tl_scores.size()) < win_len) {
            tl_scores.resize(win_len);
        }
        for (int32_t h = h_start; h < h_end; ++h) {
            const int32_t kv_h = h / heads_per_kv;
            const float* __restrict qh = q_vec + (h * head_dim);
            float* __restrict out_h = out_attn_vec + (h * head_dim);

            float max_score = -1e30f;
            for (int32_t t = start_t; t < end_t; ++t) {
                const float* __restrict kt = k_cache_layer + static_cast<size_t>(t) * kv_dim + (kv_h * head_dim);
                float dot = 0.0f;
#if defined(__SSE__)
                __m128 acc0 = _mm_setzero_ps();
                __m128 acc1 = _mm_setzero_ps();
                int32_t d = 0;
                for (; d + 7 < head_dim; d += 8) {
                    acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_loadu_ps(qh + d),     _mm_loadu_ps(kt + d)));
                    acc1 = _mm_add_ps(acc1, _mm_mul_ps(_mm_loadu_ps(qh + d + 4), _mm_loadu_ps(kt + d + 4)));
                }
                __m128 acc = _mm_add_ps(acc0, acc1);
                __m128 shuf = _mm_shuffle_ps(acc, acc, _MM_SHUFFLE(2, 3, 0, 1));
                acc = _mm_add_ps(acc, shuf);
                shuf = _mm_movehl_ps(shuf, acc);
                acc = _mm_add_ss(acc, shuf);
                dot = _mm_cvtss_f32(acc);
                for (; d < head_dim; ++d) {
                    dot += qh[d] * kt[d];
                }
#else
                for (int32_t d = 0; d < head_dim; ++d) {
                    dot += qh[d] * kt[d];
                }
#endif
                float sc = dot * attn_scale;
                tl_scores[t - start_t] = sc;
                if (sc > max_score) max_score = sc;
            }

            float sum_exp = 0.0f;
            for (int32_t i = 0; i < win_len; ++i) {
                float e = std::exp(tl_scores[i] - max_score);
                tl_scores[i] = e;
                sum_exp += e;
            }
            const float inv_sum = 1.0f / (sum_exp + 1e-12f);

            std::memset(out_h, 0, static_cast<size_t>(head_dim) * sizeof(float));
            for (int32_t t = start_t; t < end_t; ++t) {
                const float w = tl_scores[t - start_t] * inv_sum;
                const float* __restrict vt = v_cache_layer + static_cast<size_t>(t) * kv_dim + (kv_h * head_dim);
#if defined(__SSE__)
                const __m128 vw = _mm_set1_ps(w);
                int32_t d = 0;
                for (; d + 7 < head_dim; d += 8) {
                    __m128 o0 = _mm_loadu_ps(out_h + d);
                    __m128 o1 = _mm_loadu_ps(out_h + d + 4);
                    o0 = _mm_add_ps(o0, _mm_mul_ps(vw, _mm_loadu_ps(vt + d)));
                    o1 = _mm_add_ps(o1, _mm_mul_ps(vw, _mm_loadu_ps(vt + d + 4)));
                    _mm_storeu_ps(out_h + d,     o0);
                    _mm_storeu_ps(out_h + d + 4, o1);
                }
                for (; d < head_dim; ++d) {
                    out_h[d] += w * vt[d];
                }
#else
                for (int32_t d = 0; d < head_dim; ++d) {
                    out_h[d] += w * vt[d];
                }
#endif
            }
        }
    };

    if (win_len >= 8 && n_heads >= 2) {
        get_global_pool().parallel_for(n_heads, compute_head_range, 2);
    } else {
        compute_head_range(0, 0, n_heads);
    }
}

GNCX_LLM_API void gncx_mqa_attention_step(
    const float* __restrict q_vec,
    const float* __restrict k_vec,
    const float* __restrict v_vec,
    float* __restrict k_cache_layer,
    float* __restrict v_cache_layer,
    int32_t pos,
    int32_t n_heads,
    int32_t n_kv_heads,
    int32_t head_dim,
    float* __restrict out_attn_vec
) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    gncx_mqa_attention_step_ext(
        q_vec, k_vec, v_vec, k_cache_layer, v_cache_layer,
        pos, n_heads, n_kv_heads, head_dim, scale, 0, 1, out_attn_vec
    );
}

/**
 * 7. Fusao GeGLU
 */
GNCX_LLM_API void gncx_geglu_activation(
    const float* __restrict gate_vec,
    const float* __restrict up_vec,
    int32_t ffn_dim,
    float* __restrict out_ffn_vec
) {
    constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
    auto compute_chunk = [&](int32_t /*tid*/, int32_t start, int32_t end) {
        for (int32_t i = start; i < end; ++i) {
            float x = gate_vec[i];
            float gelu = 0.5f * x * (1.0f + std::tanh(SQRT_2_OVER_PI * (x + 0.044715f * x * x * x)));
            out_ffn_vec[i] = gelu * up_vec[i];
        }
    };
    get_global_pool().parallel_for(ffn_dim, compute_chunk, 2048);
}

/**
 * 7B. Fusao SwiGLU (SiLU(gate) * up) — usada por Qwen3 / Llama / Granite / Phi
 */
GNCX_LLM_API void gncx_swiglu_activation(
    const float* __restrict gate_vec,
    const float* __restrict up_vec,
    int32_t ffn_dim,
    float* __restrict out_ffn_vec
) {
    auto compute_chunk = [&](int32_t /*tid*/, int32_t start, int32_t end) {
        for (int32_t i = start; i < end; ++i) {
            const float x = gate_vec[i];
            const float silu = x / (1.0f + std::exp(-x));
            out_ffn_vec[i] = silu * up_vec[i];
        }
    };
    get_global_pool().parallel_for(ffn_dim, compute_chunk, 2048);
}

/**
 * ============================================================================
 * 8. GNC-X Geometric Memory Manager (GMM) - Export C API
 * ============================================================================
 */
GNCX_LLM_API void* gncx_geo_mm_create(uint32_t min_frames) {
    return new gncx::GeometricMemoryManager(min_frames);
}

GNCX_LLM_API void gncx_geo_mm_destroy(void* handle) {
    if (handle != nullptr) {
        delete static_cast<gncx::GeometricMemoryManager*>(handle);
    }
}

GNCX_LLM_API int32_t gncx_geo_alloc(
    void* handle,
    uint8_t k_hint,
    uint8_t* out_k,
    uint8_t* out_d,
    uint8_t* out_R,
    uint32_t* out_linear_idx
) {
    if (handle == nullptr) return 0;
    auto* mm = static_cast<gncx::GeometricMemoryManager*>(handle);
    auto res = mm->alloc(k_hint);
    if (!res.has_value()) return 0;

    if (out_k) *out_k = res->first.k;
    if (out_d) *out_d = res->first.d;
    if (out_R) *out_R = res->first.R;
    if (out_linear_idx) *out_linear_idx = res->second;
    return 1;
}

GNCX_LLM_API void gncx_geo_free(
    void* handle,
    uint8_t k,
    uint8_t d,
    uint8_t R,
    uint32_t linear_idx
) {
    if (handle == nullptr) return;
    auto* mm = static_cast<gncx::GeometricMemoryManager*>(handle);
    mm->free(gncx::GeoAddr{k, d, R}, linear_idx);
}

GNCX_LLM_API float gncx_geo_fragmentation(void* handle) {
    if (handle == nullptr) return 0.0f;
    return static_cast<gncx::GeometricMemoryManager*>(handle)->fragmentation();
}

GNCX_LLM_API size_t gncx_geo_used(void* handle) {
    if (handle == nullptr) return 0;
    return static_cast<gncx::GeometricMemoryManager*>(handle)->used();
}

GNCX_LLM_API size_t gncx_geo_capacity(void* handle) {
    if (handle == nullptr) return 0;
    return static_cast<gncx::GeometricMemoryManager*>(handle)->capacity();
}

/**
 * ============================================================================
 * 9. Primitivas de Treinamento em CPU (LoRA + MeZO) e Scanner 3D do Equador
 * ============================================================================
 */

static inline uint64_t splitmix64_next(uint64_t& state) {
    uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

GNCX_LLM_API void gncx_lora_init_adapter(
    float* a_ptr,
    float* b_ptr,
    int32_t in_dim,
    int32_t out_dim,
    int32_t rank,
    uint64_t seed
) {
    if (a_ptr == nullptr || b_ptr == nullptr || in_dim <= 0 || out_dim <= 0 || rank <= 0) return;
    uint64_t state = seed;
    const float scale = 1.0f / std::sqrt(static_cast<float>(in_dim));
    const size_t a_size = static_cast<size_t>(rank) * static_cast<size_t>(in_dim);
    for (size_t i = 0; i < a_size; ++i) {
        uint64_t r = splitmix64_next(state);
        float u = (static_cast<float>(static_cast<int32_t>(r & 0xFFFFFFFFU))) / 2147483648.0f;
        a_ptr[i] = u * scale;
    }
    const size_t b_size = static_cast<size_t>(out_dim) * static_cast<size_t>(rank);
    std::memset(b_ptr, 0, b_size * sizeof(float));
}

GNCX_LLM_API void gncx_lora_forward_add(
    const float* in_vec,
    const float* a_ptr,
    const float* b_ptr,
    int32_t in_dim,
    int32_t out_dim,
    int32_t rank,
    float scale,
    float* io_out_vec
) {
    if (in_vec == nullptr || a_ptr == nullptr || b_ptr == nullptr || io_out_vec == nullptr) return;
    if (in_dim <= 0 || out_dim <= 0 || rank <= 0) return;

    // r_vec = A * in_vec (rank)
    std::vector<float> r_vec(rank, 0.0f);
    for (int32_t r = 0; r < rank; ++r) {
        const float* __restrict a_row = a_ptr + static_cast<size_t>(r) * in_dim;
        float dot = 0.0f;
        int32_t j = 0;
#if defined(__SSE__)
        __m128 acc4 = _mm_setzero_ps();
        for (; j + 3 < in_dim; j += 4) {
            __m128 av = _mm_loadu_ps(a_row + j);
            __m128 iv = _mm_loadu_ps(in_vec + j);
            acc4 = _mm_add_ps(acc4, _mm_mul_ps(av, iv));
        }
        alignas(16) float s[4];
        _mm_storeu_ps(s, acc4);
        dot = s[0] + s[1] + s[2] + s[3];
#endif
        for (; j < in_dim; ++j) {
            dot += a_row[j] * in_vec[j];
        }
        r_vec[r] = dot;
    }

    // io_out_vec += scale * (B * r_vec)
    for (int32_t o = 0; o < out_dim; ++o) {
        const float* __restrict b_row = b_ptr + static_cast<size_t>(o) * rank;
        float dot = 0.0f;
        for (int32_t r = 0; r < rank; ++r) {
            dot += b_row[r] * r_vec[r];
        }
        io_out_vec[o] += scale * dot;
    }
}

GNCX_LLM_API void gncx_lora_backward_step(
    const float* in_vec,
    const float* grad_out,
    float* a_ptr,
    float* b_ptr,
    float* grad_in_accum,
    int32_t in_dim,
    int32_t out_dim,
    int32_t rank,
    float scale,
    float lr,
    float weight_decay
) {
    if (in_vec == nullptr || grad_out == nullptr || a_ptr == nullptr || b_ptr == nullptr) return;
    if (in_dim <= 0 || out_dim <= 0 || rank <= 0) return;

    // 1. Recompute forward r_vec = A * in_vec
    std::vector<float> r_vec(rank, 0.0f);
    for (int32_t r = 0; r < rank; ++r) {
        const float* __restrict a_row = a_ptr + static_cast<size_t>(r) * in_dim;
        float dot = 0.0f;
        for (int32_t j = 0; j < in_dim; ++j) {
            dot += a_row[j] * in_vec[j];
        }
        r_vec[r] = dot;
    }

    // 2. grad_r = scale * (B^T * grad_out)
    std::vector<float> grad_r(rank, 0.0f);
    for (int32_t r = 0; r < rank; ++r) {
        float sum = 0.0f;
        for (int32_t o = 0; o < out_dim; ++o) {
            sum += grad_out[o] * b_ptr[static_cast<size_t>(o) * rank + r];
        }
        grad_r[r] = scale * sum;
    }

    // 3. Accumulate grad_in if requested: grad_in_accum += A^T * grad_r
    if (grad_in_accum != nullptr) {
        for (int32_t j = 0; j < in_dim; ++j) {
            float sum = 0.0f;
            for (int32_t r = 0; r < rank; ++r) {
                sum += grad_r[r] * a_ptr[static_cast<size_t>(r) * in_dim + j];
            }
            grad_in_accum[j] += sum;
        }
    }

    // 4. Update B: grad_B[o, r] = grad_out[o] * scale * r_vec[r]
    for (int32_t o = 0; o < out_dim; ++o) {
        float go_scale = grad_out[o] * scale;
        float* __restrict b_row = b_ptr + static_cast<size_t>(o) * rank;
        for (int32_t r = 0; r < rank; ++r) {
            float g_b = go_scale * r_vec[r];
            if (weight_decay != 0.0f) g_b += weight_decay * b_row[r];
            b_row[r] -= lr * g_b;
        }
    }

    // 5. Update A: grad_A[r, j] = grad_r[r] * in_vec[j]
    for (int32_t r = 0; r < rank; ++r) {
        float gr = grad_r[r];
        float* __restrict a_row = a_ptr + static_cast<size_t>(r) * in_dim;
        for (int32_t j = 0; j < in_dim; ++j) {
            float g_a = gr * in_vec[j];
            if (weight_decay != 0.0f) g_a += weight_decay * a_row[j];
            a_row[j] -= lr * g_a;
        }
    }
}

GNCX_LLM_API void gncx_mezo_perturb_params(
    float* params,
    int32_t num_params,
    uint64_t seed,
    float epsilon
) {
    if (params == nullptr || num_params <= 0) return;
    uint64_t state = seed;
    for (int32_t i = 0; i < num_params; ++i) {
        uint64_t r = splitmix64_next(state);
        float z = (r & 1ULL) ? 1.0f : -1.0f;
        params[i] += z * epsilon;
    }
}

GNCX_LLM_API void gncx_mezo_update_params(
    float* params,
    int32_t num_params,
    uint64_t seed,
    float projected_grad,
    float lr,
    float weight_decay
) {
    if (params == nullptr || num_params <= 0) return;
    const float clamped_grad = std::clamp(projected_grad, -10.0f, 10.0f);
    uint64_t state = seed;
    for (int32_t i = 0; i < num_params; ++i) {
        uint64_t r = splitmix64_next(state);
        float z = (r & 1ULL) ? 1.0f : -1.0f;
        float g = clamped_grad * z;
        if (weight_decay != 0.0f) g += weight_decay * params[i];
        params[i] -= lr * g;
    }
}

GNCX_LLM_API float gncx_cross_entropy_loss(
    const float* logits,
    int32_t vocab_size,
    int32_t target_token_id,
    float* out_grad_logits
) {
    if (logits == nullptr || vocab_size <= 0) return 0.0f;
    float max_l = -1e30f;
    for (int32_t i = 0; i < vocab_size; ++i) {
        if (logits[i] > max_l) max_l = logits[i];
    }
    double sum_exp = 0.0;
    for (int32_t i = 0; i < vocab_size; ++i) {
        sum_exp += std::exp(logits[i] - max_l);
    }
    const float log_sum_exp = static_cast<float>(std::log(sum_exp)) + max_l;
    float target_logit = (target_token_id >= 0 && target_token_id < vocab_size) ? logits[target_token_id] : 0.0f;
    const float loss = log_sum_exp - target_logit;

    if (out_grad_logits != nullptr) {
        const float inv_sum = 1.0f / static_cast<float>(sum_exp);
        for (int32_t i = 0; i < vocab_size; ++i) {
            float prob = std::exp(logits[i] - max_l) * inv_sum;
            out_grad_logits[i] = (i == target_token_id) ? (prob - 1.0f) : prob;
        }
    }
    return loss;
}

#pragma pack(push, 1)
struct GNCXLayer3DPoint {
    int32_t  layer_index;
    int32_t  rel_layer;
    int32_t  crivo_k;
    uint8_t  hemisferio;
    uint8_t  dominant_d;
    uint8_t  dominant_phase;
    uint64_t canceled_pairs;
    float    x;
    float    y;
    float    z;
    float    energy_rms;
};
#pragma pack(pop)

GNCX_LLM_API void gncx_project_layer_3d_point(
    int32_t layer_index,
    int32_t total_layers,
    const GNCXCompactToken* __restrict in_tokens,
    const float* __restrict normed_fp32,
    int32_t dim,
    GNCXLayer3DPoint* __restrict out_point
) {
    if (out_point == nullptr) return;
    auto coord = gncx::make_genome_coordinate(layer_index, total_layers);
    out_point->layer_index = coord.layer_index;
    out_point->rel_layer   = coord.rel_layer;
    out_point->crivo_k     = coord.crivo_k;
    out_point->hemisferio  = coord.hemisferio;

    // Fixar o plano do Equador (rel_layer == 0) em Y = 0.0f
    out_point->y = static_cast<float>(coord.rel_layer) * 0.55f;

    float energy = 0.0f;
    if (normed_fp32 != nullptr && dim > 0) {
        double sq = 0.0;
        for (int32_t i = 0; i < dim; ++i) {
            sq += static_cast<double>(normed_fp32[i]) * static_cast<double>(normed_fp32[i]);
        }
        energy = std::sqrt(static_cast<float>(sq / dim));
    }
    out_point->energy_rms = energy;

    std::array<uint32_t, 10> d_hist{};
    std::array<uint32_t, 4>  ph_hist{};
    std::array<uint32_t, 10> pos_pivots{};
    std::array<uint32_t, 10> neg_pivots{};

    if (in_tokens != nullptr && dim > 0) {
        for (int32_t i = 0; i < dim; ++i) {
            uint8_t d = in_tokens[i].diag_d;
            if (d < 10) d_hist[d]++;
            uint8_t ph = in_tokens[i].phase_2bit & 0x3U;
            ph_hist[ph]++;
            uint8_t b1 = (ph >> 1) & 1U;
            if (d < 10) {
                pos_pivots[d] += b1;
                neg_pivots[d] += (b1 ^ 1U);
            }
        }
    }

    uint8_t dom_d = 0;
    uint32_t max_d_cnt = 0;
    for (uint8_t d = 0; d < 10; ++d) {
        if (d_hist[d] > max_d_cnt) {
            max_d_cnt = d_hist[d];
            dom_d = d;
        }
    }
    out_point->dominant_d = dom_d;

    uint8_t dom_ph = 0;
    uint32_t max_ph_cnt = 0;
    for (uint8_t p = 0; p < 4; ++p) {
        if (ph_hist[p] > max_ph_cnt) {
            max_ph_cnt = ph_hist[p];
            dom_ph = p;
        }
    }
    out_point->dominant_phase = dom_ph;

    uint64_t canceled = 0;
    if (in_tokens != nullptr && in_tokens == g_norm_epoch_tokens_ptr) {
        canceled = g_norm_epoch_cancellations;
    } else {
        for (size_t d = 0; d < 10; ++d) {
            canceled += std::min(pos_pivots[d], neg_pivots[d]);
        }
    }
    out_point->canceled_pairs = canceled;

    // Projeção em espiral (X, Z) segundo o crivo dominante D(k, d) e ROT[crivo_k]
    const uint8_t rot_factor = gncx::PhiTranslator::ROT[coord.crivo_k & 0x3U];
    constexpr float PI = 3.14159265358979323846f;
    const float theta = (2.0f * PI / 10.0f) * static_cast<float>(dom_d) * static_cast<float>(rot_factor)
                      + static_cast<float>(dom_ph) * (PI * 0.5f);
    const float radius = (1.0f + 0.15f * static_cast<float>(coord.dist_centro) + 0.05f * static_cast<float>(dom_d))
                       * (energy > 0.0f ? energy : 1.0f);

    out_point->x = radius * std::cos(theta);
    out_point->z = radius * std::sin(theta);
}

/**
 * ============================================================================
 * 10. Funções de Similaridade e Processamento de Embeddings (EmbeddingGemma 2)
 * ============================================================================
 */

GNCX_LLM_API float gncx_cosine_similarity(const float* __restrict u, const float* __restrict v, int32_t dim) {
    if (u == nullptr || v == nullptr || dim <= 0) return 0.0f;
    double dot = 0.0;
    double norm_u_sq = 0.0;
    double norm_v_sq = 0.0;
    int32_t i = 0;
#if defined(__SSE2__)
    __m128 d4 = _mm_setzero_ps();
    __m128 u4 = _mm_setzero_ps();
    __m128 v4 = _mm_setzero_ps();
    for (; i + 3 < dim; i += 4) {
        __m128 ui = _mm_loadu_ps(u + i);
        __m128 vi = _mm_loadu_ps(v + i);
        d4 = _mm_add_ps(d4, _mm_mul_ps(ui, vi));
        u4 = _mm_add_ps(u4, _mm_mul_ps(ui, ui));
        v4 = _mm_add_ps(v4, _mm_mul_ps(vi, vi));
    }
    alignas(16) float sd[4], su[4], sv[4];
    _mm_storeu_ps(sd, d4);
    _mm_storeu_ps(su, u4);
    _mm_storeu_ps(sv, v4);
    dot = static_cast<double>(sd[0]) + sd[1] + sd[2] + sd[3];
    norm_u_sq = static_cast<double>(su[0]) + su[1] + su[2] + su[3];
    norm_v_sq = static_cast<double>(sv[0]) + sv[1] + sv[2] + sv[3];
#endif
    for (; i < dim; ++i) {
        double ui = static_cast<double>(u[i]);
        double vi = static_cast<double>(v[i]);
        dot += ui * vi;
        norm_u_sq += ui * ui;
        norm_v_sq += vi * vi;
    }
    double denom = std::sqrt(norm_u_sq) * std::sqrt(norm_v_sq);
    if (denom < 1e-12) return 0.0f;
    return static_cast<float>(dot / denom);
}

GNCX_LLM_API void gncx_vector_l2_normalize(float* __restrict vec, int32_t dim) {
    if (vec == nullptr || dim <= 0) return;
    double sum_sq = 0.0;
    int32_t i = 0;
#if defined(__SSE2__)
    __m128 s4 = _mm_setzero_ps();
    for (; i + 3 < dim; i += 4) {
        __m128 vi = _mm_loadu_ps(vec + i);
        s4 = _mm_add_ps(s4, _mm_mul_ps(vi, vi));
    }
    alignas(16) float s[4];
    _mm_storeu_ps(s, s4);
    sum_sq = static_cast<double>(s[0]) + s[1] + s[2] + s[3];
#endif
    for (; i < dim; ++i) {
        double v = static_cast<double>(vec[i]);
        sum_sq += v * v;
    }
    double norm = std::sqrt(sum_sq);
    if (norm < 1e-12) return;
    float inv_norm = static_cast<float>(1.0 / norm);
    i = 0;
#if defined(__SSE2__)
    __m128 in4 = _mm_set1_ps(inv_norm);
    for (; i + 3 < dim; i += 4) {
        __m128 vi = _mm_loadu_ps(vec + i);
        _mm_storeu_ps(vec + i, _mm_mul_ps(vi, in4));
    }
#endif
    for (; i < dim; ++i) {
        vec[i] *= inv_norm;
    }
}

GNCX_LLM_API void gncx_mean_pooling(
    const float* __restrict hidden_batch,
    int32_t batch_size,
    int32_t dim,
    float* __restrict out_pooled
) {
    if (hidden_batch == nullptr || out_pooled == nullptr || batch_size <= 0 || dim <= 0) return;
    std::memset(out_pooled, 0, static_cast<size_t>(dim) * sizeof(float));
    for (int32_t b = 0; b < batch_size; ++b) {
        const float* __restrict src = hidden_batch + static_cast<size_t>(b) * dim;
        int32_t i = 0;
#if defined(__SSE2__)
        for (; i + 3 < dim; i += 4) {
            __m128 dest = _mm_loadu_ps(out_pooled + i);
            __m128 s = _mm_loadu_ps(src + i);
            _mm_storeu_ps(out_pooled + i, _mm_add_ps(dest, s));
        }
#endif
        for (; i < dim; ++i) {
            out_pooled[i] += src[i];
        }
    }
    const float inv_b = 1.0f / static_cast<float>(batch_size);
    int32_t i = 0;
#if defined(__SSE2__)
    __m128 ib4 = _mm_set1_ps(inv_b);
    for (; i + 3 < dim; i += 4) {
        __m128 dest = _mm_loadu_ps(out_pooled + i);
        _mm_storeu_ps(out_pooled + i, _mm_mul_ps(dest, ib4));
    }
#endif
    for (; i < dim; ++i) {
        out_pooled[i] *= inv_b;
    }
}

GNCX_LLM_API void gncx_bidirectional_attention_gqa(
    const float* __restrict q_batch,
    const float* __restrict k_batch,
    const float* __restrict v_batch,
    int32_t batch_size,
    int32_t n_heads,
    int32_t n_kv_heads,
    int32_t head_dim,
    float attn_scale,
    float* __restrict out_attn_batch
) {
    if (q_batch == nullptr || k_batch == nullptr || v_batch == nullptr || out_attn_batch == nullptr) return;
    if (batch_size <= 0 || n_heads <= 0 || head_dim <= 0) return;
    if (n_kv_heads <= 0) n_kv_heads = n_heads;

    const int32_t q_total_dim = n_heads * head_dim;
    const int32_t kv_total_dim = n_kv_heads * head_dim;
    const int32_t heads_per_kv = n_heads / n_kv_heads;

    std::vector<float> scores(batch_size);

    for (int32_t h = 0; h < n_heads; ++h) {
        const int32_t kv_h = (heads_per_kv > 0) ? (h / heads_per_kv) : 0;
        for (int32_t t = 0; t < batch_size; ++t) {
            const float* __restrict q = q_batch + static_cast<size_t>(t) * q_total_dim + (h * head_dim);
            float* __restrict out_h = out_attn_batch + static_cast<size_t>(t) * q_total_dim + (h * head_dim);

            float max_s = -1e30f;
            for (int32_t s = 0; s < batch_size; ++s) {
                const float* __restrict k = k_batch + static_cast<size_t>(s) * kv_total_dim + (kv_h * head_dim);
                float dot = 0.0f;
                int32_t d = 0;
#if defined(__SSE__)
                __m128 acc0 = _mm_setzero_ps();
                __m128 acc1 = _mm_setzero_ps();
                for (; d + 7 < head_dim; d += 8) {
                    acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_loadu_ps(q + d),     _mm_loadu_ps(k + d)));
                    acc1 = _mm_add_ps(acc1, _mm_mul_ps(_mm_loadu_ps(q + d + 4), _mm_loadu_ps(k + d + 4)));
                }
                __m128 acc = _mm_add_ps(acc0, acc1);
                __m128 shuf = _mm_shuffle_ps(acc, acc, _MM_SHUFFLE(2, 3, 0, 1));
                acc = _mm_add_ps(acc, shuf);
                shuf = _mm_movehl_ps(shuf, acc);
                acc = _mm_add_ss(acc, shuf);
                dot = _mm_cvtss_f32(acc);
#endif
                for (; d < head_dim; ++d) {
                    dot += q[d] * k[d];
                }
                float val = dot * attn_scale;
                scores[s] = val;
                if (val > max_s) max_s = val;
            }

            float sum_exp = 0.0f;
            for (int32_t s = 0; s < batch_size; ++s) {
                float e = std::exp(scores[s] - max_s);
                scores[s] = e;
                sum_exp += e;
            }
            const float inv_sum = 1.0f / (sum_exp + 1e-12f);

            std::memset(out_h, 0, static_cast<size_t>(head_dim) * sizeof(float));
            for (int32_t s = 0; s < batch_size; ++s) {
                const float w = scores[s] * inv_sum;
                const float* __restrict v = v_batch + static_cast<size_t>(s) * kv_total_dim + (kv_h * head_dim);
#if defined(__SSE__)
                const __m128 vw = _mm_set1_ps(w);
                int32_t d = 0;
                for (; d + 7 < head_dim; d += 8) {
                    __m128 o0 = _mm_loadu_ps(out_h + d);
                    __m128 o1 = _mm_loadu_ps(out_h + d + 4);
                    o0 = _mm_add_ps(o0, _mm_mul_ps(vw, _mm_loadu_ps(v + d)));
                    o1 = _mm_add_ps(o1, _mm_mul_ps(vw, _mm_loadu_ps(v + d + 4)));
                    _mm_storeu_ps(out_h + d,     o0);
                    _mm_storeu_ps(out_h + d + 4, o1);
                }
                for (; d < head_dim; ++d) {
                    out_h[d] += w * v[d];
                }
#else
                for (int32_t d = 0; d < head_dim; ++d) {
                    out_h[d] += w * v[d];
                }
#endif
            }
        }
    }
}

GNCX_LLM_API void gncx_bidirectional_attention(
    const float* __restrict q_batch,
    const float* __restrict k_batch,
    const float* __restrict v_batch,
    int32_t batch_size,
    int32_t n_heads,
    int32_t head_dim,
    float attn_scale,
    float* __restrict out_attn_batch
) {
    gncx_bidirectional_attention_gqa(
        q_batch, k_batch, v_batch, batch_size, n_heads, n_heads, head_dim, attn_scale, out_attn_batch
    );
}

