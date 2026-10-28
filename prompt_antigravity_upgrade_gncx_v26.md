# Prompt de Engenharia para Antigravity (Gemini Flash) — Upgrade GNC-X v2.6 (Memória DDR3, SIMD, Treino LoRA/MeZO e Espiral 3D no Equador)

Copie e cole o comando completo abaixo no **Antigravity** para que a IA execute de ponta a ponta todas as correções arquiteturais, otimizações de memória/processos para CPU legada sem GPU, primitivas de treinamento (`LoRA` + `MeZO`) e o mapeamento espacial 3D centrado no Equador (`Y = 0`) no seu projeto derivado.

## Instruções Diretas para a IA (Copiar a partir daqui)

Você é um Engenheiro de Sistemas C++20, LuaJIT FFI e Python responsável por atualizar este projeto derivado da arquitetura **GNC-X (Genoma Computacional Espacial)** para a revisão **v2.6**, otimizada para **hardware legado sem placa gráfica dedicada** (CPU Dual-Core x86_64 com SSSE3/SSE4.1 sem AVX2, 6 GB de RAM DDR3 @ 657 MHz e HD mecânico).

Execute as **6 etapas obrigatórias** abaixo diretamente nos arquivos do repositório (`csrc/gncx_llm_kernels.cpp`, `lua/gncx_streamer.lua` e `gncx_engine/gla_streamer.py`), mantendo compatibilidade retroativa com as assinaturas C existentes e validando com um script de testes ao final.

### Etapa 1 — Corrigir o Alocador Colorido DDR3/L1d com Slab HugePage de 4 MB Compartilhado (`csrc/gncx_llm_kernels.cpp`)

**Problema a corrigir:** Em versões anteriores de `gncx_ram_alloc_colored`, buffers menores que `2 MB` (como os 11 *scratch buffers* de inferência e fatias pequenas de KV-Cache) eram alinhados apenas a `64 KB` (`0x10000`) sem `MADV_HUGEPAGE`. Como `65536` zera apenas os bits `15..0`, o bit `16` do endereço virtual/físico permanecia arbitrário e invertia o bit mais significativo do seletor de banco DDR3 (`bits 16..14`), causando colisão de bancos (`Bank mismatch: 0 != 4`) e múltiplos *misses* de dTLB em páginas de `4 KB`.

**Implementação obrigatória:**
1. Substitua a estrutura `GLARamBufferHandle` e a função `gncx_ram_alloc_colored` / `gncx_ram_free` pelo alocador de **Slab HugePage Compartilhado (`GNCXHugePageSlab`)** abaixo, que agrupa todos os buffers `<= 1 MB` dentro de uma única arena de `4 MB` alinhada a `2 MB` (`HUGEPAGE_2MB`) com `MADV_HUGEPAGE`:

```cpp
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
```

2. Em `gncx_ram_free`, decremente `handle->shared_slab->ref_count` quando `shared_slab != nullptr` e libere o Slab apenas quando `ref_count <= 0`.

### Etapa 2 — Destravar `GNCXThreadPool::parallel_for` e Paralelizar Atenção MQA/GQA (`csrc/gncx_llm_kernels.cpp`)

1. Na classe `GNCXThreadPool`, altere a assinatura de `parallel_for` para aceitar `int32_t min_grain = 128` e use `using FnDecay = std::remove_reference_t<Fn>;` dentro do `struct Context { FnDecay* fn_ptr; ... };` para permitir passar *lambdas* nomeadas (lvalues) sem erro de compilação (`forming pointer to reference type`).
2. Em `gncx_mqa_attention_step_ext`:
   - Substitua o buffer global estático `scores` por um buffer `static thread_local std::vector<float> tl_scores;` dentro da lambda `compute_heads` (evitando *data race* entre threads).
   - Vetorize o produto escalar $Q \cdot K^T$ e a acumulação ponderada $w \cdot V$ com intrínsecos SSE (`_mm_loadu_ps`, `_mm_mul_ps`, `_mm_add_ps`).
   - Despache as cabeças de atenção em paralelo quando `win_len >= 8 && n_heads >= 2` chamando `get_global_pool().parallel_for(n_heads, compute_heads, 2);`.
3. Em `gncx_geglu_activation` e `gncx_swiglu_activation`, envolva o laço principal em `get_global_pool().parallel_for(ffn_dim, ..., 2048);`.

### Etapa 3 — Cache `thread_local` de 2 Slots no RoPE + Fusão SSE2 no `RMSNorm` (`csrc/gncx_llm_kernels.cpp`)

1. Em `gncx_apply_rope_xor11_ext`:
   - Crie uma estrutura `RoPECacheSlot` alinhada a 64 bytes com `static thread_local RoPECacheSlot tl_rope_cache[2];` indexada por `slot_idx = (has_rope_freqs != 0) ? 1 : 0`. Isso mantém em cache simultaneamente as tabelas `cos_table` e `sin_table` das camadas SWA e Global para a mesma posição `pos`, eliminando milhares de chamadas `std::pow`, `std::cos` e `std::sin` entre camadas.
   - Vetorize a rotação de $Q$ e $K$ em blocos de 4 floats com SSE2 (`_mm_sub_ps(_mm_mul_ps(x1, c), _mm_mul_ps(x2, s))` e `_mm_add_ps(_mm_mul_ps(x2, c), _mm_mul_ps(x1, s))`), aplicando `phase_2bit ^= 0x03U` apenas se `q_tokens != nullptr`.
2. Em `gncx_rmsnorm_and_encode_4phase` e `gncx_rmsnorm_inplace_heads`:
   - Vetorize a soma de quadrados e a multiplicação por `inv_rms * norm_w[j]` com SSE2.
   - Dentro de `gncx_rmsnorm_and_encode_4phase`, se `out_tokens != nullptr`, acumule `pos_pivots[d]` e `neg_pivots[d]` no próprio laço de codificação e salve em `g_norm_epoch_cancellations` associado a `g_norm_epoch`.
   - Em `count_4phase_cancellations`, retorne `g_norm_epoch_cancellations` em $\mathcal{O}(1)$ quando `in_tokens == g_norm_epoch_tokens_ptr`, eliminando a segunda varredura sobre o vetor.

### Etapa 4 — Adicionar Primitivas de Treinamento em CPU sem GPU (`LoRA` + `MeZO`) e Scanner 3D do Equador (`csrc/gncx_llm_kernels.cpp`)

Adicione e exporte com `GNCX_LLM_API` no final de `csrc/gncx_llm_kernels.cpp`:
1. **Gerador SplitMix64 em Registrador (`splitmix64_next`)**: gera perturbações estocásticas reproduzíveis com 0 bytes de RAM extra.
2. **LoRA Analítico sobre Base GGUF Read-Only**:
   - `gncx_lora_init_adapter(float* a_ptr, float* b_ptr, int32_t in_dim, int32_t out_dim, int32_t rank, uint64_t seed)`
   - `gncx_lora_forward_add(const float* in_vec, const float* a_ptr, const float* b_ptr, int32_t in_dim, int32_t out_dim, int32_t rank, float scale, float* io_out_vec)`
   - `gncx_lora_backward_step(const float* in_vec, const float* grad_out, float* a_ptr, float* b_ptr, float* grad_in_accum, int32_t in_dim, int32_t out_dim, int32_t rank, float scale, float lr, float weight_decay)`
3. **MeZO (Memory-Efficient Zeroth-Order Optimization) com Gradient Clipping**:
   - `gncx_mezo_perturb_params(float* params, int32_t num_params, uint64_t seed, float epsilon)`
   - `gncx_mezo_update_params(float* params, int32_t num_params, uint64_t seed, float projected_grad, float lr, float weight_decay)` — aplique `const float clamped_grad = std::clamp(projected_grad, -10.0f, 10.0f);` para evitar divergência quadrática em adaptadores bilineares.
   - `gncx_cross_entropy_loss(const float* logits, int32_t vocab_size, int32_t target_token_id, float* out_grad_logits)`
4. **Projeção Espacial 3D Centrada no Equador (`gncx_project_layer_3d_point`)**:
   - Exportar struct `GNCXLayer3DPoint` (`layer_index`, `rel_layer`, `crivo_k`, `hemisferio`, `dominant_d`, `dominant_phase`, `canceled_pairs`, `x`, `y`, `z`, `energy_rms`) e função `gncx_project_layer_3d_point` que lê `in_tokens` e `normed_fp32`, calcula a coordenada da Genome VM (`ORIGEM_CENTRAL = total_layers / 2`), fixa o plano do Equador (`rel_layer == 0`) em `Y = 0.0f` (`y = rel_layer * 0.55f`) e projeta `(X, Z)` em espiral segundo o crivo dominante `D(k, d=0..9)` e o fator de rotação `ROT[crivo_k]`.

### Etapa 5 — Atualizar `lua/gncx_streamer.lua` e `gncx_engine/gla_streamer.py`

1. Em `lua/gncx_streamer.lua`:
   - Adicione no `ffi.cdef` as structs `GNCXLayer3DPoint` e todas as funções de treino (`gncx_lora_*`, `gncx_mezo_*`, `gncx_cross_entropy_loss`, `gncx_project_layer_3d_point`).
   - Remova o fallback sequencial `if self.ple_dim > 0 then ... forward_token` de `GNCXEngine:prefill_batch`: implemente o `prefill` em lote completo para o Gemma 4 (`b_ple_tok`, `b_ple_proj`, `b_ple_gate` via `self:_matbatch`) e passe `nil` no argumento `out_tokens` de `gncx_rmsnorm_and_encode_4phase` durante o `prefill_batch`.
   - Adicione o método `GNCXEngine:scan_layer_3d_point(l_idx)` para retornar a coordenada 3D real da camada no Equador.
2. Em `gncx_engine/gla_streamer.py`:
   - Registre os `argtypes` e `restype` das novas funções `gncx_ram_alloc_colored`, `gncx_lora_*`, `gncx_mezo_*` e `gncx_cross_entropy_loss` em `GNCXLLMCoreBridge`.
   - No `prefill_batch`, passe `None` para `out_tokens` nas chamadas `gncx_rmsnorm_and_encode_4phase` das camadas intermediárias.

### Etapa 6 — Compilar e Executar Bateria de Verificação Automatizada

1. Recompile a biblioteca compartilhada (`bin/gncx_llm_core.so` no Linux ou `bin/gncx_llm_core.dll` no Windows) com `-O3 -std=c++20 -march=native`.
2. Crie e execute um script `verify_gncx_fixes.py` que valide:
   - **Teste 1 (HugePage Slab Coloring)**: aloque 11 buffers de `4 KB` a `128 KB` com `gncx_ram_alloc_colored` e comprove que todos residem dentro de `< 1 MB` de distância no mesmo Slab HugePage e que `gncx_ram_get_ddr3_bank` e `gncx_ram_get_l1_set` batem 100% com `(d * R_odd + k) & 7` e `(d * R_odd * 3 + k * 7) & 63`.
   - **Teste 2 (Fast-Path RMSNorm)**: meça o tempo de `gncx_rmsnorm_and_encode_4phase` com e sem `out_tokens`.
   - **Teste 3 (Convergência LoRA Analítico)**: execute 40 passos de `gncx_lora_backward_step` reduzindo a `gncx_cross_entropy_loss` para `< 0.01`.
   - **Teste 4 (Convergência MeZO Sem Backprop)**: verifique restauração *bit-exact* após `+epsilon` / `-epsilon` (`max_diff < 1e-6`) e execute 80 passos de `gncx_mezo_update_params` comprovando queda monotônica da loss.
   - **Teste 5 (Projeção 3D no Equador)**: verifique que a camada central (`layer_index = total_layers // 2`) produz exatamente `rel_layer == 0` e `Y == 0.0`.
