#pragma once
/**
 * GNC-X Geometric Memory Manager (`gncx_geometric_mm.hpp`)
 * --------------------------------------------------------
 * Implementa o Gerenciador de Memória Geométrica não-linear baseado no Crivo
 * Diagonal D(k, d) e na Unidade de Tradução Bijetiva O(1) Φ(k, d, R).
 *
 * Autor da Pesquisa Original: Zaqueu Ribeiro da Costa (omega-Core-Dev)
 */

#include <cstdint>
#include <array>
#include <bitset>
#include <optional>
#include <algorithm>
#include <utility>

namespace gncx {

struct GeoAddr {
    uint8_t k;      // Nível de escala L0..L3 (0..3)
    uint8_t d;      // Índice diagonal (0..7 no módulo 8)
    uint8_t R;      // Fator de rotação R_{k+1} coprimo (1, 3, 5, 7)
    bool operator==(const GeoAddr& o) const = default;
};

struct PhiTranslator {
    // Fatores de rotação ímpares (coprimos com 8, garantem bijeção matemática)
    static constexpr std::array<uint8_t, 4> ROT = {1, 3, 5, 7};

    // Φ(k, d, R) -> (i, j) matriz física 8x8
    static constexpr std::pair<uint8_t, uint8_t> forward(GeoAddr g) {
        const uint8_t Rk = ROT[g.k & 0x3];
        const uint8_t D  = static_cast<uint8_t>((static_cast<uint32_t>(g.d) * Rk) & 0x7);
        const uint8_t i  = static_cast<uint8_t>((static_cast<uint32_t>(g.d) + static_cast<uint32_t>(g.k) * Rk) & 0x7);
        const uint8_t j  = static_cast<uint8_t>((static_cast<uint32_t>(D) + g.k) & 0x7);
        return {i, j};
    }

    // Φ^-1(i, j, k) -> GeoAddr reversível (prova construtiva de bijeção)
    static constexpr GeoAddr inverse(uint8_t i, uint8_t /*j*/, uint8_t k) {
        const uint8_t Rk = ROT[k & 0x3];
        const uint8_t d  = static_cast<uint8_t>((static_cast<uint32_t>(i) + 8U - ((static_cast<uint32_t>(k) * Rk) & 0x7)) & 0x7);
        return GeoAddr{k, d, Rk};
    }

    // Converte coordenada geométrica em índice linear dentro do bloco 8x8
    static constexpr uint32_t to_linear_index(GeoAddr g) {
        const auto [i, j] = forward(g);
        return (static_cast<uint32_t>(i) * 8U + static_cast<uint32_t>(j));
    }
};

/**
 * Alocador Diagonal Geométrico GNC-X
 * Suporta múltiplos grids de 64 frames (escalável para acomodar até 512 frames de tensores)
 */
class GeometricMemoryManager {
public:
    static constexpr uint32_t GRID_FRAMES = 64; // 8x8 frames por bloco
    static constexpr uint32_t NUM_DIAGONALS = 8;
    static constexpr uint32_t MAX_GRIDS = 8;    // Até 512 frames de camadas/tensores

private:
    uint32_t total_grids_;
    uint32_t total_capacity_;

    std::array<std::bitset<GRID_FRAMES>, MAX_GRIDS> grid_bitmaps_;
    std::array<std::array<std::bitset<8>, NUM_DIAGONALS>, MAX_GRIDS> diagonal_sieves_;

public:
    explicit GeometricMemoryManager(uint32_t min_frames = 64) {
        total_grids_ = std::clamp((min_frames + GRID_FRAMES - 1) / GRID_FRAMES, 1U, MAX_GRIDS);
        total_capacity_ = total_grids_ * GRID_FRAMES;
        for (uint32_t g = 0; g < MAX_GRIDS; ++g) {
            grid_bitmaps_[g].reset();
            for (auto& s : diagonal_sieves_[g]) {
                s.reset();
            }
        }
    }

    // Aloca um slot no crivo diagonal D(k, d)
    std::optional<std::pair<GeoAddr, uint32_t>> alloc(uint8_t k_hint = 2) {
        const uint8_t Rk = PhiTranslator::ROT[k_hint & 0x3];

        for (uint32_t g_idx = 0; g_idx < total_grids_; ++g_idx) {
            for (uint8_t d = 0; d < 8; ++d) {
                const uint8_t D = static_cast<uint8_t>((static_cast<uint32_t>(d) * Rk) & 0x7);
                if (diagonal_sieves_[g_idx][D].count() < 8) {
                    for (uint8_t slot = 0; slot < 8; ++slot) {
                        GeoAddr g{k_hint, static_cast<uint8_t>((d + slot) & 0x7), Rk};
                        const uint32_t local_idx = PhiTranslator::to_linear_index(g);
                        if (!grid_bitmaps_[g_idx].test(local_idx)) {
                            grid_bitmaps_[g_idx].set(local_idx);
                            diagonal_sieves_[g_idx][D].set(slot);
                            const uint32_t global_idx = (g_idx * GRID_FRAMES) + local_idx;
                            return std::make_pair(g, global_idx);
                        }
                    }
                }
            }
        }
        return std::nullopt;
    }

    // Libera slot geométrico
    void free(GeoAddr g, uint32_t global_idx) {
        const uint32_t g_idx = global_idx / GRID_FRAMES;
        const uint32_t local_idx = global_idx % GRID_FRAMES;
        if (g_idx >= total_grids_) return;

        grid_bitmaps_[g_idx].reset(local_idx);
        const uint8_t D = static_cast<uint8_t>((static_cast<uint32_t>(g.d) * PhiTranslator::ROT[g.k & 0x3]) & 0x7);
        for (int i = 0; i < 8; ++i) {
            if (diagonal_sieves_[g_idx][D].test(i)) {
                diagonal_sieves_[g_idx][D].reset(i);
                break;
            }
        }
    }

    // Mede a fragmentação física externa (0.0 = sem fragmentação, 1.0 = totalmente fragmentado)
    float fragmentation() const {
        uint32_t total_free = 0;
        uint32_t max_free_run = 0;
        uint32_t current_run = 0;

        for (uint32_t g_idx = 0; g_idx < total_grids_; ++g_idx) {
            for (uint32_t i = 0; i < GRID_FRAMES; ++i) {
                if (!grid_bitmaps_[g_idx].test(i)) {
                    total_free++;
                    current_run++;
                } else {
                    max_free_run = std::max(max_free_run, current_run);
                    current_run = 0;
                }
            }
        }
        max_free_run = std::max(max_free_run, current_run);
        if (total_free == 0) return 0.0f;
        return 1.0f - (static_cast<float>(max_free_run) / static_cast<float>(total_free));
    }

    size_t used() const {
        size_t u = 0;
        for (uint32_t g_idx = 0; g_idx < total_grids_; ++g_idx) {
            u += grid_bitmaps_[g_idx].count();
        }
        return u;
    }

    size_t capacity() const {
        return total_capacity_;
    }
};

} // namespace gncx
