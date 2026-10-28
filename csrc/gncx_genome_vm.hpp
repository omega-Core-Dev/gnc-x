#pragma once
/**
 * GNC-X Genome Virtual Machine (`gncx_genome_vm.hpp`)
 * ---------------------------------------------------
 * Arquitetura de Coordenadas Centradas no Equador do Transformer.
 * Em vez de indexação unilateral monótona (0 .. total_layers-1),
 * define:
 *   ORIGEM_CENTRAL = total_layers / 2
 *   rel_layer      = layer_index - ORIGEM_CENTRAL
 * 
 * O Crivo Diagonal D(k, d) é calculado simetricamente a partir do centro,
 * eliminando drift numérico e espelhando a simetria de abstração (Encoder <-> Decoder).
 */

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <array>

namespace gncx {

struct GenomeLayerCoordinate {
    int32_t layer_index;      // Índice absoluto da camada: 0 .. total_layers - 1
    int32_t total_layers;     // Total de camadas (ex: 18 no Gemma 2B, 35 no Gemma 4)
    int32_t origem_central;   // Centro exato: total_layers / 2 (ex: 9 ou 17)
    int32_t rel_layer;        // Coordenada relativa: layer_index - ORIGEM_CENTRAL (-17 .. +17)
    int32_t dist_centro;      // Distância absoluta do núcleo: |rel_layer|
    int32_t crivo_k;          // Ordem k do Crivo D(k, d) calculada a partir do centro
    uint8_t hemisferio;       // 0 = Núcleo/Superior (rel >= 0), 1 = Inferior (rel < 0)
};

/**
 * Calcula a coordenada e ordem de Crivo D(k, d) centrada na ORIGEM_CENTRAL.
 */
inline GenomeLayerCoordinate make_genome_coordinate(
    int32_t layer_index,
    int32_t total_layers,
    int32_t base_k = 3
) {
    GenomeLayerCoordinate coord{};
    coord.layer_index = layer_index;
    coord.total_layers = std::max(1, total_layers);
    coord.origem_central = coord.total_layers / 2;
    coord.rel_layer = layer_index - coord.origem_central;
    coord.dist_centro = std::abs(coord.rel_layer);
    coord.hemisferio = (coord.rel_layer >= 0) ? 0U : 1U;

    // Modulação simétrica do Crivo D(k, d) a partir do centro:
    // No núcleo (dist_centro == 0): alta fidelidade (k=2 ou 3)
    // Nas bordas: crivo proporcional à distância do equador
    if (coord.origem_central > 0) {
        // Varia k harmonicamente entre 2 e 4 (mantendo resíduos em uint16_t)
        int32_t delta_k = (coord.dist_centro * 2) / coord.origem_central;
        coord.crivo_k = std::clamp(2 + delta_k, 2, 4);
    } else {
        coord.crivo_k = std::clamp(base_k, 1, 4);
    }

    return coord;
}

} // namespace gncx
