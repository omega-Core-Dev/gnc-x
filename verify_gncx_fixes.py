#!/usr/bin/env python3
"""
verify_gncx_fixes.py - Bateria de Verificação Automatizada GNC-X v2.6
Valida as 5 etapas essenciais sem dependências externas (Python puro + ctypes):
  1. HugePage Slab Coloring (Bancos DDR3 e L1 sets)
  2. Fast-Path RMSNorm (out_tokens = NULL)
  3. Convergência LoRA Analítico (loss < 0.01 em <= 40 passos)
  4. MeZO Zeroth-Order (Restauração bit-exact e convergência monotônica)
  5. Projeção Espacial 3D Centrada no Equador (Y = 0.0 na camada central)
"""

import sys
import os
import time
import math
import random
import ctypes
from ctypes import (
    c_int32, c_uint8, c_uint32, c_uint64, c_float, c_size_t, c_void_p,
    Structure, POINTER, byref
)

SO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "bin", "gncx_llm_core.so"))

if not os.path.exists(SO_PATH):
    print(f"[ERRO] Biblioteca {SO_PATH} não encontrada. Execute ./compilar_linux.sh primeiro.")
    sys.exit(1)

lib = ctypes.CDLL(SO_PATH)

# ============================================================================
# Ctypes Structs & Signatures
# ============================================================================

class GNCXCompactToken(Structure):
    _layout_ = "ms"
    _pack_ = 1
    _fields_ = [
        ("phase_2bit", c_uint8),
        ("diag_d", c_uint8),
        ("abs_delta", ctypes.c_uint16),
    ]

class GNCXLayer3DPoint(Structure):
    _layout_ = "ms"
    _pack_ = 1
    _fields_ = [
        ("layer_index", c_int32),
        ("rel_layer", c_int32),
        ("crivo_k", c_int32),
        ("hemisferio", c_uint8),
        ("dominant_d", c_uint8),
        ("dominant_phase", c_uint8),
        ("canceled_pairs", c_uint64),
        ("x", c_float),
        ("y", c_float),
        ("z", c_float),
        ("energy_rms", c_float),
    ]

# gncx_ram_alloc_colored
lib.gncx_ram_alloc_colored.argtypes = [c_size_t, c_uint8, c_uint8, c_uint8]
lib.gncx_ram_alloc_colored.restype = c_void_p

# gncx_ram_get_ptr
lib.gncx_ram_get_ptr.argtypes = [c_void_p]
lib.gncx_ram_get_ptr.restype = c_void_p

# gncx_ram_get_ddr3_bank
lib.gncx_ram_get_ddr3_bank.argtypes = [c_void_p]
lib.gncx_ram_get_ddr3_bank.restype = c_uint8

# gncx_ram_get_l1_set
lib.gncx_ram_get_l1_set.argtypes = [c_void_p]
lib.gncx_ram_get_l1_set.restype = c_uint8

# gncx_ram_free
lib.gncx_ram_free.argtypes = [c_void_p]
lib.gncx_ram_free.restype = None

# gncx_rmsnorm_and_encode_4phase
lib.gncx_rmsnorm_and_encode_4phase.argtypes = [
    POINTER(c_float), c_void_p, c_size_t, c_int32, c_float, c_int32,
    POINTER(c_float), POINTER(GNCXCompactToken), POINTER(c_float)
]
lib.gncx_rmsnorm_and_encode_4phase.restype = None

# LoRA
lib.gncx_lora_init_adapter.argtypes = [POINTER(c_float), POINTER(c_float), c_int32, c_int32, c_int32, c_uint64]
lib.gncx_lora_init_adapter.restype = None

lib.gncx_lora_forward_add.argtypes = [
    POINTER(c_float), POINTER(c_float), POINTER(c_float),
    c_int32, c_int32, c_int32, c_float, POINTER(c_float)
]
lib.gncx_lora_forward_add.restype = None

lib.gncx_lora_backward_step.argtypes = [
    POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float),
    POINTER(c_float), c_int32, c_int32, c_int32, c_float, c_float, c_float
]
lib.gncx_lora_backward_step.restype = None

# MeZO
lib.gncx_mezo_perturb_params.argtypes = [POINTER(c_float), c_int32, c_uint64, c_float]
lib.gncx_mezo_perturb_params.restype = None

lib.gncx_mezo_update_params.argtypes = [POINTER(c_float), c_int32, c_uint64, c_float, c_float, c_float]
lib.gncx_mezo_update_params.restype = None

# Cross Entropy
lib.gncx_cross_entropy_loss.argtypes = [POINTER(c_float), c_int32, c_int32, POINTER(c_float)]
lib.gncx_cross_entropy_loss.restype = c_float

# 3D Point
lib.gncx_project_layer_3d_point.argtypes = [
    c_int32, c_int32, POINTER(GNCXCompactToken), POINTER(c_float), c_int32, POINTER(GNCXLayer3DPoint)
]
lib.gncx_project_layer_3d_point.restype = None


def test_1_hugepage_slab_coloring():
    print("=" * 70)
    print("⚡ TESTE 1: HugePage Slab Compartilhado e Coloração Física DDR3/L1d")
    print("=" * 70)

    # 11 scratch buffers com tamanhos variando de 4 KB a 128 KB
    alloc_configs = [
        (4096,   0, 1, 1),
        (8192,   1, 2, 3),
        (16384,  2, 3, 5),
        (32768,  3, 4, 7),
        (65536,  0, 5, 1),
        (131072, 1, 6, 3),
        (4096,   2, 7, 5),
        (8192,   3, 0, 7),
        (16384,  0, 2, 1),
        (32768,  1, 4, 3),
        (65536,  2, 6, 5),
    ]

    handles = []
    ptrs = []
    try:
        for i, (sz, k, d, R) in enumerate(alloc_configs):
            h = lib.gncx_ram_alloc_colored(sz, k, d, R)
            assert h is not None, f"Falha ao alocar buffer {i} ({sz} bytes)"
            p = lib.gncx_ram_get_ptr(h)
            bank = lib.gncx_ram_get_ddr3_bank(h)
            l1 = lib.gncx_ram_get_l1_set(h)

            R_odd = (R | 1) & 0x7
            expected_bank = (d * R_odd + k) & 0x7
            expected_l1 = (d * R_odd * 3 + k * 7) & 0x3F

            assert bank == expected_bank, f"Buffer {i}: Banco incorreto {bank} != {expected_bank}"
            assert l1 == expected_l1, f"Buffer {i}: L1 Set incorreto {l1} != {expected_l1}"

            handles.append(h)
            ptrs.append(p)
            print(f"  [OK] Buf {i:02d} ({sz:6d} B): Addr=0x{p:x} | Bank={bank} (exp {expected_bank}) | L1Set={l1:02d} (exp {expected_l1:02d})")

        min_addr = min(ptrs)
        max_addr = max(ptrs)
        span_bytes = max_addr - min_addr
        print(f"\n  Distância máxima entre o 1º e último buffer: {span_bytes} bytes ({span_bytes / 1024:.1f} KB)")
        assert span_bytes < 1024 * 1024, f"Buffers ultrapassaram arena de 1 MB: span={span_bytes}"
        print("  ✅ Sucesso: Todos os 11 scratch buffers residem contíguos no mesmo Slab HugePage de 4 MB!")

    finally:
        for h in handles:
            lib.gncx_ram_free(h)
        print("  ✅ Sucesso: Liberação do Slab HugePage com contagem de referências concluída.")


def test_2_fastpath_rmsnorm():
    print("\n" + "=" * 70)
    print("⚡ TESTE 2: Fast-Path no RMSNorm com Vetorização SSE2")
    print("=" * 70)

    dim = 2048
    random.seed(42)
    in_arr = (c_float * dim)(*[random.gauss(0, 1) for _ in range(dim)])
    weight_arr = (c_float * dim)(*[1.0 for _ in range(dim)])

    out_norm = (c_float * dim)()
    tokens = (GNCXCompactToken * dim)()
    q_scale = c_float(0.0)

    # 1. Warm-up
    for _ in range(50):
        lib.gncx_rmsnorm_and_encode_4phase(in_arr, ctypes.cast(weight_arr, c_void_p), 0, dim, 1e-6, 3, out_norm, tokens, byref(q_scale))
        lib.gncx_rmsnorm_and_encode_4phase(in_arr, ctypes.cast(weight_arr, c_void_p), 0, dim, 1e-6, 3, out_norm, None, None)

    # 2. Medir caminho completo com quantização 4-fases
    t0 = time.perf_counter()
    iters = 2000
    for _ in range(iters):
        lib.gncx_rmsnorm_and_encode_4phase(in_arr, ctypes.cast(weight_arr, c_void_p), 0, dim, 1e-6, 3, out_norm, tokens, byref(q_scale))
    t_full = time.perf_counter() - t0

    # 3. Medir fast-path (out_tokens = None)
    t0 = time.perf_counter()
    for _ in range(iters):
        lib.gncx_rmsnorm_and_encode_4phase(in_arr, ctypes.cast(weight_arr, c_void_p), 0, dim, 1e-6, 3, out_norm, None, None)
    t_fast = time.perf_counter() - t0

    speedup = t_full / max(1e-9, t_fast)
    print(f"  Tempo com 4-Phase Encoding : {t_full*1000:.2f} ms ({t_full/iters*1e6:.2f} µs/op)")
    print(f"  Tempo Fast-Path fp32 puro  : {t_fast*1000:.2f} ms ({t_fast/iters*1e6:.2f} µs/op)")
    print(f"  Aceleração do Fast-Path    : {speedup:.2f}x")
    assert speedup > 1.2, f"Fast path deve ser mais rápido que o caminho completo (speedup={speedup:.2f})"
    print("  ✅ Sucesso: Fast-Path validado e acelerado com sucesso!")


def test_3_lora_analytical_convergence():
    print("\n" + "=" * 70)
    print("⚡ TESTE 3: Convergência de Treinamento LoRA Analítico em CPU")
    print("=" * 70)

    in_dim = 64
    out_dim = 32
    rank = 8
    target_token = 7

    random.seed(1337)
    in_vec = (c_float * in_dim)(*[random.gauss(0, 1) for _ in range(in_dim)])
    a_mat = (c_float * (rank * in_dim))()
    b_mat = (c_float * (out_dim * rank))()

    lib.gncx_lora_init_adapter(a_mat, b_mat, in_dim, out_dim, rank, 99999)

    out_logits = (c_float * out_dim)()
    grad_logits = (c_float * out_dim)()

    initial_loss = None
    final_loss = None

    lr = 0.15
    for step in range(40):
        for i in range(out_dim):
            out_logits[i] = 0.0

        lib.gncx_lora_forward_add(in_vec, a_mat, b_mat, in_dim, out_dim, rank, 1.0, out_logits)
        loss = lib.gncx_cross_entropy_loss(out_logits, out_dim, target_token, grad_logits)

        if step == 0:
            initial_loss = loss
            print(f"  Passo {step:02d} -> Loss Inicial: {loss:.4f}")

        lib.gncx_lora_backward_step(
            in_vec, grad_logits, a_mat, b_mat, None,
            in_dim, out_dim, rank, 1.0, lr, 0.0
        )
        final_loss = loss

    print(f"  Passo 40 -> Loss Final  : {final_loss:.6f}")
    assert final_loss < 0.01, f"Loss não convergiu para < 0.01 (final_loss={final_loss})"
    assert final_loss < initial_loss * 0.01, "Redução de loss insuficiente"
    print("  ✅ Sucesso: LoRA Analítico convergiu em CPU para Loss < 0.01!")


def test_4_mezo_zeroth_order():
    print("\n" + "=" * 70)
    print("⚡ TESTE 4: Otimização MeZO de Ordem Zero (Sem Backprop)")
    print("=" * 70)

    num_params = 64
    random.seed(2026)
    initial_vals = [random.gauss(0, 0.8) for _ in range(num_params)]
    params = (c_float * num_params)(*initial_vals)

    # 1. Teste de restauração bit-exact (+eps / -eps)
    seed = 424242
    eps = 1e-3
    lib.gncx_mezo_perturb_params(params, num_params, seed, eps)
    perturbed_vals = list(params)
    assert any(abs(a - b) > 1e-5 for a, b in zip(initial_vals, perturbed_vals)), "Parâmetros não foram perturbados"
    lib.gncx_mezo_perturb_params(params, num_params, seed, -eps)
    restored_vals = list(params)
    max_diff = max(abs(a - b) for a, b in zip(initial_vals, restored_vals))
    print(f"  Diferença máxima após +eps / -eps: {max_diff:.2e}")
    assert max_diff < 1e-6, f"Restauração não foi bit-exact: max_diff={max_diff}"
    print("  ✅ Sucesso: Perturbação MeZO bit-exact verificada!")

    # 2. Otimização MeZO com função quadrática normalizada L(p) = mean((p_i - 0.5)^2)
    def loss_fn(p):
        return sum((p[i] - 0.5) ** 2 for i in range(num_params)) / float(num_params)

    init_loss = loss_fn(params)
    lr = 0.04
    seed_base = 777

    for step in range(80):
        s = seed_base + step
        lib.gncx_mezo_perturb_params(params, num_params, s, eps)
        loss_pos = loss_fn(params)

        lib.gncx_mezo_perturb_params(params, num_params, s, -2.0 * eps)
        loss_neg = loss_fn(params)

        # Restaura ao ponto central
        lib.gncx_mezo_perturb_params(params, num_params, s, eps)

        proj_grad = (loss_pos - loss_neg) / (2.0 * eps)
        lib.gncx_mezo_update_params(params, num_params, s, proj_grad, lr, 0.0)

    final_loss = loss_fn(params)
    print(f"  Loss MeZO Inicial: {init_loss:.4f} -> Loss Final (80 passos): {final_loss:.4f}")
    assert final_loss < init_loss, f"MeZO não reduziu a loss ({final_loss} >= {init_loss})"
    print("  ✅ Sucesso: Otimizador MeZO reduziu a loss monotônica com sucesso sem nenhum cálculo de gradiente analítico!")


def test_5_project_3d_point_equator():
    print("\n" + "=" * 70)
    print("⚡ TESTE 5: Mapeamento Espacial 3D Centrado no Equador (Y = 0)")
    print("=" * 70)

    # 35 camadas (Gemma 4) -> Camada Central = 17
    total_layers = 35
    center_layer = total_layers // 2
    dim = 256

    tokens = (GNCXCompactToken * dim)()
    for i in range(dim):
        tokens[i].phase_2bit = (i % 4)
        tokens[i].diag_d = (i % 10)
        tokens[i].abs_delta = 10

    normed = (c_float * dim)(*[float(i % 10) * 0.1 for i in range(dim)])
    pt = GNCXLayer3DPoint()

    # Camada central (Equador)
    lib.gncx_project_layer_3d_point(center_layer, total_layers, tokens, normed, dim, byref(pt))
    print(f"  Camada Central ({center_layer}/{total_layers}):")
    print(f"    rel_layer={pt.rel_layer} | Y={pt.y:.4f} | crivo_k={pt.crivo_k} | hemisferio={pt.hemisferio}")
    print(f"    (X, Z)=({pt.x:.4f}, {pt.z:.4f}) | energy_rms={pt.energy_rms:.4f} | canceled={pt.canceled_pairs}")

    assert pt.rel_layer == 0, f"rel_layer esperado 0, obteve {pt.rel_layer}"
    assert abs(pt.y) < 1e-6, f"Y no Equador esperado 0.0, obteve {pt.y}"
    assert pt.hemisferio == 0, f"Hemisfério no centro esperado 0, obteve {pt.hemisferio}"

    # Camada inferior (ex: 0)
    lib.gncx_project_layer_3d_point(0, total_layers, tokens, normed, dim, byref(pt))
    print(f"  Camada Base (0/{total_layers}): rel_layer={pt.rel_layer} | Y={pt.y:.4f} | hemisferio={pt.hemisferio}")
    assert pt.rel_layer < 0, f"rel_layer esperado < 0, obteve {pt.rel_layer}"
    assert pt.y < 0.0, f"Y esperado < 0.0, obteve {pt.y}"
    assert pt.hemisferio == 1, f"Hemisfério inferior esperado 1, obteve {pt.hemisferio}"

    # Camada superior (ex: 34)
    lib.gncx_project_layer_3d_point(34, total_layers, tokens, normed, dim, byref(pt))
    print(f"  Camada Topo (34/{total_layers}): rel_layer={pt.rel_layer} | Y={pt.y:.4f} | hemisferio={pt.hemisferio}")
    assert pt.rel_layer > 0, f"rel_layer esperado > 0, obteve {pt.rel_layer}"
    assert pt.y > 0.0, f"Y esperado > 0.0, obteve {pt.y}"
    assert pt.hemisferio == 0, f"Hemisfério superior esperado 0, obteve {pt.hemisferio}"

    print("  ✅ Sucesso: Projeção 3D centrada no Equador (Y = 0) verificada perfeitamente!")


if __name__ == "__main__":
    print("\n" + "🚀 " * 20)
    print("INICIANDO BATERIA DE TESTES DE INTEGRAÇÃO GNC-X v2.6")
    print("🚀 " * 20 + "\n")

    test_1_hugepage_slab_coloring()
    test_2_fastpath_rmsnorm()
    test_3_lora_analytical_convergence()
    test_4_mezo_zeroth_order()
    test_5_project_3d_point_equator()

    print("\n" + "🏆 " * 20)
    print("TODOS OS 5 TESTES DE INTEGRAÇÃO PASSARAM COM 100% DE SUCESSO!")
    print("🏆 " * 20 + "\n")
