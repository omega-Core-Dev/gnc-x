--[[
  testes_v26.lua - Bateria de Testes em LuaJIT para as novidades do Upgrade v2.6
--]]

package.path = "./?.lua;" .. package.path
local ffi = require("ffi")

print("=" .. string.rep("=", 69))
print("🚀 TESTES GNC-X v2.6 (LUAJIT FFI)")
print("=" .. string.rep("=", 69))

-- 1. Carrega o streamer (que define todo o ffi.cdef das funções nativas)
local streamer_mod = require("lua.gncx_streamer")
print("  [OK] Módulo lua.gncx_streamer carregado com sucesso")

local lib = ffi.load("./bin/gncx_llm_core.so")
assert(lib ~= nil, "Falha ao carregar bin/gncx_llm_core.so")

-- Teste 1: Validação do Alocador de HugePages via LuaJIT
local h = lib.gncx_ram_alloc_colored(8192, 1, 2, 3)
assert(h ~= nil, "gncx_ram_alloc_colored falhou em LuaJIT")
local ptr = lib.gncx_ram_get_ptr(h)
local bank = lib.gncx_ram_get_ddr3_bank(h)
local l1 = lib.gncx_ram_get_l1_set(h)
print(string.format("  [OK] LuaJIT HugePage Alloc: Addr=%s | Bank=%d | L1Set=%d", tostring(ptr), bank, l1))
lib.gncx_ram_free(h)

-- Teste 2: Validação da Projeção 3D centrada no Equador via LuaJIT
local pt = ffi.new("GNCXLayer3DPoint[1]")
local total_layers = 35
local center_layer = math.floor(total_layers / 2)

lib.gncx_project_layer_3d_point(center_layer, total_layers, nil, nil, 0, pt)
print(string.format("  [OK] LuaJIT Projeção 3D: Camada Central %d/%d -> rel_layer=%d, Y=%.4f",
    center_layer, total_layers, pt[0].rel_layer, pt[0].y))
assert(pt[0].rel_layer == 0, "rel_layer deve ser 0")
assert(math.abs(pt[0].y) < 1e-6, "Y no Equador deve ser 0.0")

print("\n" .. string.rep("🏆 ", 20))
print("TESTES LUAJIT v2.6 CONCLUÍDOS COM 100% DE SUCESSO!")
print(string.rep("🏆 ", 20) .. "\n")
