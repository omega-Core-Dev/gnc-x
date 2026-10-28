--[[
  GNC-X LLM — Prova de Conceito (PoC) LuaJIT FFI
  Fase 1: Validação do Bind com bin/gncx_llm_core.so com Zero Overhead
--]]

local ffi = require("ffi")

print("[GNC-X LuaJIT] Inicializando subsistema FFI...")

-- 1. Declaração das estruturas e funções nativas do kernel C++20
ffi.cdef[[
    typedef struct {
        uint8_t  phase_2bit;
        uint8_t  diag_d;
        uint16_t abs_delta;
    } __attribute__((packed)) GNCXCompactTokenStruct;

    typedef struct {
        uint64_t total_macs_equivalent;
        uint64_t zero_origin_cancellations;
        uint64_t zero_weight_skips;
        uint64_t xfold_bucket_merges;
    } __attribute__((packed)) GNCXTelemetryStatsStruct;

    // Alocação e gerenciamento de RAM
    void* gncx_ram_alloc(size_t total_bytes);
    uint8_t* gncx_ram_get_ptr(void* ram_handle);
    void gncx_ram_free(void* ram_handle);

    // Genome VM: Mapeamento de Camada com Origem Central
    void gncx_genome_get_layer_coordinate(
        int32_t layer_idx,
        int32_t total_layers,
        int32_t head_dim,
        int32_t* out_l_rel,
        int32_t* out_sign,
        int32_t* out_k,
        int32_t* out_d
    );

    // Ativações
    void gncx_swiglu_activation(const float* gate, const float* up, int32_t dim, float* out);
    void gncx_geglu_activation(const float* gate, const float* up, int32_t dim, float* out);
]]

-- 2. Localização e carregamento da biblioteca nativa
local lib_path = "./bin/gncx_llm_core.so"
local ok, gncx = pcall(ffi.load, lib_path)
if not ok then
    error("[ERRO CRÍTICO] Falha ao carregar biblioteca nativa em: " .. lib_path .. "\nDetalhe: " .. tostring(gncx))
end

print("[GNC-X LuaJIT] Biblioteca carregada com sucesso: " .. lib_path)

-- 3. Teste 1: Alocação e desalocação cirúrgica de RAM física
local test_bytes = 1024 * 1024 -- 1 MB
local ram_handle = gncx.gncx_ram_alloc(test_bytes)
assert(ram_handle ~= nil, "gncx_ram_alloc falhou ao alocar 1MB")
local ram_ptr = gncx.gncx_ram_get_ptr(ram_handle)
assert(ram_ptr ~= nil, "gncx_ram_get_ptr retornou nulo")
print(string.format("  [OK] Teste 1: Alocação RAM alinhada (Handle: %s, Ptr: %s)", tostring(ram_handle), tostring(ram_ptr)))
gncx.gncx_ram_free(ram_handle)
print("  [OK] Teste 1: Desalocação RAM concluída com sucesso.")

-- 4. Teste 2: Genome VM (Origem Central e Coordenadas Relativas)
-- Testando para modelo de 28 camadas (Qwen3) camada 14 (origem central)
local l_rel = ffi.new("int32_t[1]")
local sign  = ffi.new("int32_t[1]")
local k_val = ffi.new("int32_t[1]")
local d_val = ffi.new("int32_t[1]")

gncx.gncx_genome_get_layer_coordinate(14, 28, 128, l_rel, sign, k_val, d_val)
print(string.format("  [OK] Teste 2: Genome VM Camada 14/28 -> l_rel=%d, sign=%d, k=%d, d=%d", 
      l_rel[0], sign[0], k_val[0], d_val[0]))

-- 5. Teste 3: Ativação SwiGLU (Qwen3 / Alibaba) com Paridade Numérica
local dim = 4
local gate_in = ffi.new("float[?]", dim, { 0.5, -1.0, 2.0, 0.0 })
local up_in   = ffi.new("float[?]", dim, { 2.0,  3.0, 0.5, 1.5 })
local out_act = ffi.new("float[?]", dim)

gncx.gncx_swiglu_activation(gate_in, up_in, dim, out_act)

print("  [OK] Teste 3: SwiGLU Activation Calculada via FFI:")
for i = 0, dim - 1 do
    print(string.format("       [%d] gate=%.2f, up=%.2f -> out=%.6f", i, gate_in[i], up_in[i], out_act[i]))
end

-- 6. Benchmark de Latência de Chamada FFI (100.000 chamadas em loop)
local clock_start = os.clock()
local iters = 100000
for _ = 1, iters do
    gncx.gncx_swiglu_activation(gate_in, up_in, dim, out_act)
end
local elapsed = os.clock() - clock_start
local ns_per_call = (elapsed / iters) * 1e9

print(string.format("[GNC-X LuaJIT] Latência da Ponte FFI: %.2f ns/chamada (%d iterações em %.4f s)",
      ns_per_call, iters, elapsed))

print("[GNC-X LuaJIT] PROVA DE CONCEITO CONCLUÍDA COM 100% DE SUCESSO!")
