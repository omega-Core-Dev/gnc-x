--[[
  GNC-X LLM — Módulo GGUF Zero-Copy em LuaJIT
  Fase 2: Inspetor e Mapeador de Tensores GLA via Memória Virtual
--]]

local ffi = require("ffi")

-- Declaração das funções de mapeamento nativo C++20
ffi.cdef[[
    void* gncx_gla_map_readonly(const char* filepath, size_t* out_size);
    const uint8_t* gncx_gla_get_base_ptr(void* opaque_handle);
    void gncx_gla_unmap(void* opaque_handle);
]]

local GGUFValueType = {
    UINT8   = 0,
    INT8    = 1,
    UINT16  = 2,
    INT16   = 3,
    UINT32  = 4,
    INT32   = 5,
    FLOAT32 = 6,
    BOOL    = 7,
    STRING  = 8,
    ARRAY   = 9,
    UINT64  = 10,
    INT64   = 11,
    FLOAT64 = 12,
}

local GGMLQuantType = {
    F32  = 0,
    F16  = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    BF16 = 30,
}

-- Informação de blocos (block_size, bytes_per_block)
local GGML_BLOCK_INFO = {
    [0]  = { 1, 4 },    -- F32
    [1]  = { 1, 2 },    -- F16
    [30] = { 1, 2 },    -- BF16
    [2]  = { 32, 18 },  -- Q4_0
    [3]  = { 32, 20 },  -- Q4_1
    [6]  = { 32, 22 },  -- Q5_0
    [7]  = { 32, 24 },  -- Q5_1
    [8]  = { 32, 34 },  -- Q8_0
    [9]  = { 32, 36 },  -- Q8_1
    [10] = { 256, 84 }, -- Q2_K
    [11] = { 256, 110 },-- Q3_K
    [12] = { 256, 144 },-- Q4_K
    [13] = { 256, 176 },-- Q5_K
    [14] = { 256, 210 },-- Q6_K
    [15] = { 256, 292 },-- Q8_K
}

local GNCXGGUF = {}
GNCXGGUF.__index = GNCXGGUF

function GNCXGGUF.open(filepath, lib_path)
    lib_path = lib_path or "./bin/gncx_llm_core.so"
    local gncx_core = ffi.load(lib_path)

    local size_out = ffi.new("size_t[1]")
    local handle = gncx_core.gncx_gla_map_readonly(filepath, size_out)
    if handle == nil then
        return nil, "Falha ao mapear arquivo GGUF via GLA: " .. filepath
    end

    local base_ptr = gncx_core.gncx_gla_get_base_ptr(handle)
    local self = setmetatable({
        filepath    = filepath,
        file_size   = tonumber(size_out[0]),
        core        = gncx_core,
        handle      = handle,
        base        = base_ptr,
        cursor      = 0,
        metadata    = {},
        tensors     = {},
        data_offset = 0,
    }, GNCXGGUF)

    local ok, err = pcall(self._parse, self)
    if not ok then
        self:close()
        return nil, "Erro no parse GGUF: " .. tostring(err)
    end

    return self
end

function GNCXGGUF:close()
    if self.handle ~= nil then
        self.core.gncx_gla_unmap(self.handle)
        self.handle = nil
        self.base = nil
    end
end

-- Métodos internos de leitura por ponteiro
function GNCXGGUF:_read_bytes(n)
    local cur = self.cursor
    self.cursor = cur + n
    return self.base + cur
end

function GNCXGGUF:_read_u8()
    local p = self:_read_bytes(1)
    return p[0]
end

function GNCXGGUF:_read_i8()
    local p = ffi.cast("const int8_t*", self:_read_bytes(1))
    return p[0]
end

function GNCXGGUF:_read_u16()
    local p = ffi.cast("const uint16_t*", self:_read_bytes(2))
    return p[0]
end

function GNCXGGUF:_read_i16()
    local p = ffi.cast("const int16_t*", self:_read_bytes(2))
    return p[0]
end

function GNCXGGUF:_read_u32()
    local p = ffi.cast("const uint32_t*", self:_read_bytes(4))
    return p[0]
end

function GNCXGGUF:_read_i32()
    local p = ffi.cast("const int32_t*", self:_read_bytes(4))
    return p[0]
end

function GNCXGGUF:_read_f32()
    local p = ffi.cast("const float*", self:_read_bytes(4))
    return p[0]
end

function GNCXGGUF:_read_f64()
    local p = ffi.cast("const double*", self:_read_bytes(8))
    return p[0]
end

function GNCXGGUF:_read_u64()
    local p = ffi.cast("const uint64_t*", self:_read_bytes(8))
    return p[0]
end

function GNCXGGUF:_read_i64()
    local p = ffi.cast("const int64_t*", self:_read_bytes(8))
    return p[0]
end

function GNCXGGUF:_read_string()
    local len = tonumber(self:_read_u64())
    local str_ptr = self:_read_bytes(len)
    return ffi.string(str_ptr, len)
end

function GNCXGGUF:_read_value(vtype)
    if vtype == GGUFValueType.UINT8   then return self:_read_u8() end
    if vtype == GGUFValueType.INT8    then return self:_read_i8() end
    if vtype == GGUFValueType.UINT16  then return self:_read_u16() end
    if vtype == GGUFValueType.INT16   then return self:_read_i16() end
    if vtype == GGUFValueType.UINT32  then return self:_read_u32() end
    if vtype == GGUFValueType.INT32   then return self:_read_i32() end
    if vtype == GGUFValueType.FLOAT32 then return self:_read_f32() end
    if vtype == GGUFValueType.BOOL    then return (self:_read_u8() ~= 0) end
    if vtype == GGUFValueType.STRING  then return self:_read_string() end
    if vtype == GGUFValueType.UINT64  then return self:_read_u64() end
    if vtype == GGUFValueType.INT64   then return self:_read_i64() end
    if vtype == GGUFValueType.FLOAT64 then return self:_read_f64() end

    if vtype == GGUFValueType.ARRAY then
        local elem_type = self:_read_u32()
        local count = tonumber(self:_read_u64())
        local arr = {}

        if elem_type == GGUFValueType.STRING then
            for i = 1, count do
                arr[i] = self:_read_string()
            end
        elseif elem_type == GGUFValueType.FLOAT32 then
            local ptr = ffi.cast("const float*", self:_read_bytes(count * 4))
            for i = 1, count do
                arr[i] = ptr[i - 1]
            end
        elseif elem_type == GGUFValueType.INT32 then
            local ptr = ffi.cast("const int32_t*", self:_read_bytes(count * 4))
            for i = 1, count do
                arr[i] = ptr[i - 1]
            end
        elseif elem_type == GGUFValueType.UINT32 then
            local ptr = ffi.cast("const uint32_t*", self:_read_bytes(count * 4))
            for i = 1, count do
                arr[i] = ptr[i - 1]
            end
        else
            for i = 1, count do
                arr[i] = self:_read_value(elem_type)
            end
        end
        return arr
    end

    error("Tipo GGUF desconhecido: " .. tostring(vtype))
end

function GNCXGGUF:_parse()
    -- Validar Assinatura Mágica
    local magic = ffi.string(self:_read_bytes(4), 4)
    if magic ~= "GGUF" then
        error("Arquivo inválido (magic não é GGUF: " .. tostring(magic) .. ")")
    end

    self.version = self:_read_u32()
    self.tensor_count = tonumber(self:_read_u64())
    self.kv_count = tonumber(self:_read_u64())

    -- 1. Parse de Metadados
    for _ = 1, self.kv_count do
        local key = self:_read_string()
        local vtype = self:_read_u32()
        local val = self:_read_value(vtype)
        self.metadata[key] = val
    end

    -- 2. Parse de Tensores
    local raw_tensors = {}
    for i = 1, self.tensor_count do
        local name = self:_read_string()
        local n_dims = self:_read_u32()
        local dims = {}
        for d = 1, n_dims do
            dims[d] = tonumber(self:_read_u64())
        end
        local ggml_type = self:_read_u32()
        local rel_offset = self:_read_u64()
        raw_tensors[i] = {
            name = name,
            dims = dims,
            ggml_type = ggml_type,
            rel_offset = rel_offset
        }
    end

    -- 3. Alinhamento da Seção de Dados
    local alignment = self.metadata["general.alignment"] or 32
    if type(alignment) ~= "number" then alignment = 32 end
    local rem = self.cursor % alignment
    if rem == 0 then
        self.data_offset = self.cursor
    else
        self.data_offset = self.cursor + (alignment - rem)
    end

    -- 4. Construção dos Descritores Geométricos
    for _, item in ipairs(raw_tensors) do
        local num_el = 1
        for _, d in ipairs(item.dims) do
            num_el = num_el * d
        end

        local byte_size = 0
        local binfo = GGML_BLOCK_INFO[item.ggml_type]
        if binfo then
            byte_size = math.floor(num_el / binfo[1]) * binfo[2]
        end

        local layer_idx = -1
        local blk_num = item.name:match("^blk%.(%d+)%.")
        if blk_num then
            layer_idx = tonumber(blk_num)
        end

        -- abs_file_offset como uint64_t FFI nativo para segurança além de 4GB
        local abs_offset = self.data_offset + item.rel_offset

        self.tensors[item.name] = {
            name            = item.name,
            shape           = item.dims,
            ggml_type       = item.ggml_type,
            rel_offset      = item.rel_offset,
            abs_file_offset = abs_offset,
            num_elements    = num_el,
            byte_size       = byte_size,
            layer_idx       = layer_idx,
            -- Ponteiro de memória virtual direto para leitura imediata
            ptr             = self.base + abs_offset,
        }
    end
end

-- Métodos públicos de conveniência
function GNCXGGUF:get_tensor(name)
    return self.tensors[name]
end

function GNCXGGUF:get_metadata(key, default)
    local v = self.metadata[key]
    if v == nil then return default end
    return v
end

return GNCXGGUF
