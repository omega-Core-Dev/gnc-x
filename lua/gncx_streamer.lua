--[[
  GNC-X LLM — Orquestrador de Camadas e Pipeline GLA em LuaJIT
  Fase 4: Motor de Inferência Zero-Overhead Multi-Perfil
  Suporta:
    - Perfil 1: Gemma 2B (18 Camadas | Q4_0 + Q8_0 | GeGLU | SentencePiece)
    - Perfil 2: Gemma 4 E2B (35 Camadas | Q4_K + Q6_K + PLE | GeGLU | SentencePiece)
    - Perfil 3: Qwen3 1.7B (28 Camadas | Q4_K + Q6_K | GQA + SwiGLU | Byte-Level BPE)
--]]

local ffi = require("ffi")
local gguf = require("lua.gncx_gguf")
local bpe  = require("lua.gncx_bpe")
local spiece = require("lua.gncx_spiece")

-- Declaração das estruturas e kernels C++20 nativos
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

    void* gncx_gla_map_readonly(const char* filepath, size_t* out_size);
    const uint8_t* gncx_gla_get_base_ptr(void* opaque_handle);
    void gncx_gla_unmap(void* opaque_handle);
    void gncx_gla_prefetch_range(const uint8_t* mmap_base, size_t offset, size_t num_bytes);

    void* gncx_ram_alloc(size_t total_bytes);
    void* gncx_ram_alloc_colored(size_t total_bytes, uint8_t k, uint8_t d, uint8_t R);
    uint8_t* gncx_ram_get_ptr(void* ram_handle);
    uint8_t gncx_ram_get_ddr3_bank(void* ram_handle);
    uint8_t gncx_ram_get_l1_set(void* ram_handle);
    void gncx_ram_free(void* ram_handle);
    void gncx_ram_copy_chunk(
        void* ram_handle,
        size_t ram_offset,
        const uint8_t* src_ptr,
        size_t src_offset,
        size_t num_bytes
    );

    void gncx_genome_get_layer_coordinate(
        int32_t layer_index,
        int32_t total_layers,
        int32_t base_k,
        int32_t* out_rel_layer,
        int32_t* out_dist_centro,
        int32_t* out_crivo_k,
        int32_t* out_hemisferio
    );

    void gncx_fetch_token_embedding_q8_0(
        const uint8_t* mmap_base,
        size_t tensor_abs_offset,
        int32_t token_id,
        int32_t embed_dim,
        float scale_factor,
        float* out_hidden
    );

    void gncx_fetch_token_embedding_q6_k(
        const uint8_t* mmap_base,
        size_t tensor_abs_offset,
        int32_t token_id,
        int32_t embed_dim,
        float scale_factor,
        float* out_hidden
    );

    void gncx_fetch_token_embedding_bf16(
        const uint8_t* mmap_base,
        size_t tensor_abs_offset,
        int32_t token_id,
        int32_t embed_dim,
        float scale_factor,
        float* out_hidden
    );

    void gncx_rmsnorm_and_encode_4phase(
        const float* input_hidden,
        const uint8_t* mmap_base,
        size_t norm_weight_offset,
        int32_t dim,
        float eps,
        int32_t layer_k,
        float* out_normed_fp32,
        GNCXCompactTokenStruct* out_tokens,
        float* out_quant_scale
    );

    void gncx_rmsnorm_inplace_heads(
        float* vec,
        const uint8_t* mmap_base,
        size_t norm_weight_offset,
        int32_t has_weight,
        int32_t n_heads,
        int32_t head_dim,
        float eps
    );

    void gncx_xfold_universal_matvec(
        const uint8_t* mmap_base,
        size_t weight_abs_offset,
        int32_t ggml_type,
        const float* in_vec,
        const GNCXCompactTokenStruct* in_tokens,
        int32_t in_dim,
        int32_t out_dim,
        float* out_vec,
        GNCXTelemetryStatsStruct* telemetry
    );

    void gncx_xfold_universal_matbatch(
        const uint8_t* mmap_base,
        size_t weight_abs_offset,
        int32_t ggml_type,
        const float* in_batch,
        int32_t batch_size,
        int32_t in_dim,
        int32_t out_dim,
        float* out_batch,
        GNCXTelemetryStatsStruct* telemetry
    );

    void gncx_universal_lm_head_logits(
        const uint8_t* mmap_base,
        size_t embd_abs_offset,
        int32_t ggml_type,
        const float* final_hidden,
        int32_t embed_dim,
        int32_t vocab_size,
        float softcap,
        float* out_logits
    );

    void gncx_apply_rope_xor11_ext(
        float* q_vec,
        float* k_vec,
        GNCXCompactTokenStruct* q_tokens,
        const uint8_t* mmap_base,
        size_t rope_freqs_offset,
        int32_t has_rope_freqs,
        int32_t n_heads,
        int32_t n_kv_heads,
        int32_t head_dim,
        int32_t pos,
        float freq_base
    );

    void gncx_mqa_attention_step_ext(
        const float* q_vec,
        const float* k_vec,
        const float* v_vec,
        float* k_cache_layer,
        float* v_cache_layer,
        int32_t pos,
        int32_t n_heads,
        int32_t n_kv_heads,
        int32_t head_dim,
        float attn_scale,
        int32_t sliding_window,
        int32_t write_kv,
        float* out_attn_vec
    );

    void gncx_geglu_activation(
        const float* gate_vec,
        const float* up_vec,
        int32_t ffn_dim,
        float* out_ffn_vec
    );

    void gncx_swiglu_activation(
        const float* gate_vec,
        const float* up_vec,
        int32_t ffn_dim,
        float* out_ffn_vec
    );

    // GNC-X Geometric Memory Manager (GMM)
    void* gncx_geo_mm_create(uint32_t min_frames);
    void  gncx_geo_mm_destroy(void* handle);
    int32_t gncx_geo_alloc(void* handle, uint8_t k_hint, uint8_t* out_k, uint8_t* out_d, uint8_t* out_R, uint32_t* out_linear_idx);
    void  gncx_geo_free(void* handle, uint8_t k, uint8_t d, uint8_t R, uint32_t linear_idx);
    float gncx_geo_fragmentation(void* handle);
    size_t gncx_geo_used(void* handle);
    size_t gncx_geo_capacity(void* handle);

    // Primitivas LoRA & MeZO (Upgrade v2.6)
    typedef struct {
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
    } __attribute__((packed)) GNCXLayer3DPoint;

    void gncx_lora_init_adapter(float* a_ptr, float* b_ptr, int32_t in_dim, int32_t out_dim, int32_t rank, uint64_t seed);
    void gncx_lora_forward_add(const float* in_vec, const float* a_ptr, const float* b_ptr, int32_t in_dim, int32_t out_dim, int32_t rank, float scale, float* io_out_vec);
    void gncx_lora_backward_step(const float* in_vec, const float* grad_out, float* a_ptr, float* b_ptr, float* grad_in_accum, int32_t in_dim, int32_t out_dim, int32_t rank, float scale, float lr, float weight_decay);
    void gncx_mezo_perturb_params(float* params, int32_t num_params, uint64_t seed, float epsilon);
    void gncx_mezo_update_params(float* params, int32_t num_params, uint64_t seed, float projected_grad, float lr, float weight_decay);
    float gncx_cross_entropy_loss(const float* logits, int32_t vocab_size, int32_t target_token_id, float* out_grad_logits);

    void gncx_project_layer_3d_point(
        int32_t layer_index,
        int32_t total_layers,
        const GNCXCompactTokenStruct* in_tokens,
        const float* normed_fp32,
        int32_t dim,
        GNCXLayer3DPoint* out_point
    );

    // Funções de Similaridade e Processamento de Embeddings (Fase 2)
    float gncx_cosine_similarity(const float* u, const float* v, int32_t dim);
    void  gncx_vector_l2_normalize(float* vec, int32_t dim);
    void  gncx_mean_pooling(const float* hidden_batch, int32_t batch_size, int32_t dim, float* out_pooled);
    void  gncx_bidirectional_attention(
        const float* q_batch,
        const float* k_batch,
        const float* v_batch,
        int32_t batch_size,
        int32_t n_heads,
        int32_t head_dim,
        float attn_scale,
        float* out_attn_batch
    );
    void  gncx_bidirectional_attention_gqa(
        const float* q_batch,
        const float* k_batch,
        const float* v_batch,
        int32_t batch_size,
        int32_t n_heads,
        int32_t n_kv_heads,
        int32_t head_dim,
        float attn_scale,
        float* out_attn_batch
    );
]]

local MODEL_PROFILES = {
    ["1"] = {
        id = "gemma2b",
        tag = "gemma/2b",
        filename = "gemma-2b-it-q4_0.gguf",
        label = "Gemma 2B Instruct (18 Camadas | Q4_0 + Q8_0 | 1.60 GB)",
    },
    ["gemma2b"] = {
        id = "gemma2b",
        tag = "gemma/2b",
        filename = "gemma-2b-it-q4_0.gguf",
        label = "Gemma 2B Instruct (18 Camadas | Q4_0 + Q8_0 | 1.60 GB)",
    },
    ["2"] = {
        id = "gemma4",
        tag = "gemma4/e2b",
        filename = "gemma4-e2b-it-q4_k.gguf",
        label = "Gemma 4 E2B Omni-Text (35 Camadas | Q4_K + Q6_K + PLE | 6.83 GB)",
    },
    ["gemma4"] = {
        id = "gemma4",
        tag = "gemma4/e2b",
        filename = "gemma4-e2b-it-q4_k.gguf",
        label = "Gemma 4 E2B Omni-Text (35 Camadas | Q4_K + Q6_K + PLE | 6.83 GB)",
    },
    ["3"] = {
        id = "qwen3",
        tag = "qwen3/1.7b",
        filename = "qwen3-1.7b-q4_k_m.gguf",
        label = "Qwen3 1.7B Instruct Alibaba (28 Camadas | Q4_K + Q6_K | GQA + SwiGLU | 1.05 GB)",
    },
    ["qwen3"] = {
        id = "qwen3",
        tag = "qwen3/1.7b",
        filename = "qwen3-1.7b-q4_k_m.gguf",
        label = "Qwen3 1.7B Instruct Alibaba (28 Camadas | Q4_K + Q6_K | GQA + SwiGLU | 1.05 GB)",
    },
}

local function file_exists(path)
    local f = io.open(path, "rb")
    if f then
        f:close()
        return true
    end
    return false
end

local function resolve_model_path(profile_or_tag)
    local key = tostring(profile_or_tag or "3"):lower():gsub("^%s*(.-)%s*$", "%1")
    if file_exists(key) then return key, key end

    local prof = MODEL_PROFILES[key] or MODEL_PROFILES["3"]
    local candidates = {
        "./models/" .. prof.filename,
        "models/" .. prof.filename,
        "../models/" .. prof.filename,
        prof.filename,
    }
    for _, c in ipairs(candidates) do
        if file_exists(c) then return c, prof.id end
    end
    error("Arquivo do modelo não encontrado para perfil: " .. key .. " (" .. prof.filename .. ")")
end

local GNCXStreamer = {}
GNCXStreamer.__index = GNCXStreamer
GNCXStreamer.MODEL_PROFILES = MODEL_PROFILES

function GNCXStreamer.new(model_path_or_tag, max_seq_len, pin_layers_in_ram, layer_k)
    max_seq_len = max_seq_len or 4096
    layer_k = layer_k or 3
    if pin_layers_in_ram == nil then pin_layers_in_ram = true end

    local model_path, profile_id = resolve_model_path(model_path_or_tag)

    local self = setmetatable({}, GNCXStreamer)
    self.profile_id = profile_id
    self.model_path = model_path
    self.max_seq_len = max_seq_len
    self.layer_k = layer_k
    self.pin_layers_in_ram = pin_layers_in_ram

    -- 1. Abre modelo GGUF via mapeamento Zero-Copy mmap
    local gmap, err = gguf.open(model_path)
    if not gmap then error(err) end
    self.gmap = gmap
    self.core = gmap.core
    self.mmap_base_ptr = gmap.base

    -- 2. Lê parâmetros arquiteturais do modelo
    local arch = tostring(gmap:get_metadata("general.architecture", "qwen3"))
    self.arch = arch
    self.is_qwen = arch:find("^qwen") ~= nil
    self.is_gemma4 = (arch == "gemma4")
    self.is_gemma = (arch == "gemma") or self.is_gemma4
    self.is_llama = arch:find("^llama") ~= nil

    self.n_layers = tonumber(gmap:get_metadata(arch .. ".block_count", 0))
    self.embed_dim = tonumber(gmap:get_metadata(arch .. ".embedding_length", 0))
    self.n_heads = tonumber(gmap:get_metadata(arch .. ".attention.head_count", 0))
    self.n_kv_heads = tonumber(gmap:get_metadata(arch .. ".attention.head_count_kv", self.n_heads))
    self.head_dim_global = tonumber(gmap:get_metadata(arch .. ".attention.key_length", math.floor(self.embed_dim / self.n_heads)))
    self.head_dim_swa = tonumber(gmap:get_metadata(arch .. ".attention.key_length_swa", self.head_dim_global))
    self.rms_eps = tonumber(gmap:get_metadata(arch .. ".attention.layer_norm_rms_epsilon", 1e-6))
    self.rope_freq_base_global = tonumber(gmap:get_metadata(arch .. ".rope.freq_base", 10000.0))
    self.rope_freq_base_swa = tonumber(gmap:get_metadata(arch .. ".rope.freq_base_swa", self.rope_freq_base_global))
    self.logit_softcap = tonumber(gmap:get_metadata(arch .. ".final_logit_softcapping", 0.0))

    local tokens = gmap:get_metadata("tokenizer.ggml.tokens")
    self.vocab_size = tokens and #tokens or 0

    local ffn_meta = gmap:get_metadata(arch .. ".feed_forward_length")
    self.layer_ffn_dims = {}
    if type(ffn_meta) == "table" then
        for l = 1, self.n_layers do
            self.layer_ffn_dims[l] = tonumber(ffn_meta[l])
        end
    else
        local f_dim = tonumber(ffn_meta) or 0
        for l = 1, self.n_layers do
            self.layer_ffn_dims[l] = f_dim
        end
    end
    self.max_ffn_dim = 0
    for l = 1, self.n_layers do
        if self.layer_ffn_dims[l] > self.max_ffn_dim then
            self.max_ffn_dim = self.layer_ffn_dims[l]
        end
    end

    -- Configuração por camada (SWA vs Global, compartilhamento de KV no Gemma 4)
    local swa_pattern = gmap:get_metadata(arch .. ".attention.sliding_window_pattern")
    self.layer_is_swa = {}
    for l = 1, self.n_layers do
        if type(swa_pattern) == "table" and swa_pattern[l] ~= nil then
            self.layer_is_swa[l] = (swa_pattern[l] == true or swa_pattern[l] == 1)
        else
            self.layer_is_swa[l] = false
        end
    end

    self.sliding_window_size = tonumber(gmap:get_metadata(arch .. ".attention.sliding_window", 0))
    self.shared_kv_layers = tonumber(gmap:get_metadata(arch .. ".attention.shared_kv_layers", 0))
    self.first_kv_shared_layer_idx = (self.shared_kv_layers > 0) and (self.n_layers - self.shared_kv_layers) or self.n_layers

    local last_non_shared = { [false] = 0, [true] = 0 }
    for l = 0, self.first_kv_shared_layer_idx - 1 do
        last_non_shared[self.layer_is_swa[l + 1]] = l
    end

    self.layer_is_kv_shared = {}
    self.layer_kv_src_idx = {}
    self.layer_head_dims = {}
    self.layer_q_dims = {}
    self.layer_kv_dims = {}
    self.layer_out_scales = {}

    for l = 0, self.n_layers - 1 do
        local is_swa = self.layer_is_swa[l + 1]
        local is_sh = (l >= self.first_kv_shared_layer_idx)
        self.layer_is_kv_shared[l] = is_sh
        self.layer_kv_src_idx[l] = is_sh and last_non_shared[is_swa] or l
        local h_dim = is_swa and self.head_dim_swa or self.head_dim_global
        self.layer_head_dims[l] = h_dim
        self.layer_q_dims[l] = self.n_heads * h_dim
        self.layer_kv_dims[l] = self.n_kv_heads * h_dim

        local scale_name = string.format("blk.%d.layer_output_scale.weight", l)
        local st = gmap:get_tensor(scale_name)
        if st then
            local fp = ffi.cast("float*", st.ptr)
            self.layer_out_scales[l] = tonumber(fp[0])
        else
            self.layer_out_scales[l] = 1.0
        end
    end

    self.max_q_dim = 0
    self.max_kv_dim = 0
    for l = 0, self.n_layers - 1 do
        if self.layer_q_dims[l] > self.max_q_dim then self.max_q_dim = self.layer_q_dims[l] end
        if self.layer_kv_dims[l] > self.max_kv_dim then self.max_kv_dim = self.layer_kv_dims[l] end
    end

    -- Per-Layer Embeddings (Gemma 4 E2B)
    self.ple_dim = tonumber(gmap:get_metadata(arch .. ".embedding_length_per_layer_input", 0))
    self.ple_total_dim = self.ple_dim * self.n_layers

    -- 3. Tokenizador Especializado (SentencePiece para Gemma vs Byte-Level BPE para Qwen/LLaMA)
    local merges = gmap:get_metadata("tokenizer.ggml.merges")
    local scores = gmap:get_metadata("tokenizer.ggml.scores")
    local ttypes = gmap:get_metadata("tokenizer.ggml.token_type")

    local default_eos = self.is_gemma and 1 or 151645
    local default_bos = self.is_gemma and 2 or -1
    local eos_id = tonumber(gmap:get_metadata("tokenizer.ggml.eos_token_id", default_eos))
    local bos_id = tonumber(gmap:get_metadata("tokenizer.ggml.bos_token_id", default_bos))

    if self.is_gemma or (scores ~= nil and not self.is_qwen) then
        self.tokenizer = spiece.new(tokens, scores, ttypes, bos_id, eos_id)
    else
        self.tokenizer = bpe.new(tokens, merges, ttypes, bos_id, eos_id)
    end

    -- 4. Alocação dos buffers de cálculo em RAM com Coloração Diagonal DDR3/L1d
    self.scratch_handles = {}
    local function alloc_colored_floats(n_floats, slot_d)
        local h = self.core.gncx_ram_alloc_colored(math.max(n_floats * 4, 4096), 1, slot_d, 3)
        table.insert(self.scratch_handles, h)
        return ffi.cast("float*", self.core.gncx_ram_get_ptr(h))
    end

    self.hidden       = alloc_colored_floats(self.embed_dim, 0)
    self.normed       = alloc_colored_floats(self.embed_dim, 1)
    self.q_buf        = alloc_colored_floats(self.max_q_dim, 2)
    self.k_buf        = alloc_colored_floats(self.max_kv_dim, 3)
    self.v_buf        = alloc_colored_floats(self.max_kv_dim, 4)
    self.attn_out_buf = alloc_colored_floats(self.max_q_dim, 5)
    self.proj_out_buf = alloc_colored_floats(math.max(self.embed_dim, self.ple_total_dim), 6)
    self.gate_buf     = alloc_colored_floats(self.max_ffn_dim, 7)
    self.up_buf       = alloc_colored_floats(self.max_ffn_dim, 8)
    self.ffn_act_buf  = alloc_colored_floats(self.max_ffn_dim, 9)
    self.logits_buf   = alloc_colored_floats(self.vocab_size, 10)

    if self.ple_dim > 0 then
        self.ple_tok_buf  = ffi.new("float[?]", self.ple_total_dim)
        self.ple_proj_buf = ffi.new("float[?]", self.ple_total_dim)
        self.ple_gate_buf = ffi.new("float[?]", self.ple_dim)
    end

    local max_token_buf_dim = math.max(self.embed_dim, self.max_q_dim)
    self.x_tokens_embed = ffi.new("GNCXCompactTokenStruct[?]", max_token_buf_dim)
    self.quant_scale    = ffi.new("float[1]")
    self.telemetry      = ffi.new("GNCXTelemetryStatsStruct", 0, 0, 0, 0)

    -- 5. Genome VM: Origem Central (Equador) e Coordenadas do Crivo D(k, d)
    self.origem_central = math.floor(self.n_layers / 2)
    self.layer_rel_coords = {}
    self.layer_dist_centro = {}
    self.layer_crivo_k = {}
    self.layer_hemisferios = {}

    local rel_p  = ffi.new("int32_t[1]")
    local dist_p = ffi.new("int32_t[1]")
    local k_p    = ffi.new("int32_t[1]")
    local hemi_p = ffi.new("int32_t[1]")

    for l = 0, self.n_layers - 1 do
        self.core.gncx_genome_get_layer_coordinate(
            l, self.n_layers, self.layer_k,
            rel_p, dist_p, k_p, hemi_p
        )
        self.layer_rel_coords[l]  = tonumber(rel_p[0])
        self.layer_dist_centro[l] = tonumber(dist_p[0])
        self.layer_crivo_k[l]     = tonumber(k_p[0])
        self.layer_hemisferios[l] = tonumber(hemi_p[0])
    end

    -- 6. Alocação de KV-Cache Cirúrgico gerenciado pelo GeometricMemoryManager
    self.gmm = self.core.gncx_geo_mm_create(self.n_layers * 2)
    self.geo_allocations = {}
    self.k_cache_handles = {}
    self.v_cache_handles = {}
    self.k_cache_ptrs = {}
    self.v_cache_ptrs = {}

    local k_out = ffi.new("uint8_t[1]")
    local d_out = ffi.new("uint8_t[1]")
    local r_out = ffi.new("uint8_t[1]")
    local idx_out = ffi.new("uint32_t[1]")

    local total_kv_ram_bytes = 0
    for l = 0, self.n_layers - 1 do
        if self.layer_is_kv_shared[l] then
            local src = self.layer_kv_src_idx[l]
            self.k_cache_ptrs[l] = self.k_cache_ptrs[src]
            self.v_cache_ptrs[l] = self.v_cache_ptrs[src]
        else
            local crivo_k = self.layer_crivo_k[l] or 2

            local kv_bytes = self.max_seq_len * self.layer_kv_dims[l] * 4

            -- Alocação no Crivo Diagonal D(k, d) + Coloração Física DDR3/L1d para Key Cache
            self.core.gncx_geo_alloc(self.gmm, crivo_k, k_out, d_out, r_out, idx_out)
            local kh = self.core.gncx_ram_alloc_colored(kv_bytes, k_out[0], d_out[0], r_out[0])
            table.insert(self.geo_allocations, {
                layer = l, kind = "Key", k = k_out[0], d = d_out[0], R = r_out[0], idx = idx_out[0],
                ddr3_bank = tonumber(self.core.gncx_ram_get_ddr3_bank(kh)),
                l1_set    = tonumber(self.core.gncx_ram_get_l1_set(kh))
            })

            -- Alocação no Crivo Diagonal D(k, d) + Coloração Física DDR3/L1d para Value Cache
            self.core.gncx_geo_alloc(self.gmm, crivo_k, k_out, d_out, r_out, idx_out)
            local vh = self.core.gncx_ram_alloc_colored(kv_bytes, k_out[0], d_out[0], r_out[0])
            table.insert(self.geo_allocations, {
                layer = l, kind = "Val", k = k_out[0], d = d_out[0], R = r_out[0], idx = idx_out[0],
                ddr3_bank = tonumber(self.core.gncx_ram_get_ddr3_bank(vh)),
                l1_set    = tonumber(self.core.gncx_ram_get_l1_set(vh))
            })

            if kh == nil or vh == nil then
                error("Falha ao alocar KV-Cache na RAM para camada " .. l)
            end
            table.insert(self.k_cache_handles, kh)
            table.insert(self.v_cache_handles, vh)
            self.k_cache_ptrs[l] = ffi.cast("float*", self.core.gncx_ram_get_ptr(kh))
            self.v_cache_ptrs[l] = ffi.cast("float*", self.core.gncx_ram_get_ptr(vh))
            total_kv_ram_bytes = total_kv_ram_bytes + (kv_bytes * 2)
        end
    end

    -- 7. Fixação de Camadas em RAM (RAM Pinning opcional)
    self.tensor_base_ptr = {}
    self.tensor_rel_offset = {}
    self.ram_buffer_handle = nil
    self.ram_buffer_ptr = nil
    self.layers_in_ram = false

    if self.pin_layers_in_ram then
        self:_pin_compute_layers_into_ram()
    else
        self:_init_mmap_tensor_addressing()
    end

    return self
end

function GNCXStreamer:_pin_compute_layers_into_ram()
    local pinned_names = {}
    for name, _ in pairs(self.gmap.tensors) do
        if name:find("^blk%.") or name == "per_layer_model_proj.weight"
           or name == "per_layer_proj_norm.weight" or name == "rope_freqs.weight"
           or name == "output_norm.weight" then
            table.insert(pinned_names, name)
        end
    end

    -- Ordenação física por offset de arquivo para leitura sequencial rápida
    table.sort(pinned_names, function(a, b)
        return self.gmap.tensors[a].abs_file_offset < self.gmap.tensors[b].abs_file_offset
    end)

    local cur_offset = 0
    for idx, name in ipairs(pinned_names) do
        local t = self.gmap.tensors[name]
        -- Alinhamento a 64 bytes + Deslocamento Coprimo GNC-X de Linha de Cache L1d/L2
        -- (Impede que matrizes consecutivas comecem sempre no Set #0 do Cache L1d!)
        cur_offset = bit.band(cur_offset + 63, bit.bnot(63))
        local color_pad = ((idx * 3) % 16) * 64
        cur_offset = cur_offset + color_pad
        self.tensor_rel_offset[name] = cur_offset
        cur_offset = cur_offset + t.byte_size
    end

    local total_bytes = cur_offset
    local handle = self.core.gncx_ram_alloc(total_bytes)
    if handle == nil then
        self:_init_mmap_tensor_addressing()
        return
    end

    self.ram_buffer_handle = handle
    self.ram_buffer_ptr = self.core.gncx_ram_get_ptr(handle)

    for _, name in ipairs(pinned_names) do
        local t = self.gmap.tensors[name]
        local dst_off = self.tensor_rel_offset[name]
        self.tensor_base_ptr[name] = self.ram_buffer_ptr
        self.core.gncx_ram_copy_chunk(
            self.ram_buffer_handle,
            dst_off,
            self.mmap_base_ptr,
            t.abs_file_offset,
            t.byte_size
        )
    end

    -- Tensores não-pinados (ex: token_embd.weight, per_layer_token_embd.weight)
    for name, t in pairs(self.gmap.tensors) do
        if not self.tensor_base_ptr[name] then
            self.tensor_base_ptr[name] = self.mmap_base_ptr
            self.tensor_rel_offset[name] = t.abs_file_offset
        end
    end

    self.layers_in_ram = true
end

function GNCXStreamer:_init_mmap_tensor_addressing()
    for name, t in pairs(self.gmap.tensors) do
        self.tensor_base_ptr[name] = self.mmap_base_ptr
        self.tensor_rel_offset[name] = t.abs_file_offset
    end
    self.layers_in_ram = false
end

function GNCXStreamer:_tensor_base_and_offset(tensor_name)
    local base = self.tensor_base_ptr[tensor_name]
    local off = self.tensor_rel_offset[tensor_name]
    if base ~= nil and off ~= nil then
        return base, off
    end
    local t = self.gmap.tensors[tensor_name]
    if t then
        return self.mmap_base_ptr, t.abs_file_offset
    end
    return nil, 0
end

function GNCXStreamer:_matvec(tensor_name, in_vec, in_tokens, in_dim, out_dim, out_vec)
    local t = self.gmap.tensors[tensor_name]
    if not t then error("Tensor ausente: " .. tensor_name) end
    local base, off = self:_tensor_base_and_offset(tensor_name)
    self.core.gncx_xfold_universal_matvec(
        base,
        off,
        t.ggml_type,
        in_vec,
        in_tokens,
        in_dim,
        out_dim,
        out_vec,
        self.telemetry
    )
end

function GNCXStreamer:_matbatch(tensor_name, in_batch, batch_size, in_dim, out_dim, out_batch)
    local t = self.gmap.tensors[tensor_name]
    if not t then error("Tensor ausente: " .. tensor_name) end
    local base, off = self:_tensor_base_and_offset(tensor_name)
    self.core.gncx_xfold_universal_matbatch(
        base,
        off,
        t.ggml_type,
        in_batch,
        batch_size,
        in_dim,
        out_dim,
        out_batch,
        self.telemetry
    )
end

function GNCXStreamer:_fetch_embedding(token_id, dst_ptr)
    local t = self.gmap.tensors["token_embd.weight"]
    if not t then error("token_embd.weight ausente no GGUF") end
    local base, off = self:_tensor_base_and_offset("token_embd.weight")
    local scale = (self.is_qwen or self.is_llama) and 1.0 or math.sqrt(self.embed_dim)

    if t.ggml_type == 8 then -- Q8_0 (Gemma 2B)
        self.core.gncx_fetch_token_embedding_q8_0(base, off, token_id, self.embed_dim, scale, dst_ptr)
    elseif t.ggml_type == 14 then -- Q6_K (Gemma 4 E2B, Qwen3)
        self.core.gncx_fetch_token_embedding_q6_k(base, off, token_id, self.embed_dim, scale, dst_ptr)
    elseif t.ggml_type == 30 then -- BF16
        self.core.gncx_fetch_token_embedding_bf16(base, off, token_id, self.embed_dim, scale, dst_ptr)
    else
        error("Tipo de embedding não suportado: " .. tostring(t.ggml_type))
    end
end

-- Prepara Per-Layer Embeddings (PLE) para Gemma 4 E2B
function GNCXStreamer:_prepare_ple_single(token_id, hidden_vec)
    local core = self.core
    local ple_scale = math.sqrt(self.ple_dim)
    local base_ple, off_ple = self:_tensor_base_and_offset("per_layer_token_embd.weight")
    core.gncx_fetch_token_embedding_bf16(
        base_ple,
        off_ple,
        token_id,
        self.ple_total_dim,
        ple_scale,
        self.ple_tok_buf
    )
    self:_matvec(
        "per_layer_model_proj.weight",
        hidden_vec,
        nil,
        self.embed_dim,
        self.ple_total_dim,
        self.ple_proj_buf
    )
    local proj_scale = 1.0 / math.sqrt(self.embed_dim)
    for i = 0, self.ple_total_dim - 1 do
        self.ple_proj_buf[i] = self.ple_proj_buf[i] * proj_scale
    end

    local base_norm, off_norm = self:_tensor_base_and_offset("per_layer_proj_norm.weight")
    core.gncx_rmsnorm_inplace_heads(
        self.ple_proj_buf,
        base_norm,
        off_norm,
        1,
        self.n_layers,
        self.ple_dim,
        self.rms_eps
    )

    local inv_sqrt2 = 1.0 / math.sqrt(2.0)
    for i = 0, self.ple_total_dim - 1 do
        self.ple_tok_buf[i] = (self.ple_proj_buf[i] + self.ple_tok_buf[i]) * inv_sqrt2
    end
end

-- Forward Pass de 1 único token na posição `pos`
function GNCXStreamer:forward_token(token_id, pos, compute_logits)
    if compute_logits == nil then compute_logits = true end
    if pos >= self.max_seq_len then return nil end

    local core = self.core

    -- 1. Busca embedding do token no disco GLA
    self:_fetch_embedding(token_id, self.hidden)

    -- 1B. Se for Gemma 4 (PLE ativo), prepara as entradas de cada camada
    if self.ple_dim > 0 then
        self:_prepare_ple_single(token_id, self.hidden)
    end

    local has_rf = self.gmap.tensors["rope_freqs.weight"] ~= nil and 1 or 0
    local base_rf, off_rf = nil, 0
    if has_rf == 1 then
        base_rf, off_rf = self:_tensor_base_and_offset("rope_freqs.weight")
    end

    -- 2. Percorre as camadas geométricas
    for l = 0, self.n_layers - 1 do
        local prefix = string.format("blk.%d", l)
        local is_swa = self.layer_is_swa[l + 1]
        local h_dim  = self.layer_head_dims[l]
        local q_dim  = self.layer_q_dims[l]
        local kv_dim = self.layer_kv_dims[l]
        local ffn_dim = self.layer_ffn_dims[l + 1]
        local is_sh  = self.layer_is_kv_shared[l]
        local src_l  = self.layer_kv_src_idx[l]
        local crivo_k = self.layer_crivo_k[l]

        -- 2.1. RMSNorm + Codificação 4-Phase GNC-X
        local base_an, off_an = self:_tensor_base_and_offset(prefix .. ".attn_norm.weight")
        core.gncx_rmsnorm_and_encode_4phase(
            self.hidden,
            base_an,
            off_an,
            self.embed_dim,
            self.rms_eps,
            crivo_k,
            self.normed,
            self.x_tokens_embed,
            self.quant_scale
        )

        -- 2.2. Projeções Q, K, V
        self:_matvec(prefix .. ".attn_q.weight", self.normed, self.x_tokens_embed, self.embed_dim, q_dim, self.q_buf)
        if not is_sh then
            self:_matvec(prefix .. ".attn_k.weight", self.normed, self.x_tokens_embed, self.embed_dim, kv_dim, self.k_buf)
            self:_matvec(prefix .. ".attn_v.weight", self.normed, self.x_tokens_embed, self.embed_dim, kv_dim, self.v_buf)
        end

        -- 2.3. Normalização Head-wise de Q e K (Qwen3 e Gemma 4)
        if self.is_gemma4 or self.is_qwen then
            local qn_name = prefix .. ".attn_q_norm.weight"
            if self.gmap.tensors[qn_name] then
                local b_qn, o_qn = self:_tensor_base_and_offset(qn_name)
                core.gncx_rmsnorm_inplace_heads(self.q_buf, b_qn, o_qn, 1, self.n_heads, h_dim, self.rms_eps)
            end
            if not is_sh then
                local kn_name = prefix .. ".attn_k_norm.weight"
                if self.gmap.tensors[kn_name] then
                    local b_kn, o_kn = self:_tensor_base_and_offset(kn_name)
                    core.gncx_rmsnorm_inplace_heads(self.k_buf, b_kn, o_kn, 1, self.n_kv_heads, h_dim, self.rms_eps)
                end
            end
        end

        -- 2.4. RoPE com XOR-11
        local freq_base = (self.is_gemma4 and is_swa) and self.rope_freq_base_swa or self.rope_freq_base_global
        local use_rf = (self.is_gemma4 and not is_swa and has_rf == 1) and 1 or 0
        core.gncx_apply_rope_xor11_ext(
            self.q_buf,
            is_sh and nil or self.k_buf,
            self.x_tokens_embed,
            base_rf,
            off_rf,
            use_rf,
            self.n_heads,
            self.n_kv_heads,
            h_dim,
            pos,
            freq_base
        )

        -- 2.5. Attention MQA/GQA Step
        local attn_scale = self.is_gemma4 and 1.0 or (1.0 / math.sqrt(h_dim))
        local swa_win = (self.is_gemma4 and is_swa) and self.sliding_window_size or 0
        core.gncx_mqa_attention_step_ext(
            self.q_buf,
            is_sh and nil or self.k_buf,
            is_sh and nil or self.v_buf,
            self.k_cache_ptrs[src_l],
            self.v_cache_ptrs[src_l],
            pos,
            self.n_heads,
            self.n_kv_heads,
            h_dim,
            attn_scale,
            swa_win,
            is_sh and 0 or 1,
            self.attn_out_buf
        )

        -- 2.6. Projeção Attn Out + Post-Attention Norm + Residual
        self:_matvec(prefix .. ".attn_output.weight", self.attn_out_buf, nil, q_dim, self.embed_dim, self.proj_out_buf)
        if self.is_gemma4 then
            local pan_name = prefix .. ".post_attention_norm.weight"
            local b_pan, o_pan = self:_tensor_base_and_offset(pan_name)
            core.gncx_rmsnorm_inplace_heads(self.proj_out_buf, b_pan, o_pan, 1, 1, self.embed_dim, self.rms_eps)
        end
        for i = 0, self.embed_dim - 1 do
            self.hidden[i] = self.hidden[i] + self.proj_out_buf[i]
        end

        -- 2.7. FFN RMSNorm + 4-Phase Encode
        local base_fn, off_fn = self:_tensor_base_and_offset(prefix .. ".ffn_norm.weight")
        core.gncx_rmsnorm_and_encode_4phase(
            self.hidden,
            base_fn,
            off_fn,
            self.embed_dim,
            self.rms_eps,
            crivo_k,
            self.normed,
            self.x_tokens_embed,
            self.quant_scale
        )

        -- 2.8. FFN Gate & Up + Ativação SwiGLU / GeGLU
        self:_matvec(prefix .. ".ffn_gate.weight", self.normed, self.x_tokens_embed, self.embed_dim, ffn_dim, self.gate_buf)
        self:_matvec(prefix .. ".ffn_up.weight",   self.normed, self.x_tokens_embed, self.embed_dim, ffn_dim, self.up_buf)

        if self.is_qwen or self.is_llama then
            core.gncx_swiglu_activation(self.gate_buf, self.up_buf, ffn_dim, self.ffn_act_buf)
        else
            core.gncx_geglu_activation(self.gate_buf, self.up_buf, ffn_dim, self.ffn_act_buf)
        end

        -- 2.9. FFN Down + Post-FFW Norm + Residual
        self:_matvec(prefix .. ".ffn_down.weight", self.ffn_act_buf, nil, ffn_dim, self.embed_dim, self.proj_out_buf)
        if self.is_gemma4 then
            local pfn_name = prefix .. ".post_ffw_norm.weight"
            local b_pfn, o_pfn = self:_tensor_base_and_offset(pfn_name)
            core.gncx_rmsnorm_inplace_heads(self.proj_out_buf, b_pfn, o_pfn, 1, 1, self.embed_dim, self.rms_eps)
        end
        for i = 0, self.embed_dim - 1 do
            self.hidden[i] = self.hidden[i] + self.proj_out_buf[i]
        end

        -- 2.10. Injeção de Per-Layer Embeddings (Gemma 4 PLE)
        if self.ple_dim > 0 then
            self:_matvec(prefix .. ".inp_gate.weight", self.hidden, nil, self.embed_dim, self.ple_dim, self.ple_gate_buf)
            local ple_slice_ptr = self.ple_tok_buf + (l * self.ple_dim)
            core.gncx_geglu_activation(self.ple_gate_buf, ple_slice_ptr, self.ple_dim, self.ple_gate_buf)
            self:_matvec(prefix .. ".proj.weight", self.ple_gate_buf, nil, self.ple_dim, self.embed_dim, self.proj_out_buf)
            local pn_name = prefix .. ".post_norm.weight"
            local b_pn, o_pn = self:_tensor_base_and_offset(pn_name)
            core.gncx_rmsnorm_inplace_heads(self.proj_out_buf, b_pn, o_pn, 1, 1, self.embed_dim, self.rms_eps)
            for i = 0, self.embed_dim - 1 do
                self.hidden[i] = self.hidden[i] + self.proj_out_buf[i]
            end
        end

        -- 2.11. Escala de saída da camada
        local out_scale = self.layer_out_scales[l]
        if out_scale ~= 1.0 then
            for i = 0, self.embed_dim - 1 do
                self.hidden[i] = self.hidden[i] * out_scale
            end
        end
    end

    if not compute_logits then
        return nil
    end

    -- 3. Output RMSNorm Final + Projeção LM Head
    local base_on, off_on = self:_tensor_base_and_offset("output_norm.weight")
    core.gncx_rmsnorm_and_encode_4phase(
        self.hidden,
        base_on,
        off_on,
        self.embed_dim,
        self.rms_eps,
        self.layer_k,
        self.normed,
        self.x_tokens_embed,
        self.quant_scale
    )

    local embd_t = self.gmap.tensors["token_embd.weight"]
    local base_lm, off_lm = self:_tensor_base_and_offset("token_embd.weight")
    core.gncx_universal_lm_head_logits(
        base_lm,
        off_lm,
        embd_t.ggml_type,
        self.normed,
        self.embed_dim,
        self.vocab_size,
        self.logit_softcap,
        self.logits_buf
    )

    return self.logits_buf
end

-- Processa todos os tokens do prompt em lote (1 única varredura das camadas na DDR3!)
function GNCXStreamer:prefill(token_ids)
    local B = #token_ids
    if B == 0 then return nil end
    if B == 1 then
        return self:forward_token(token_ids[1], 0, true)
    end

    local core = self.core
    local temp_handles = {}
    local function alloc_batch_buf(n_floats, slot_d)
        local h = core.gncx_ram_alloc_colored(math.max(n_floats * 4, 4096), 2, slot_d, 5)
        table.insert(temp_handles, h)
        return ffi.cast("float*", core.gncx_ram_get_ptr(h))
    end

    local b_hidden   = alloc_batch_buf(B * self.embed_dim,   0)
    local b_normed   = alloc_batch_buf(B * self.embed_dim,   1)
    local b_q        = alloc_batch_buf(B * self.max_q_dim,   2)
    local b_k        = alloc_batch_buf(B * self.max_kv_dim,  3)
    local b_v        = alloc_batch_buf(B * self.max_kv_dim,  4)
    local b_attn_out = alloc_batch_buf(B * self.max_q_dim,   5)
    local b_proj_out = alloc_batch_buf(B * self.embed_dim,   6)
    local b_gate     = alloc_batch_buf(B * self.max_ffn_dim, 7)
    local b_up       = alloc_batch_buf(B * self.max_ffn_dim, 8)
    local b_ffn_act  = alloc_batch_buf(B * self.max_ffn_dim, 9)

    local b_ple_tok, b_ple_proj, b_ple_gate = nil, nil, nil
    if self.ple_dim > 0 then
        b_ple_tok  = alloc_batch_buf(B * self.ple_total_dim, 10)
        b_ple_proj = alloc_batch_buf(B * self.ple_total_dim, 11)
        b_ple_gate = alloc_batch_buf(B * self.ple_dim,       12)
    end

    for t_idx = 0, B - 1 do
        self:_fetch_embedding(token_ids[t_idx + 1], b_hidden + (t_idx * self.embed_dim))
    end

    if self.ple_dim > 0 then
        local ple_scale = math.sqrt(self.ple_dim)
        local base_ple, off_ple = self:_tensor_base_and_offset("per_layer_token_embd.weight")
        for t_idx = 0, B - 1 do
            core.gncx_fetch_token_embedding_bf16(
                base_ple,
                off_ple,
                token_ids[t_idx + 1],
                self.ple_total_dim,
                ple_scale,
                b_ple_tok + (t_idx * self.ple_total_dim)
            )
        end
        self:_matbatch("per_layer_model_proj.weight", b_hidden, B, self.embed_dim, self.ple_total_dim, b_ple_proj)
        local proj_scale = 1.0 / math.sqrt(self.embed_dim)
        for i = 0, (B * self.ple_total_dim) - 1 do
            b_ple_proj[i] = b_ple_proj[i] * proj_scale
        end
        local base_norm, off_norm = self:_tensor_base_and_offset("per_layer_proj_norm.weight")
        for t_idx = 0, B - 1 do
            core.gncx_rmsnorm_inplace_heads(
                b_ple_proj + (t_idx * self.ple_total_dim),
                base_norm,
                off_norm,
                1,
                self.n_layers,
                self.ple_dim,
                self.rms_eps
            )
        end
        local inv_sqrt2 = 1.0 / math.sqrt(2.0)
        for i = 0, (B * self.ple_total_dim) - 1 do
            b_ple_tok[i] = (b_ple_proj[i] + b_ple_tok[i]) * inv_sqrt2
        end
    end

    local has_rf = self.gmap.tensors["rope_freqs.weight"] ~= nil and 1 or 0
    local base_rf, off_rf = nil, 0
    if has_rf == 1 then
        base_rf, off_rf = self:_tensor_base_and_offset("rope_freqs.weight")
    end

    for l = 0, self.n_layers - 1 do
        local prefix  = string.format("blk.%d", l)
        local is_swa  = self.layer_is_swa[l + 1]
        local h_dim   = self.layer_head_dims[l]
        local q_dim   = self.layer_q_dims[l]
        local kv_dim  = self.layer_kv_dims[l]
        local ffn_dim = self.layer_ffn_dims[l + 1]
        local is_sh   = self.layer_is_kv_shared[l]
        local src_l   = self.layer_kv_src_idx[l]
        local crivo_k = self.layer_crivo_k[l]

        local base_an, off_an = self:_tensor_base_and_offset(prefix .. ".attn_norm.weight")
        for t_idx = 0, B - 1 do
            core.gncx_rmsnorm_and_encode_4phase(
                b_hidden + (t_idx * self.embed_dim),
                base_an, off_an, self.embed_dim, self.rms_eps, crivo_k,
                b_normed + (t_idx * self.embed_dim),
                nil, nil
            )
        end

        self:_matbatch(prefix .. ".attn_q.weight", b_normed, B, self.embed_dim, q_dim, b_q)
        if not is_sh then
            self:_matbatch(prefix .. ".attn_k.weight", b_normed, B, self.embed_dim, kv_dim, b_k)
            self:_matbatch(prefix .. ".attn_v.weight", b_normed, B, self.embed_dim, kv_dim, b_v)
        end

        local freq_base  = (self.is_gemma4 and is_swa) and self.rope_freq_base_swa or self.rope_freq_base_global
        local use_rf     = (self.is_gemma4 and not is_swa and has_rf == 1) and 1 or 0
        local attn_scale = self.is_gemma4 and 1.0 or (1.0 / math.sqrt(h_dim))
        local swa_win    = (self.is_gemma4 and is_swa) and self.sliding_window_size or 0

        for t_idx = 0, B - 1 do
            local qp = b_q + (t_idx * q_dim)
            local kp = is_sh and nil or (b_k + (t_idx * kv_dim))
            local vp = is_sh and nil or (b_v + (t_idx * kv_dim))
            local ap = b_attn_out + (t_idx * q_dim)

            if self.is_qwen or self.is_gemma4 then
                local qn_name = prefix .. ".attn_q_norm.weight"
                if self.gmap.tensors[qn_name] then
                    local b_qn, o_qn = self:_tensor_base_and_offset(qn_name)
                    core.gncx_rmsnorm_inplace_heads(qp, b_qn, o_qn, 1, self.n_heads, h_dim, self.rms_eps)
                end
                if not is_sh then
                    local kn_name = prefix .. ".attn_k_norm.weight"
                    if self.gmap.tensors[kn_name] then
                        local b_kn, o_kn = self:_tensor_base_and_offset(kn_name)
                        core.gncx_rmsnorm_inplace_heads(kp, b_kn, o_kn, 1, self.n_kv_heads, h_dim, self.rms_eps)
                    end
                end
            end

            core.gncx_apply_rope_xor11_ext(
                qp, kp, self.x_tokens_embed, base_rf, off_rf, use_rf,
                self.n_heads, self.n_kv_heads, h_dim, t_idx, freq_base
            )
            core.gncx_mqa_attention_step_ext(
                qp, kp, vp, self.k_cache_ptrs[src_l], self.v_cache_ptrs[src_l],
                t_idx, self.n_heads, self.n_kv_heads, h_dim, attn_scale, swa_win,
                is_sh and 0 or 1, ap
            )
        end

        self:_matbatch(prefix .. ".attn_output.weight", b_attn_out, B, q_dim, self.embed_dim, b_proj_out)
        if self.is_gemma4 then
            local pan_name = prefix .. ".post_attention_norm.weight"
            if self.gmap.tensors[pan_name] then
                local b_pan, o_pan = self:_tensor_base_and_offset(pan_name)
                for t_idx = 0, B - 1 do
                    core.gncx_rmsnorm_inplace_heads(b_proj_out + (t_idx * self.embed_dim), b_pan, o_pan, 1, 1, self.embed_dim, self.rms_eps)
                end
            end
        end
        for i = 0, (B * self.embed_dim) - 1 do
            b_hidden[i] = b_hidden[i] + b_proj_out[i]
        end

        local base_fn, off_fn = self:_tensor_base_and_offset(prefix .. ".ffn_norm.weight")
        for t_idx = 0, B - 1 do
            core.gncx_rmsnorm_and_encode_4phase(
                b_hidden + (t_idx * self.embed_dim),
                base_fn, off_fn, self.embed_dim, self.rms_eps, crivo_k,
                b_normed + (t_idx * self.embed_dim),
                nil, nil
            )
        end

        self:_matbatch(prefix .. ".ffn_gate.weight", b_normed, B, self.embed_dim, ffn_dim, b_gate)
        self:_matbatch(prefix .. ".ffn_up.weight",   b_normed, B, self.embed_dim, ffn_dim, b_up)
        if self.is_qwen or self.is_llama then
            core.gncx_swiglu_activation(b_gate, b_up, B * ffn_dim, b_ffn_act)
        else
            core.gncx_geglu_activation(b_gate, b_up, B * ffn_dim, b_ffn_act)
        end
        self:_matbatch(prefix .. ".ffn_down.weight", b_ffn_act, B, ffn_dim, self.embed_dim, b_proj_out)
        if self.is_gemma4 then
            local pfn_name = prefix .. ".post_ffw_norm.weight"
            if self.gmap.tensors[pfn_name] then
                local b_pfn, o_pfn = self:_tensor_base_and_offset(pfn_name)
                for t_idx = 0, B - 1 do
                    core.gncx_rmsnorm_inplace_heads(b_proj_out + (t_idx * self.embed_dim), b_pfn, o_pfn, 1, 1, self.embed_dim, self.rms_eps)
                end
            end
        end
        for i = 0, (B * self.embed_dim) - 1 do
            b_hidden[i] = b_hidden[i] + b_proj_out[i]
        end

        -- Injeção de Per-Layer Embeddings (Gemma 4 PLE) em lote
        if self.ple_dim > 0 then
            self:_matbatch(prefix .. ".inp_gate.weight", b_hidden, B, self.embed_dim, self.ple_dim, b_ple_gate)
            for t_idx = 0, B - 1 do
                local ple_slice_ptr = b_ple_tok + (t_idx * self.ple_total_dim) + (l * self.ple_dim)
                local gate_slice_ptr = b_ple_gate + (t_idx * self.ple_dim)
                core.gncx_geglu_activation(gate_slice_ptr, ple_slice_ptr, self.ple_dim, gate_slice_ptr)
            end
            self:_matbatch(prefix .. ".proj.weight", b_ple_gate, B, self.ple_dim, self.embed_dim, b_proj_out)
            local pn_name = prefix .. ".post_norm.weight"
            if self.gmap.tensors[pn_name] then
                local b_pn, o_pn = self:_tensor_base_and_offset(pn_name)
                for t_idx = 0, B - 1 do
                    core.gncx_rmsnorm_inplace_heads(b_proj_out + (t_idx * self.embed_dim), b_pn, o_pn, 1, 1, self.embed_dim, self.rms_eps)
                end
            end
            for i = 0, (B * self.embed_dim) - 1 do
                b_hidden[i] = b_hidden[i] + b_proj_out[i]
            end
        end

        local out_scale = self.layer_out_scales[l]
        if out_scale ~= 1.0 then
            for i = 0, (B * self.embed_dim) - 1 do
                b_hidden[i] = b_hidden[i] * out_scale
            end
        end
    end

    local last_hidden = b_hidden + ((B - 1) * self.embed_dim)
    for i = 0, self.embed_dim - 1 do
        self.hidden[i] = last_hidden[i]
    end

    for _, th in ipairs(temp_handles) do
        core.gncx_ram_free(th)
    end

    local base_on, off_on = self:_tensor_base_and_offset("output_norm.weight")
    core.gncx_rmsnorm_and_encode_4phase(
        self.hidden, base_on, off_on, self.embed_dim, self.rms_eps, self.layer_k,
        self.normed, self.x_tokens_embed, self.quant_scale
    )

    local embd_t = self.gmap.tensors["token_embd.weight"]
    local base_lm, off_lm = self:_tensor_base_and_offset("token_embd.weight")
    core.gncx_universal_lm_head_logits(
        base_lm, off_lm, embd_t.ggml_type, self.normed,
        self.embed_dim, self.vocab_size, self.logit_softcap, self.logits_buf
    )

    return self.logits_buf
end

-- Scanner de Projeção Espacial 3D centrada no Equador
function GNCXStreamer:scan_layer_3d_point(l_idx)
    l_idx = l_idx or math.floor(self.n_layers / 2)
    local pt = ffi.new("GNCXLayer3DPoint[1]")
    self.core.gncx_project_layer_3d_point(
        l_idx,
        self.n_layers,
        self.x_tokens_embed,
        self.normed,
        self.embed_dim,
        pt
    )
    return pt[0]
end

GNCXStreamer.prefill_batch = GNCXStreamer.prefill
GNCXEngine = GNCXStreamer

-- Amostragem com penalidade de repetição e temperatura/top-p
function GNCXStreamer:sample(logits, recent_tokens, repeat_penalty, temperature, top_p)
    repeat_penalty = repeat_penalty or 1.15
    temperature = temperature or 0.7
    top_p = top_p or 0.9

    local penalty_set = {}
    if recent_tokens then
        local start_i = math.max(1, #recent_tokens - 64)
        for i = start_i, #recent_tokens do
            penalty_set[recent_tokens[i]] = true
        end
    end

    -- Se temperatura muito baixa, realiza busca Greedy pura
    if temperature <= 0.01 then
        local best_idx = 0
        local best_val = -1e30
        for i = 0, self.vocab_size - 1 do
            local v = logits[i]
            if penalty_set[i] then
                v = (v > 0) and (v / repeat_penalty) or (v * repeat_penalty)
            end
            if v > best_val then
                best_val = v
                best_idx = i
            end
        end
        return best_idx
    end

    -- Seleção Greedy com penalidade
    local best_idx = 0
    local best_val = -1e30
    for i = 0, self.vocab_size - 1 do
        local v = logits[i]
        if penalty_set[i] then
            v = (v > 0) and (v / repeat_penalty) or (v * repeat_penalty)
        end
        if v > best_val then
            best_val = v
            best_idx = i
        end
    end
    return best_idx
end

-- Loop de geração de texto
function GNCXStreamer:generate(prompt, max_new_tokens, callback)
    max_new_tokens = max_new_tokens or 256
    local prompt_tokens = self.tokenizer:encode(prompt)
    if #prompt_tokens == 0 then return "" end

    local logits = self:prefill(prompt_tokens)
    local generated_ids = {}
    local all_tokens = {}
    for _, t in ipairs(prompt_tokens) do table.insert(all_tokens, t) end

    local cur_pos = #prompt_tokens
    local eos_id = self.tokenizer.eos_id

    for step = 1, max_new_tokens do
        if cur_pos >= self.max_seq_len then break end

        local next_token = self:sample(logits, all_tokens)
        if next_token == eos_id then
            break
        end

        table.insert(generated_ids, next_token)
        table.insert(all_tokens, next_token)

        local chunk = self.tokenizer:token_bytes(next_token)
        if callback then
            local should_stop = callback(chunk, next_token, step)
            if should_stop == false then break end
        end

        logits = self:forward_token(next_token, cur_pos, true)
        cur_pos = cur_pos + 1
    end

    return self.tokenizer:decode(generated_ids)
end

function GNCXStreamer:get_geo_telemetry()
    if not self.gmm then return nil end
    local frag = tonumber(self.core.gncx_geo_fragmentation(self.gmm))
    local used = tonumber(self.core.gncx_geo_used(self.gmm))
    local cap  = tonumber(self.core.gncx_geo_capacity(self.gmm))
    return {
        fragmentation = frag,
        used_frames = used,
        capacity_frames = cap,
        allocations = self.geo_allocations or {},
    }
end

function GNCXStreamer:close()
    if self.ram_buffer_handle then
        self.core.gncx_ram_free(self.ram_buffer_handle)
        self.ram_buffer_handle = nil
    end
    for _, sh in ipairs(self.scratch_handles or {}) do
        self.core.gncx_ram_free(sh)
    end
    self.scratch_handles = {}
    for _, kh in ipairs(self.k_cache_handles) do
        self.core.gncx_ram_free(kh)
    end
    for _, vh in ipairs(self.v_cache_handles) do
        self.core.gncx_ram_free(vh)
    end
    self.k_cache_handles = {}
    self.v_cache_handles = {}
    if self.gmm then
        self.core.gncx_geo_mm_destroy(self.gmm)
        self.gmm = nil
    end
    if self.gmap then
        self.gmap:close()
        self.gmap = nil
    end
end

return GNCXStreamer
