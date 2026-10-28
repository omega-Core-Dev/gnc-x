--[[
  GNC-X LLM — Módulo de Pré-Inferência e Embeddings com EmbeddingGemma 2
  Fase 2: Mapeamento mmap zero-copy de embeddinggemma-2-Q8_0.gguf (296 MB),
  extração de vetores semânticos (Mean Pooling + Dense Proj 768d + L2 Norm)
  e Similaridade de Cosseno ultrarrápida em CPU via SIMD SSE2.
--]]

local ffi = require("ffi")
local gguf = require("lua.gncx_gguf")
local spiece = require("lua.gncx_spiece")
local streamer = require("lua.gncx_streamer")

local GNCXEmbedding = {}
GNCXEmbedding.__index = GNCXEmbedding

-- Garante carregamento da biblioteca nativa compartilhada
local CORE_LIB_PATH = "./bin/gncx_llm_core.so"
local ok, core = pcall(ffi.load, CORE_LIB_PATH)
if not ok then
    error("[GNCXEmbedding] Falha ao carregar biblioteca nativa: " .. tostring(core))
end

function GNCXEmbedding.new(model_path, max_seq_len)
    model_path = model_path or "models/embeddinggemma-2-Q8_0.gguf"
    max_seq_len = max_seq_len or 256

    local t0 = os.clock()
    local gmap, err = gguf.open(model_path)
    if not gmap then
        error("[GNCXEmbedding] Falha ao abrir modelo GGUF em: " .. model_path .. " (" .. tostring(err) .. ")")
    end
    local load_time_ms = (os.clock() - t0) * 1000

    local self = setmetatable({}, GNCXEmbedding)
    self.core = core
    self.gmap = gmap
    self.model_path = model_path
    self.mmap_base_ptr = gmap.base
    self.max_seq_len = max_seq_len
    self.load_time_ms = load_time_ms

    -- 1. Metadados Arquiteturais
    local arch = gmap:get_metadata("general.architecture", "gemma-embedding2")
    self.arch = arch
    self.n_layers = tonumber(gmap:get_metadata(arch .. ".block_count", 24))
    self.embed_dim = tonumber(gmap:get_metadata(arch .. ".embedding_length", 512))
    self.ffn_dim = tonumber(gmap:get_metadata(arch .. ".feed_forward_length", 2048))
    self.head_dim = 256
    self.n_kv_heads = 2
    self.kv_dim = self.n_kv_heads * self.head_dim -- 512
    self.output_dim = tonumber(gmap:get_metadata(arch .. ".embedding_length_out", 768))
    self.rms_eps = tonumber(gmap:get_metadata(arch .. ".attention.layer_norm_rms_epsilon", 1e-6))
    self.rope_freq_base_global = tonumber(gmap:get_metadata(arch .. ".rope.freq_base", 1000000.0))
    self.rope_freq_base_swa    = tonumber(gmap:get_metadata(arch .. ".rope.freq_base_swa", 10000.0))
    self.has_ple = gmap.tensors["per_layer_model_proj.weight"] ~= nil
    self.ple_total_dim = self.n_layers * self.embed_dim -- 12288

    -- 2. Tokenizador SentencePiece
    local tokens = gmap:get_metadata("tokenizer.ggml.tokens")
    local scores = gmap:get_metadata("tokenizer.ggml.scores")
    local ttypes = gmap:get_metadata("tokenizer.ggml.token_type")
    local bos_id = tonumber(gmap:get_metadata("tokenizer.ggml.bos_token_id", 2))
    local eos_id = tonumber(gmap:get_metadata("tokenizer.ggml.eos_token_id", 1))
    local unk_id = tonumber(gmap:get_metadata("tokenizer.ggml.unknown_token_id", 3))

    self.tokenizer = spiece.new(tokens, scores, ttypes, bos_id, eos_id, unk_id)

    -- 3. Scratch Buffers Alocados via Slab HugePage Colorido DDR3
    self.scratch_handles = {}
    local function alloc_buf(n_floats, slot_d)
        local h = self.core.gncx_ram_alloc_colored(math.max(n_floats * 4, 4096), 1, slot_d, 3)
        table.insert(self.scratch_handles, h)
        return ffi.cast("float*", self.core.gncx_ram_get_ptr(h))
    end

    local B = max_seq_len
    local max_q_dim = 2048
    local max_ffn_dim = self.ffn_dim

    self.b_hidden   = alloc_buf(B * self.embed_dim, 0)
    self.b_normed   = alloc_buf(B * self.embed_dim, 1)
    self.b_q        = alloc_buf(B * max_q_dim,      2)
    self.b_k        = alloc_buf(B * self.kv_dim,    3)
    self.b_v        = alloc_buf(B * self.kv_dim,    4)
    self.b_attn_out = alloc_buf(B * max_q_dim,      5)
    self.b_proj_out = alloc_buf(B * math.max(self.embed_dim, max_ffn_dim), 6)
    self.b_gate     = alloc_buf(B * max_ffn_dim,   7)
    self.b_up       = alloc_buf(B * max_ffn_dim,   8)
    self.b_ffn_act  = alloc_buf(B * max_ffn_dim,   9)

    if self.has_ple then
        self.b_ple_proj = alloc_buf(B * self.ple_total_dim, 10)
        self.b_ple_gate = alloc_buf(B * self.embed_dim,     11)
    end

    -- Buffers finais de pooling e saída
    self.pooled_buf  = alloc_buf(self.embed_dim,    12)
    self.out_emb_buf = alloc_buf(self.output_dim,   13)

    -- Cache de embeddings de ferramentas para roteamento instantâneo (< 5 ms)
    self.tool_cache = {}

    return self
end

function GNCXEmbedding:_tensor_base_and_offset(tensor_name)
    local t = self.gmap.tensors[tensor_name]
    if t then
        return self.mmap_base_ptr, t.abs_file_offset
    end
    return nil, 0
end

function GNCXEmbedding:_matbatch(tensor_name, in_batch, batch_size, in_dim, out_dim, out_batch)
    local t = self.gmap.tensors[tensor_name]
    if not t then error("[GNCXEmbedding] Tensor ausente: " .. tensor_name) end
    local base, off = self:_tensor_base_and_offset(tensor_name)
    self.core.gncx_xfold_universal_matbatch(
        base, off, t.ggml_type, in_batch, batch_size, in_dim, out_dim, out_batch, nil
    )
end

function GNCXEmbedding:_matvec(tensor_name, in_vec, in_dim, out_dim, out_vec)
    local t = self.gmap.tensors[tensor_name]
    if not t then error("[GNCXEmbedding] Tensor ausente: " .. tensor_name) end
    local base, off = self:_tensor_base_and_offset(tensor_name)
    self.core.gncx_xfold_universal_matvec(
        base, off, t.ggml_type, in_vec, nil, in_dim, out_dim, out_vec, nil
    )
end

-- Gera vetor de embedding de dimensão 768 para o texto informado
function GNCXEmbedding:encode(text)
    if type(text) ~= "string" or text == "" then
        text = " "
    end

    local tokens = self.tokenizer:encode(text)
    local token_ids = {}

    -- O tokenizador SentencePiece do Gemma insere BOS (2) no início.
    -- Se porventura não tiver BOS, inserimos:
    if #tokens == 0 or tokens[1] ~= (self.tokenizer.bos_id or 2) then
        table.insert(token_ids, self.tokenizer.bos_id or 2)
    end
    for _, tid in ipairs(tokens) do
        table.insert(token_ids, tid)
        if #token_ids >= self.max_seq_len - 1 then break end
    end
    -- Insere EOS no final se não estiver presente
    if token_ids[#token_ids] ~= (self.tokenizer.eos_id or 1) then
        table.insert(token_ids, self.tokenizer.eos_id or 1)
    end

    local B = #token_ids
    local core = self.core

    -- 1. Extração dos embeddings de entrada com escala sqrt(embed_dim)
    local embd_t = self.gmap.tensors["token_embd.weight"]
    local base_emb, off_emb = self:_tensor_base_and_offset("token_embd.weight")
    local scale_factor = math.sqrt(self.embed_dim)

    for t_idx = 0, B - 1 do
        core.gncx_fetch_token_embedding_q8_0(
            base_emb, off_emb, token_ids[t_idx + 1], self.embed_dim,
            scale_factor, self.b_hidden + (t_idx * self.embed_dim)
        )
    end

    -- 1.1 Pré-cálculo de PLE se disponível
    if self.has_ple then
        self:_matbatch("per_layer_model_proj.weight", self.b_hidden, B, self.embed_dim, self.ple_total_dim, self.b_ple_proj)
        local proj_scale = 1.0 / math.sqrt(self.embed_dim)
        for i = 0, (B * self.ple_total_dim) - 1 do
            self.b_ple_proj[i] = self.b_ple_proj[i] * proj_scale
        end
        local base_norm, off_norm = self:_tensor_base_and_offset("per_layer_proj_norm.weight")
        for t_idx = 0, B - 1 do
            core.gncx_rmsnorm_inplace_heads(
                self.b_ple_proj + (t_idx * self.ple_total_dim),
                base_norm,
                off_norm,
                1,
                self.n_layers,
                self.embed_dim,
                self.rms_eps
            )
        end
    end

    -- 2. Passagem pelas 24 Camadas de Transformador
    local attn_scale = 1.0 / math.sqrt(self.head_dim)

    for l = 0, self.n_layers - 1 do
        local prefix = string.format("blk.%d", l)
        local is_global = (l == 5 or l == 11 or l == 17 or l == 23)
        local n_heads = is_global and 8 or 4
        local q_dim = n_heads * self.head_dim
        local freq_base = is_global and self.rope_freq_base_global or self.rope_freq_base_swa

        -- 2.1. Atenção Normalização (Fast-Path fp32)
        local base_an, off_an = self:_tensor_base_and_offset(prefix .. ".attn_norm.weight")
        for t_idx = 0, B - 1 do
            core.gncx_rmsnorm_and_encode_4phase(
                self.b_hidden + (t_idx * self.embed_dim),
                base_an, off_an, self.embed_dim, self.rms_eps, 2,
                self.b_normed + (t_idx * self.embed_dim),
                nil, nil
            )
        end

        -- 2.2. Projeções Lineares Q, K, V em Lote
        self:_matbatch(prefix .. ".attn_q.weight", self.b_normed, B, self.embed_dim, q_dim, self.b_q)
        self:_matbatch(prefix .. ".attn_k.weight", self.b_normed, B, self.embed_dim, self.kv_dim, self.b_k)
        self:_matbatch(prefix .. ".attn_v.weight", self.b_normed, B, self.embed_dim, self.kv_dim, self.b_v)

        -- 2.3. Normalização de Cabeças Q/K
        local qn_name = prefix .. ".attn_q_norm.weight"
        local kn_name = prefix .. ".attn_k_norm.weight"
        if self.gmap.tensors[qn_name] then
            local b_qn, o_qn = self:_tensor_base_and_offset(qn_name)
            for t_idx = 0, B - 1 do
                core.gncx_rmsnorm_inplace_heads(self.b_q + (t_idx * q_dim), b_qn, o_qn, 1, n_heads, self.head_dim, self.rms_eps)
            end
        end
        if self.gmap.tensors[kn_name] then
            local b_kn, o_kn = self:_tensor_base_and_offset(kn_name)
            for t_idx = 0, B - 1 do
                core.gncx_rmsnorm_inplace_heads(self.b_k + (t_idx * self.kv_dim), b_kn, o_kn, 1, self.n_kv_heads, self.head_dim, self.rms_eps)
            end
        end

        -- 2.4. RoPE nas cabeças
        for t_idx = 0, B - 1 do
            core.gncx_apply_rope_xor11_ext(
                self.b_q + (t_idx * q_dim),
                self.b_k + (t_idx * self.kv_dim),
                nil, nil, 0, 0,
                n_heads, self.n_kv_heads, self.head_dim, t_idx, freq_base
            )
        end

        -- 2.5. Atenção Bidirecional GQA SSE2 em Bloco Completo
        core.gncx_bidirectional_attention_gqa(
            self.b_q, self.b_k, self.b_v, B, n_heads, self.n_kv_heads, self.head_dim, attn_scale, self.b_attn_out
        )

        -- 2.6. Projeção Attn Output + Post Attn Norm + Residual
        self:_matbatch(prefix .. ".attn_output.weight", self.b_attn_out, B, q_dim, self.embed_dim, self.b_proj_out)
        local pan_name = prefix .. ".post_attention_norm.weight"
        if self.gmap.tensors[pan_name] then
            local b_pan, o_pan = self:_tensor_base_and_offset(pan_name)
            for t_idx = 0, B - 1 do
                core.gncx_rmsnorm_inplace_heads(self.b_proj_out + (t_idx * self.embed_dim), b_pan, o_pan, 1, 1, self.embed_dim, self.rms_eps)
            end
        end
        for i = 0, (B * self.embed_dim) - 1 do
            self.b_hidden[i] = self.b_hidden[i] + self.b_proj_out[i]
        end

        -- 2.7. FFN Norm
        local base_fn, off_fn = self:_tensor_base_and_offset(prefix .. ".ffn_norm.weight")
        for t_idx = 0, B - 1 do
            core.gncx_rmsnorm_and_encode_4phase(
                self.b_hidden + (t_idx * self.embed_dim),
                base_fn, off_fn, self.embed_dim, self.rms_eps, 2,
                self.b_normed + (t_idx * self.embed_dim),
                nil, nil
            )
        end

        -- 2.8. FFN Gate / Up / GeGLU / Down
        self:_matbatch(prefix .. ".ffn_gate.weight", self.b_normed, B, self.embed_dim, self.ffn_dim, self.b_gate)
        self:_matbatch(prefix .. ".ffn_up.weight",   self.b_normed, B, self.embed_dim, self.ffn_dim, self.b_up)
        core.gncx_geglu_activation(self.b_gate, self.b_up, B * self.ffn_dim, self.b_ffn_act)
        self:_matbatch(prefix .. ".ffn_down.weight", self.b_ffn_act, B, self.ffn_dim, self.embed_dim, self.b_proj_out)

        local pfn_name = prefix .. ".post_ffw_norm.weight"
        if self.gmap.tensors[pfn_name] then
            local b_pfn, o_pfn = self:_tensor_base_and_offset(pfn_name)
            for t_idx = 0, B - 1 do
                core.gncx_rmsnorm_inplace_heads(self.b_proj_out + (t_idx * self.embed_dim), b_pfn, o_pfn, 1, 1, self.embed_dim, self.rms_eps)
            end
        end
        for i = 0, (B * self.embed_dim) - 1 do
            self.b_hidden[i] = self.b_hidden[i] + self.b_proj_out[i]
        end

        -- 2.9. PLE Injeção por Camada
        if self.has_ple then
            self:_matbatch(prefix .. ".inp_gate.weight", self.b_hidden, B, self.embed_dim, self.embed_dim, self.b_ple_gate)
            for t_idx = 0, B - 1 do
                local ple_slice_ptr = self.b_ple_proj + (t_idx * self.ple_total_dim) + (l * self.embed_dim)
                local gate_slice_ptr = self.b_ple_gate + (t_idx * self.embed_dim)
                core.gncx_geglu_activation(gate_slice_ptr, ple_slice_ptr, self.embed_dim, gate_slice_ptr)
            end
            self:_matbatch(prefix .. ".proj.weight", self.b_ple_gate, B, self.embed_dim, self.embed_dim, self.b_proj_out)
            local pn_name = prefix .. ".post_norm.weight"
            if self.gmap.tensors[pn_name] then
                local b_pn, o_pn = self:_tensor_base_and_offset(pn_name)
                for t_idx = 0, B - 1 do
                    core.gncx_rmsnorm_inplace_heads(self.b_proj_out + (t_idx * self.embed_dim), b_pn, o_pn, 1, 1, self.embed_dim, self.rms_eps)
                end
            end
            for i = 0, (B * self.embed_dim) - 1 do
                self.b_hidden[i] = self.b_hidden[i] + self.b_proj_out[i]
            end
        end

        -- 2.10. Layer output scale
        local scale_name = prefix .. ".layer_output_scale.weight"
        if self.gmap.tensors[scale_name] then
            local base_sc, off_sc = self:_tensor_base_and_offset(scale_name)
            local sc_val = ffi.cast("const float*", base_sc + off_sc)[0]
            if sc_val ~= 1.0 then
                for i = 0, (B * self.embed_dim) - 1 do
                    self.b_hidden[i] = self.b_hidden[i] * sc_val
                end
            end
        end
    end

    -- 3. Output Norm
    local base_on, off_on = self:_tensor_base_and_offset("output_norm.weight")
    for t_idx = 0, B - 1 do
        core.gncx_rmsnorm_and_encode_4phase(
            self.b_hidden + (t_idx * self.embed_dim),
            base_on, off_on, self.embed_dim, self.rms_eps, 2,
            self.b_normed + (t_idx * self.embed_dim),
            nil, nil
        )
    end

    -- 4. Mean Pooling em todos os tokens
    core.gncx_mean_pooling(self.b_normed, B, self.embed_dim, self.pooled_buf)

    -- 5. Projeção Densa Final para output_dim (768)
    if self.gmap.tensors["output.weight"] then
        self:_matvec("output.weight", self.pooled_buf, self.embed_dim, self.output_dim, self.out_emb_buf)
    else
        for i = 0, self.embed_dim - 1 do
            self.out_emb_buf[i] = self.pooled_buf[i]
        end
    end

    -- 6. Normalização L2 do Vetor Gerado
    core.gncx_vector_l2_normalize(self.out_emb_buf, self.output_dim)

    -- Retorna uma cópia compacta em tabela Lua ou buffer reutilizável
    local emb_copy = {}
    for i = 0, self.output_dim - 1 do
        emb_copy[i + 1] = self.out_emb_buf[i]
    end
    return emb_copy, self.out_emb_buf
end

-- Calcula a Similaridade de Cosseno entre dois vetores de embedding
function GNCXEmbedding:similarity(vec_a, vec_b)
    local dim = self.output_dim
    local ptr_a = ffi.new("float[?]", dim)
    local ptr_b = ffi.new("float[?]", dim)
    for i = 1, dim do
        ptr_a[i - 1] = vec_a[i] or 0.0
        ptr_b[i - 1] = vec_b[i] or 0.0
    end
    return self.core.gncx_cosine_similarity(ptr_a, ptr_b, dim)
end

-- Atalho: Calcula a Similaridade Semântica direta entre dois textos
function GNCXEmbedding:similarity_text(text_a, text_b)
    local vec_a = self:encode(text_a)
    local vec_b = self:encode(text_b)
    return self:similarity(vec_a, vec_b)
end

-- Roteamento Semântico de Ferramentas / Ações
-- Compara o texto de entrada com a lista de descrições e retorna o melhor índice
function GNCXEmbedding:route(text, candidates)
    local query_vec = self:encode(text)
    local best_idx = nil
    local best_score = -1.0

    for idx, cand in ipairs(candidates) do
        local cand_desc = type(cand) == "table" and (cand.desc or cand[1]) or tostring(cand)
        if not self.tool_cache[cand_desc] then
            self.tool_cache[cand_desc] = self:encode(cand_desc)
        end
        local cand_vec = self.tool_cache[cand_desc]
        local score = self:similarity(query_vec, cand_vec)
        if score > best_score then
            best_score = score
            best_idx = idx
        end
    end

    return best_idx, best_score
end

function GNCXEmbedding:close()
    for _, h in ipairs(self.scratch_handles) do
        self.core.gncx_ram_free(h)
    end
    self.scratch_handles = {}
    if self.gmap then
        self.gmap:close()
        self.gmap = nil
    end
end

return GNCXEmbedding
