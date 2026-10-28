local gguf = require("lua.gncx_gguf")

print("[GNC-X LuaJIT] Testando carregamento de GGUF via gncx_gguf.lua...")

local t0 = os.clock()
local model, err = gguf.open("./models/qwen3-1.7b-q4_k_m.gguf")
if not model then
    error(err)
end
local t1 = os.clock()

print(string.format("[GNC-X LuaJIT] Parse concluído em %.4f segundos!", t1 - t0))
print("  Tamanho do Arquivo: " .. string.format("%.2f MB", model.file_size / (1024 * 1024)))
print("  Arquitetura:        " .. tostring(model:get_metadata("general.architecture")))
print("  Camadas (Layers):   " .. tostring(model:get_metadata("qwen2.block_count") or model:get_metadata("qwen3.block_count") or "N/D"))
print("  Dimensão Embedding: " .. tostring(model:get_metadata("qwen2.embedding_length") or model:get_metadata("qwen3.embedding_length") or "N/D"))
print("  Heads de Atenção:   " .. tostring(model:get_metadata("qwen2.attention.head_count") or "N/D"))
print("  KV Heads (GQA):     " .. tostring(model:get_metadata("qwen2.attention.head_count_kv") or "N/D"))
print("  Data Offset:        " .. tostring(model.data_offset))
print("  Total de Tensores:  " .. tostring(model.tensor_count))

-- Validando tensor de embeddings e primeira camada
local embd = model:get_tensor("token_embd.weight")
if embd then
    print(string.format("  Tensor 'token_embd.weight': Tipo=%d, Bytes=%d, Offset=%s, Ptr=%s",
          embd.ggml_type, embd.byte_size, tostring(embd.abs_file_offset), tostring(embd.ptr)))
end

local blk0_attn_q = model:get_tensor("blk.0.attn_q.weight")
if blk0_attn_q then
    print(string.format("  Tensor 'blk.0.attn_q.weight': Tipo=%d, Shape=[%s], Ptr=%s",
          blk0_attn_q.ggml_type, table.concat(blk0_attn_q.shape, ", "), tostring(blk0_attn_q.ptr)))
end

model:close()
print("[GNC-X LuaJIT] Teste GGUF finalizado com SUCESSO!")
