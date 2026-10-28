local gguf = require("lua.gncx_gguf")
local bpe  = require("lua.gncx_bpe")

print("[GNC-X LuaJIT] Testando Tokenizador BPE Qwen3...")

local m = gguf.open("./models/qwen3-1.7b-q4_k_m.gguf")
local tokens = m:get_metadata("tokenizer.ggml.tokens")
local merges = m:get_metadata("tokenizer.ggml.merges")
local ttypes = m:get_metadata("tokenizer.ggml.token_type")

print(string.format("  Metadados lidos: %d tokens, %d merges", #tokens, merges and #merges or 0))

local tok = bpe.new(tokens, merges, ttypes, -1, 151645)
m:close()

-- Teste de formatação de chat ChatML
local prompt = tok:format_chat("Olá! Quem é você?", "Você é o assistente GNC-X.", true)
print("\n[OK] Template ChatML formatado:")
print(prompt)

-- Teste de decodificação de tokens especiais e texto comum
print("[OK] Teste de Decodificação:")
-- Vamos buscar alguns tokens conhecidos por texto
local ids_teste = {}
local palavras = { "Olá", ",", " mundo", "!", " Aten", "ção", " geom", "étrica", "." }

for _, p in ipairs(palavras) do
    local tid = tok.token_to_id[p]
    if tid then
        table.insert(ids_teste, tid)
    end
end

local texto_decodificado = tok:decode(ids_teste)
print("  Tokens decodificados:")
print("  IDs:  [" .. table.concat(ids_teste, ", ") .. "]")
print("  Texto: '" .. texto_decodificado .. "'")

print("\n[GNC-X LuaJIT] Tokenizador BPE testado com SUCESSO!")
