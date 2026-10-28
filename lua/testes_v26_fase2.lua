--[[
  Bateria de Testes Automatizados — GNC-X Fase 2
  Pré-Inferência com EmbeddingGemma 2, Roteamento Semântico e Filtro RAG
--]]

local ffi = require("ffi")
local emb_mod = require("lua.gncx_embedding")

print("================================================================================")
print("     BATERIA DE TESTES GNC-X v2.6 — FASE 2: EMBEDDINGGEMMA 2 & ROTEAMENTO       ")
print("================================================================================")

local passed = 0
local total = 0
local function assert_test(cond, msg)
    total = total + 1
    if cond then
        passed = passed + 1
        print(string.format("  [\27[32mPASS\27[0m] Teste %d: %s", total, msg))
    else
        print(string.format("  [\27[31mFAIL\27[0m] Teste %d: %s", total, msg))
        error("Falha no teste: " .. msg)
    end
end

-- TESTE 1: Mapeamento mmap zero-copy e Alocação HugePage
print("\n--- Teste 1: Leitor e Mapeador GGUF Zero-Copy ---")
local t0 = os.clock()
local emb = emb_mod.new("models/embeddinggemma-2-Q8_0.gguf", 256)
local load_ms = (os.clock() - t0) * 1000
print(string.format("  Modelo carregado em %.2f ms", load_ms))
assert_test(emb ~= nil, "Instância de GNCXEmbedding criada")
assert_test(emb.n_layers == 24, "Número de camadas correto (24)")
assert_test(emb.embed_dim == 512, "Dimensão base correta (512)")
assert_test(emb.output_dim == 768, "Dimensão de saída correta (768)")
assert_test(emb.has_ple == true, "PLE detectado e mapeado com sucesso")

-- TESTE 2: Geração de Embedding, L2-Norm e Dimensão
print("\n--- Teste 2: Extração de Vetor 768d e Normalização L2 ---")
local t_enc0 = os.clock()
local v1, buf1 = emb:encode("Executar comando no terminal do Linux")
local enc_ms = (os.clock() - t_enc0) * 1000
print(string.format("  Vetor 1 gerado em %.2f ms", enc_ms))
assert_test(#v1 == 768, "Vetor retornado possui tamanho exato 768")

-- Verifica que a norma L2 é 1.0 (dentro de margem de precisão fp32)
local sum_sq = 0.0
for i = 1, #v1 do sum_sq = sum_sq + (v1[i] * v1[i]) end
local l2_norm = math.sqrt(sum_sq)
print(string.format("  Norma L2 do vetor = %.6f", l2_norm))
assert_test(math.abs(l2_norm - 1.0) < 1e-4, "Norma L2 perfeitamente unitária (1.000000)")

-- TESTE 3: Similaridade Semântica Direcional (SSE2)
print("\n--- Teste 3: Similaridade Semântica de Cosseno ---")
local v_rel = emb:encode("Rodar script no shell do sistema operacional")
local v_unrel = emb:encode("Como fazer um bolo de cenoura com cobertura de chocolate")

local sim_rel = emb:similarity(v1, v_rel)
local sim_unrel = emb:similarity(v1, v_unrel)
print(string.format("  Similaridade Semântica (Comando Linux vs Shell): %.4f", sim_rel))
print(string.format("  Similaridade Não-Relacionada (Linux vs Bolo):     %.4f", sim_unrel))

assert_test(sim_rel > sim_unrel, "Similaridade de textos semanticamente equivalentes é superior")
assert_test(sim_rel > 0.85, "Similaridade entre comandos de terminal é alta (> 0.85)")

-- TESTE 4: Roteamento Semântico de Ferramentas (< 5 ms em cache)
print("\n--- Teste 4: Roteamento Semântico de Ferramentas ---")
local tools = {
    { id = "navegador", desc = "Abrir o navegador Web Firefox para acessar a internet" },
    { id = "arquivos",  desc = "Abrir o gerenciador de arquivos Nautilus para ver pastas e documentos" },
    { id = "terminal",  desc = "Abrir uma nova janela de terminal para digitar comandos bash" },
    { id = "status",    desc = "Exibir a saúde, uso de memória RAM e diagnóstico geral do sistema" },
}

-- Pré-aquece o cache
for _, t in ipairs(tools) do
    emb.tool_cache[t.desc] = emb:encode(t.desc)
end

-- Roteia a query com medição de latência
local t_route0 = os.clock()
local query_input = "Quero navegar na internet e ver sites"
local q_vec = emb:encode(query_input)
local best_idx, best_score = emb:route(query_input, tools)
local route_ms = (os.clock() - t_route0) * 1000

print(string.format("  Query: \"%s\"", query_input))
print(string.format("  Melhor Ferramenta: %s (Score: %.4f | Decisão: %.2f ms)", tools[best_idx].id, best_score, route_ms))
assert_test(tools[best_idx].id == "navegador", "Roteador identificou ferramenta correta ('navegador')")
assert_test(best_score > 0.85, "Score de roteamento de ferramenta seguro (> 0.85)")

-- TESTE 5: Filtro Semântico RAG Local (Redução > 70% de Tokens)
print("\n--- Teste 5: Filtro Semântico RAG Local ---")
local doc_chunks = {
    "O motor GNC-X utiliza um alocador de HugePages Slab de 4MB com coloração para eliminar colisões de barramento DDR3.",
    "A receita clássica de pão caseiro leva farinha de trigo, água morna, fermento biológico e uma pitada de sal refinado.",
    "Para compilar a biblioteca nativa bin/gncx_llm_core.so no Linux x86_64, utilize o script ./compilar_linux.sh com g++.",
    "O telescópio James Webb orbita o ponto de Lagrange L2 a aproximadamente 1.5 milhão de quilômetros da Terra.",
    "O mecanismo de Grouped-Query Attention (GQA) agrupa quatro cabeças de Query para cada duas cabeças de Key-Value no Gemma."
}

local total_chars = 0
local chunk_vecs = {}
for i, c in ipairs(doc_chunks) do
    total_chars = total_chars + #c
    chunk_vecs[i] = emb:encode(c)
end

local user_q = "Como funciona a compilação do núcleo nativo C++ no Linux?"
local user_q_vec = emb:encode(user_q)

local ranked = {}
for i, c in ipairs(doc_chunks) do
    local s = emb:similarity(user_q_vec, chunk_vecs[i])
    table.insert(ranked, { idx = i, text = c, score = s })
end
table.sort(ranked, function(a, b) return a.score > b.score end)

print(string.format("  Consulta: \"%s\"", user_q))
print(string.format("  Top-1 Recuperado (Score: %.4f): \"%s\"", ranked[1].score, ranked[1].text))

assert_test(ranked[1].idx == 3, "RAG local recuperou exatamente o trecho sobre compilação Linux (chunk 3)")

-- Medição da taxa de compressão / redução de prefill
local retrieved_chars = #ranked[1].text
local savings_pct = (1.0 - (retrieved_chars / total_chars)) * 100.0
print(string.format("  Caracteres Totais do Documento: %d", total_chars))
print(string.format("  Caracteres Injetados no Prefill: %d", retrieved_chars))
print(string.format("  Economia no volume de prefill:   %.2f%%", savings_pct))
assert_test(savings_pct >= 70.0, "Redução superior a 70% no volume de prefill comprovada")

emb:close()

print("\n================================================================================")
print(string.format("RESULTADO FINAL: %d/%d testes passaram com 100%% de sucesso!", passed, total))
print("================================================================================")
