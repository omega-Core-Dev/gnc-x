local streamer = require("lua.gncx_streamer")

print("=================================================================")
print("  GNC-X LLM — TESTE DO ORQUESTRADOR DE CAMADAS LUAJIT (FASE 4)   ")
print("=================================================================")

local t0 = os.clock()
print("[1] Inicializando motor GNCXStreamer no Perfil 3 (Qwen3 1.7B)...")
local engine = streamer.new("3", 2048, true, 3)
local boot_time = (os.clock() - t0) * 1000
print(string.format("[OK] Motor pronto em %.1f ms!", boot_time))
print(string.format("     Camadas: %d | Dim: %d | Heads: %d (KV: %d) | Vocab: %d",
    engine.n_layers, engine.embed_dim, engine.n_heads, engine.n_kv_heads, engine.vocab_size))

local user_msg = "Diga 'GNC-X em LuaJIT funcionando perfeitamente!' e explique em uma frase o que é uma GPU."
local prompt = engine.tokenizer:format_chat(user_msg, "Você é o assistente GNC-X.", false)

print("\n[2] Prompt formatado:")
print(prompt)

print("[3] Testando geração autorregressiva...")
io.write("\n>>> RESPOSTA: ")
io.flush()

local token_count = 0
local gen_t0 = os.clock()

local full_text = engine:generate(prompt, 64, function(chunk, tid, step)
    token_count = token_count + 1
    io.write(chunk)
    io.flush()
    if step >= 64 then return false end
    return true
end)

local total_time = os.clock() - gen_t0
local tps = token_count / (total_time > 0 and total_time or 0.001)

print(string.format("\n\n[OK] Concluído! %d tokens gerados em %.2f s (%.2f tokens/s)", token_count, total_time, tps))

local geo = engine:get_geo_telemetry()
if geo then
    print(string.format("[GMM Telemetria] Frames: %d/%d | Fragmentação: %.4f%% | Alocações registradas: %d",
        geo.used_frames, geo.capacity_frames, geo.fragmentation * 100.0, #geo.allocations))
end

engine:close()
print("[OK] Memória liberada e handles fechados com sucesso.")
