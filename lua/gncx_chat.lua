--[[
  GNC-X LLM — Interactive Multi-Profile Chat CLI em LuaJIT
  Fase 5: Console Interativo com Typewriter Nativo, Orçamento Desacoplado
  e Controle Inteligente de Raciocínio (<think>)
--]]

local ffi = require("ffi")
local streamer = require("lua.gncx_streamer")

-- Declara usleep e gettimeofday nativos da libc para máquina de escrever e relógio real (wall-clock)
ffi.cdef[[
    int usleep(unsigned int usec);
    typedef struct {
        long tv_sec;
        long tv_usec;
    } timeval_t;
    int gettimeofday(timeval_t* tv, void* tz);
]]

local tv_buf = ffi.new("timeval_t")
local function wall_clock_sec()
    ffi.C.gettimeofday(tv_buf, nil)
    return tonumber(tv_buf.tv_sec) + tonumber(tv_buf.tv_usec) * 1e-6
end

local function sleep_ms(ms)
    if ms > 0 then
        ffi.C.usleep(math.floor(ms * 1000))
    end
end

-- Telemetria de memória RAM do processo via /proc/self/status
local function get_process_memory_mb()
    local f = io.open("/proc/self/status", "r")
    if not f then return 0 end
    local rss_kb = 0
    for line in f:lines() do
        local m = line:match("^VmRSS:%s+(%d+)%s+kB")
        if m then
            rss_kb = tonumber(m)
            break
        end
    end
    f:close()
    return rss_kb / 1024.0
end

local function get_timestamp_str()
    return os.date("%Y%m%d_%H%M%S")
end

local function get_datetime_display()
    return os.date("%d/%m/%Y %H:%M:%S")
end

-- Cores e Estilos ANSI
local ANSI = {
    RESET     = "\27[0m",
    BOLD      = "\27[1m",
    DIM       = "\27[2m",
    ITALIC    = "\27[3m",
    UNDERLINE = "\27[4m",
    GRAY      = "\27[90m",
    CYAN      = "\27[36m",
    GREEN     = "\27[32m",
    YELLOW    = "\27[33m",
    BLUE      = "\27[34m",
    MAGENTA   = "\27[35m",
    RED       = "\27[31m",
}

-- Salva conversa em Markdown
local function save_markdown(filename, last_query, thought_text, answer_text, profile_id)
    if not filename or filename == "" then
        filename = string.format("gncx_resposta_%s.md", get_timestamp_str())
    else
        filename = filename:gsub("^%s*(.-)%s*$", "%1")
        if not filename:match("%.%w+$") then
            filename = filename .. ".md"
        end
    end

    local f, err = io.open(filename, "w")
    if not f then
        return nil, "Não foi possível criar o arquivo: " .. tostring(err)
    end

    f:write(string.format("# Resposta GNC-X — %s\n", profile_id:upper()))
    f:write(string.format("**Data/Hora:** %s  \n", get_datetime_display()))
    f:write(string.format("**Solicitação:** %s\n\n", last_query))
    f:write("---\n\n")

    if thought_text and #thought_text > 0 then
        f:write("### Raciocínio Interno (<thought>)\n```text\n")
        f:write(thought_text)
        f:write("\n```\n\n---\n\n")
    end

    f:write("### Resposta Final\n")
    f:write(answer_text)
    f:write("\n")
    f:close()

    return filename
end

-- Prompts de sistema otimizados por perfil para evitar respostas truncadas
local DEFAULT_SYSTEM_PROMPTS = {
    ["1"] = "Você é o assistente GNC-X. Responda em português do Brasil de forma completa, detalhada e explicativa, desenvolvendo suas explicações com riqueza de informações e exemplos.",
    ["gemma2b"] = "Você é o assistente GNC-X. Responda em português do Brasil de forma completa, detalhada e explicativa, desenvolvendo suas explicações com riqueza de informações e exemplos.",
    ["2"] = "Você é o assistente GNC-X. Responda em português do Brasil de forma completa, estruturada e aprofundada.",
    ["gemma4"] = "Você é o assistente GNC-X. Responda em português do Brasil de forma completa, estruturada e aprofundada.",
    ["3"] = "Você é o assistente GNC-X. Responda em português do Brasil de forma precisa, completa e bem estruturada.",
    ["qwen3"] = "Você é o assistente GNC-X. Responda em português do Brasil de forma precisa, completa e bem estruturada.",
}

-- Módulo RAG Semântico Local via EmbeddingGemma 2 (Fase 2)
local GNCXSemanticRAG = {
    chunks = {},
    total_doc_chars = 0,
    embedding_engine = nil,
}

function GNCXSemanticRAG:get_engine()
    if not self.embedding_engine then
        local ok, mod = pcall(require, "lua.gncx_embedding")
        if ok then
            local ok_inst, inst = pcall(mod.new, "models/embeddinggemma-2-Q8_0.gguf", 256)
            if ok_inst then
                self.embedding_engine = inst
            end
        end
    end
    return self.embedding_engine
end

function GNCXSemanticRAG:index_text(text, source_name, chunk_size, chunk_overlap)
    chunk_size = chunk_size or 350
    chunk_overlap = chunk_overlap or 50
    local emb = self:get_engine()
    if not emb then
        return false, "Modelo EmbeddingGemma não disponível"
    end

    local len = #text
    self.total_doc_chars = self.total_doc_chars + len
    local start_pos = 1
    local added = 0

    while start_pos <= len do
        local end_pos = math.min(start_pos + chunk_size, len)
        if end_pos < len then
            local newline_pos = text:find("\n", end_pos - 40)
            if newline_pos and newline_pos <= end_pos + 40 and newline_pos > start_pos then
                end_pos = newline_pos
            end
        end
        local chunk = text:sub(start_pos, end_pos):gsub("^%s*(.-)%s*$", "%1")
        if #chunk > 30 then
            local vec = emb:encode(chunk)
            table.insert(self.chunks, {
                text = chunk,
                vec = vec,
                source = source_name or "documento"
            })
            added = added + 1
        end
        start_pos = end_pos + 1 - chunk_overlap
        if start_pos >= len then break end
    end
    return true, added
end

function GNCXSemanticRAG:retrieve_top_k(query, k, min_score)
    k = k or 2
    min_score = min_score or 0.70
    local emb = self:get_engine()
    if not emb or #self.chunks == 0 then return {} end

    local q_vec = emb:encode(query)
    local scored = {}
    for i, c in ipairs(self.chunks) do
        local sim = emb:similarity(q_vec, c.vec)
        if sim >= min_score then
            table.insert(scored, { chunk = c, score = sim })
        end
    end
    table.sort(scored, function(a, b) return a.score > b.score end)

    local results = {}
    for i = 1, math.min(k, #scored) do
        table.insert(results, scored[i])
    end
    return results
end

function GNCXSemanticRAG:clear()
    self.chunks = {}
    self.total_doc_chars = 0
end

-- Banner e Apresentação
local function print_banner(profile_label, n_layers, vocab_size, max_seq_len, max_tokens, enable_thinking)
    print(ANSI.CYAN .. "================================================================================" .. ANSI.RESET)
    print(ANSI.BOLD .. ANSI.GREEN .. "       MOTOR GEOMÉTRICO GNC-X LLM — RUNTIME NATIVO LUAJIT ZERO-OVERHEAD       " .. ANSI.RESET)
    print(ANSI.CYAN .. "================================================================================" .. ANSI.RESET)
    print(string.format("  %sModelo Ativo:%s      %s", ANSI.BOLD, ANSI.RESET, profile_label))
    print(string.format("  %sTopologia:%s         %d Camadas | Vocabulário: %s tokens",
        ANSI.BOLD, ANSI.RESET, n_layers, string.format("%d", vocab_size)))
    print(string.format("  %sKV-Cache:%s          %d posições alinhadas em RAM",
        ANSI.BOLD, ANSI.RESET, max_seq_len))
    print(string.format("  %sOrçamento Resposta:%s %d tokens livres (não consumidos pelo pensamento)",
        ANSI.BOLD, ANSI.RESET, max_tokens))
    local think_status = enable_thinking and (ANSI.GREEN .. "ATIVADO" .. ANSI.RESET) or (ANSI.YELLOW .. "DESATIVADO (Respostas Instantâneas)" .. ANSI.RESET)
    print(string.format("  %sModo Raciocínio:%s   %s", ANSI.BOLD, ANSI.RESET, think_status))
    print(string.format("  %sMemória Base:%s      %.1f MB (Processo LuaJIT FFI)",
        ANSI.BOLD, ANSI.RESET, get_process_memory_mb()))
    print(ANSI.CYAN .. "--------------------------------------------------------------------------------" .. ANSI.RESET)
    print("  Comandos: " .. ANSI.YELLOW .. "/raciocinio [on/off]" .. ANSI.RESET .. " | "
          .. ANSI.YELLOW .. "/tokens [n]" .. ANSI.RESET .. " | "
          .. ANSI.YELLOW .. "/sistema [txt]" .. ANSI.RESET .. " | "
          .. ANSI.YELLOW .. "/perfis" .. ANSI.RESET .. " | "
          .. ANSI.YELLOW .. "/rag [status/index/buscar]" .. ANSI.RESET .. " | "
          .. ANSI.YELLOW .. "/salvar" .. ANSI.RESET .. " | "
          .. ANSI.YELLOW .. "/sair" .. ANSI.RESET)
    print(ANSI.CYAN .. "================================================================================\n" .. ANSI.RESET)
end

-- Menu de seleção de perfis
local function select_profile()
    print("\n" .. ANSI.BOLD .. "Perfis de Modelo Disponíveis no Disco GLA:" .. ANSI.RESET)
    print("  " .. ANSI.GREEN .. "[1]" .. ANSI.RESET .. " " .. streamer.MODEL_PROFILES["1"].label)
    print("  " .. ANSI.GREEN .. "[2]" .. ANSI.RESET .. " " .. streamer.MODEL_PROFILES["2"].label)
    print("  " .. ANSI.GREEN .. "[3]" .. ANSI.RESET .. " " .. streamer.MODEL_PROFILES["3"].label)
    io.write("\nEscolha o perfil [1, 2 ou 3, Enter=3]: ")
    io.flush()
    local choice = io.read()
    if not choice or choice == "" then return "3" end
    choice = choice:gsub("^%s*(.-)%s*$", "%1")
    if choice == "1" or choice == "gemma2b" then return "1" end
    if choice == "2" or choice == "gemma4" then return "2" end
    return "3"
end

-- Parser de argumentos de linha de comando
local function parse_args(args)
    local config = {
        profile = "3",
        max_seq_len = 2048,
        max_tokens = 1024,
        max_think_tokens = 1024,
        char_delay = 0.0,
        enable_thinking = true,
        temperature = 0.7,
        repeat_penalty = 1.10,
        pin_ram = true,
    }

    local i = 1
    while i <= #args do
        local a = args[i]
        if a == "-p" or a == "--perfil" then
            config.profile = args[i + 1] or "3"
            i = i + 2
        elseif a == "-m" or a == "--max-tokens" then
            config.max_tokens = tonumber(args[i + 1]) or 1024
            i = i + 2
        elseif a == "-s" or a == "--max-seq" then
            config.max_seq_len = tonumber(args[i + 1]) or 2048
            i = i + 2
        elseif a == "-d" or a == "--delay" then
            config.char_delay = tonumber(args[i + 1]) or 0.0
            i = i + 2
        elseif a == "-t" or a == "--temp" then
            config.temperature = tonumber(args[i + 1]) or 0.7
            i = i + 2
        elseif a == "--no-think" then
            config.enable_thinking = false
            i = i + 1
        elseif a == "--no-pin" then
            config.pin_ram = false
            i = i + 1
        else
            i = i + 1
        end
    end
    return config
end

-- Loop Principal de Chat
local function run_chat()
    local config = parse_args(arg or {})

    local engine = nil
    local current_system_prompt = ""

    local function load_engine(prof_id)
        if engine then
            print(ANSI.YELLOW .. "  Liberando recursos do modelo anterior..." .. ANSI.RESET)
            engine:close()
            engine = nil
        end
        local prof_info = streamer.MODEL_PROFILES[prof_id] or streamer.MODEL_PROFILES["3"]
        print(string.format("\n%s  Carregando perfil [%s] -> %s...%s",
            ANSI.CYAN, prof_info.id, prof_info.label, ANSI.RESET))
        local t0 = wall_clock_sec()
        engine = streamer.new(prof_id, config.max_seq_len, config.pin_ram, 3)
        local boot_ms = (wall_clock_sec() - t0) * 1000
        print(string.format("%s  [OK] Motor pronto em %.1f ms!%s", ANSI.GREEN, boot_ms, ANSI.RESET))
        current_system_prompt = DEFAULT_SYSTEM_PROMPTS[prof_info.id] or DEFAULT_SYSTEM_PROMPTS["3"]
        return prof_info
    end

    local current_prof_info = load_engine(config.profile)
    print_banner(current_prof_info.label, engine.n_layers, engine.vocab_size, engine.max_seq_len, config.max_tokens, config.enable_thinking)

    local last_query = ""
    local last_thought = ""
    local last_answer = ""

    while true do
        io.write(ANSI.BOLD .. ANSI.GREEN .. "\nVocê: " .. ANSI.RESET)
        io.flush()
        local line = io.read()
        if line == nil then break end
        line = line:gsub("^%s*(.-)%s*$", "%1")

        if line == "" then
            -- Linha vazia: continua
        elseif line == "/sair" or line == "/exit" or line == "sair" or line == "exit" then
            print(ANSI.YELLOW .. "\nEncerrando sessão GNC-X LuaJIT. Até logo!\n" .. ANSI.RESET)
            break
        elseif line == "/memoria" or line == "/ram" then
            local ram_mb = get_process_memory_mb()
            print(string.format("%s  [Telemetria] Consumo de RAM do processo: %.2f MB%s", ANSI.CYAN, ram_mb, ANSI.RESET))
            local geo = engine:get_geo_telemetry()
            if geo then
                print(string.format("%s  [GMM] Memória Geométrica: %d/%d frames | Fragmentação: %.4f%%%s",
                    ANSI.MAGENTA, geo.used_frames, geo.capacity_frames, geo.fragmentation * 100.0, ANSI.RESET))
            end
        elseif line == "/geometria" or line == "/geomem" then
            local geo = engine:get_geo_telemetry()
            if geo then
                print(ANSI.MAGENTA .. "\n=== Telemetria do GeometricMemoryManager (GMM) ===" .. ANSI.RESET)
                print(string.format("  Capacidade Total:  %d frames geométricos", geo.capacity_frames))
                print(string.format("  Quadros Alocados:  %d frames", geo.used_frames))
                print(string.format("  Fragmentação:      %.4f%% (Zero fragmentação externa)", geo.fragmentation * 100.0))
                print(string.format("  Topologia D(k, d): Mapeamento bijetivo Phi(k, d, R)"))
                print(ANSI.GRAY .. "  Amostragem de Alocações KV-Cache por Camada:" .. ANSI.RESET)
                local max_show = math.min(10, #geo.allocations)
                for idx = 1, max_show do
                    local a = geo.allocations[idx]
                    print(string.format("    Camada %02d [%s] -> Crivo k=%d, d=%d, R=%d | Frame: #%02d | DDR3 Bank: #%d | L1d Set: #%02d",
                        a.layer, a.kind, a.k, a.d, a.R, a.idx, a.ddr3_bank or 0, a.l1_set or 0))
                end
                if #geo.allocations > max_show then
                    print(string.format("    ... (+%d alocações em órbita geométrica)", #geo.allocations - max_show))
                end
                print(ANSI.MAGENTA .. "=================================================\n" .. ANSI.RESET)
            else
                print(ANSI.YELLOW .. "  [GMM] Telemetria geométrica não disponível." .. ANSI.RESET)
            end
        elseif line == "/perfis" then
            local chosen = select_profile()
            current_prof_info = load_engine(chosen)
            print_banner(current_prof_info.label, engine.n_layers, engine.vocab_size, engine.max_seq_len, config.max_tokens, config.enable_thinking)
        elseif line:find("^/raciocinio") or line:find("^/think") then
            local sub = line:match("^/%w+%s*(.*)$")
            sub = sub and sub:lower():gsub("^%s*(.-)%s*$", "%1") or ""
            if sub == "on" or sub == "1" or sub == "sim" or sub == "ativo" then
                config.enable_thinking = true
                print(ANSI.GREEN .. "  [Raciocínio] Modo de pensamento ATIVADO." .. ANSI.RESET)
            elseif sub == "off" or sub == "0" or sub == "nao" or sub == "desativo" then
                config.enable_thinking = false
                print(ANSI.YELLOW .. "  [Raciocínio] Modo de pensamento DESATIVADO (respostas imediatas sem <think>)." .. ANSI.RESET)
            else
                config.enable_thinking = not config.enable_thinking
                local status = config.enable_thinking and "ATIVADO" or "DESATIVADO"
                print(ANSI.CYAN .. "  [Raciocínio] Modo de pensamento alternado para: " .. status .. ANSI.RESET)
            end
        elseif line:find("^/tokens") then
            local n = tonumber(line:match("^/tokens%s+(%d+)"))
            if n and n > 0 then
                config.max_tokens = n
                print(ANSI.GREEN .. string.format("  [Orçamento] Limite de tokens da resposta ajustado para: %d", n) .. ANSI.RESET)
            else
                print(ANSI.YELLOW .. string.format("  [Orçamento] Limite atual da resposta: %d tokens. Use: /tokens <numero>", config.max_tokens) .. ANSI.RESET)
            end
        elseif line:find("^/sistema") or line:find("^/system") then
            local s = line:match("^/%w+%s*(.*)$")
            if s and s ~= "" then
                current_system_prompt = s
                print(ANSI.GREEN .. "  [Sistema] Nova instrução de sistema aplicada:\n  \"" .. s .. "\"" .. ANSI.RESET)
            else
                print(ANSI.YELLOW .. "  [Sistema] Instrução atual:\n  \"" .. current_system_prompt .. "\"\n  Para alterar: /sistema <novo prompt>" .. ANSI.RESET)
            end
        elseif line:find("^/temperatura") or line:find("^/temp") then
            local t = tonumber(line:match("^/%w+%s+([%d%.]+)"))
            if t and t >= 0 then
                config.temperature = t
                print(ANSI.GREEN .. string.format("  [Amostragem] Temperatura ajustada para: %.2f", t) .. ANSI.RESET)
            else
                print(ANSI.YELLOW .. string.format("  [Amostragem] Temperatura atual: %.2f. Use: /temperatura 0.7", config.temperature) .. ANSI.RESET)
            end
        elseif line:find("^/delay") or line:find("^/velocidade") then
            local d = tonumber(line:match("^/%w+%s+([%d%.]+)"))
            if d and d >= 0 then
                config.char_delay = d
                print(ANSI.GREEN .. string.format("  [Typewriter] Cadência ajustada para: %.3fs por caractere (0 = imediato)", d) .. ANSI.RESET)
            else
                print(ANSI.YELLOW .. string.format("  [Typewriter] Delay atual: %.3fs. Use: /delay 0.015 ou /delay 0", config.char_delay) .. ANSI.RESET)
            end
        elseif line == "/ajuda" or line == "/help" then
            print(ANSI.CYAN .. "\nComandos disponíveis:" .. ANSI.RESET)
            print("  /raciocinio [on/off] — ativa ou desativa o pensamento interno <think>")
            print("  /tokens <n>          — ajusta o tamanho máximo da resposta (ex: /tokens 1024)")
            print("  /sistema <texto>     — altera a instrução de sistema (personalidade)")
            print("  /perfis              — lista e alterna entre os modelos (Gemma 2B / Gemma 4 / Qwen3)")
            print("  /temperatura <n>     — ajusta a criatividade da amostragem (padrão: 0.70)")
            print("  /delay <segundos>    — ajusta o efeito de máquina de escrever (padrão: 0 = imediato)")
            print("  /memoria             — exibe o consumo real de RAM do processo")
            print("  /geometria           — exibe a telemetria do GeometricMemoryManager (GMM)")
            print("  /rag [subcomando]    — gerencia a memória RAG (index, buscar, status, limpar)")
            print("  /salvar [arq.md]     — exporta a última interação para arquivo Markdown")
            print("  /sair                — encerra a sessão com segurança")
        elseif line:find("^/rag") then
            local args = line:match("^/rag%s*(.*)$")
            args = args and args:gsub("^%s*(.-)%s*$", "%1") or ""
            local subcmd, param = args:match("^(%w+)%s*(.*)$")
            subcmd = subcmd and subcmd:lower() or "status"

            if subcmd == "index" then
                if not param or param == "" then
                    print(ANSI.YELLOW .. "  Use: /rag index <caminho_do_arquivo>" .. ANSI.RESET)
                else
                    local f = io.open(param, "r")
                    if not f then
                        print(ANSI.RED .. "  [RAG] Arquivo não encontrado: " .. param .. ANSI.RESET)
                    else
                        local content = f:read("*a")
                        f:close()
                        local ok_idx, n_added = GNCXSemanticRAG:index_text(content, param)
                        if ok_idx then
                            print(string.format("%s  [RAG] Indexados %d blocos semânticos do arquivo %s (%d chars).%s",
                                ANSI.GREEN, n_added, param, #content, ANSI.RESET))
                        else
                            print(string.format("%s  [RAG] Falha na indexação: %s%s", ANSI.RED, tostring(n_added), ANSI.RESET))
                        end
                    end
                end
            elseif subcmd == "buscar" then
                if not param or param == "" then
                    print(ANSI.YELLOW .. "  Use: /rag buscar <termo de consulta>" .. ANSI.RESET)
                else
                    local results = GNCXSemanticRAG:retrieve_top_k(param, 3, 0.60)
                    print(string.format("%s  [RAG] Resultados para: \"%s\" (%d encontrados)%s", ANSI.CYAN, param, #results, ANSI.RESET))
                    for idx, res in ipairs(results) do
                        print(string.format("    %s#%d (Similaridade: %.3f | Origem: %s):%s\n    \"%s\"",
                            ANSI.YELLOW, idx, res.score, res.chunk.source, ANSI.WHITE,
                            res.chunk.text:sub(1, 150):gsub("\n", " ") .. "..."))
                    end
                end
            elseif subcmd == "limpar" then
                GNCXSemanticRAG:clear()
                print(ANSI.GREEN .. "  [RAG] Memória semântica RAG limpa." .. ANSI.RESET)
            else
                print(string.format("%s  [RAG Status] Blocos na memória: %d | Total indexado: %d chars%s",
                    ANSI.CYAN, #GNCXSemanticRAG.chunks, GNCXSemanticRAG.total_doc_chars, ANSI.RESET))
                print("  Comandos: /rag index <arq> | /rag buscar <termo> | /rag limpar")
            end
        elseif line:find("^/salvar") then
            local target_name = line:match("^/salvar%s*(.*)$")
            if last_answer == "" then
                print(ANSI.RED .. "  Nenhuma resposta recente para salvar!" .. ANSI.RESET)
            else
                local saved_path, err = save_markdown(target_name, last_query, last_thought, last_answer, current_prof_info.id)
                if saved_path then
                    print(string.format("%s  [OK] Diálogo salvo com sucesso em: %s%s", ANSI.GREEN, saved_path, ANSI.RESET))
                else
                    print(string.format("%s  [ERRO] %s%s", ANSI.RED, err, ANSI.RESET))
                end
            end
        else
            -- Prompt do Usuário
            last_query = line
            last_thought = ""
            last_answer = ""

            -- Injeção Semântica RAG (Fase 2)
            local effective_user_prompt = line
            local rag_info = nil
            if #GNCXSemanticRAG.chunks > 0 then
                local top_results = GNCXSemanticRAG:retrieve_top_k(line, 2, 0.70)
                if #top_results > 0 then
                    local retrieved_text = {}
                    local retrieved_chars = 0
                    for _, r in ipairs(top_results) do
                        table.insert(retrieved_text, r.chunk.text)
                        retrieved_chars = retrieved_chars + #r.chunk.text
                    end
                    local savings_pct = math.max(0, (1.0 - (retrieved_chars / math.max(GNCXSemanticRAG.total_doc_chars, 1))) * 100.0)
                    rag_info = string.format("[RAG Local: %d trechos relevantes injetados | Economia de prefill: %.1f%%]",
                        #top_results, savings_pct)
                    effective_user_prompt = string.format("%s\n\n[Contexto Semântico Recuperado via RAG]:\n%s",
                        line, table.concat(retrieved_text, "\n\n"))
                    print(string.format("%s  %s%s", ANSI.CYAN, rag_info, ANSI.RESET))
                end
            end

            local formatted_prompt = engine.tokenizer:format_chat(effective_user_prompt, current_system_prompt, config.enable_thinking)

            io.write(ANSI.BOLD .. ANSI.BLUE .. "\nGNC-X: " .. ANSI.RESET)
            io.flush()

            local in_think_block = false
            local think_buffer = {}
            local answer_buffer = {}
            local think_tokens = 0
            local answer_tokens = 0
            local gen_t0 = wall_clock_sec()

            local prompt_tokens = engine.tokenizer:encode(formatted_prompt)
            local logits = engine:prefill(prompt_tokens)
            local decode_t0 = wall_clock_sec()

            local all_tokens = {}
            for _, t in ipairs(prompt_tokens) do table.insert(all_tokens, t) end

            local cur_pos = #prompt_tokens

            -- Monta conjunto de tokens de parada oficiais
            local stop_ids = { [engine.tokenizer.eos_id] = true }
            for _, st in ipairs({"<end_of_turn>", "<turn|>", "<|turn>", "<|im_end|>", "<|endoftext|>", "<|im_start|>"}) do
                local tid = engine.tokenizer.token_to_id[st]
                if tid then stop_ids[tid] = true end
            end

            -- Loop de geração com orçamento desacoplado:
            -- `answer_tokens` só é consumido na resposta final!
            local max_safe_pos = engine.max_seq_len - 1

            while cur_pos < max_safe_pos do
                local next_token = engine:sample(logits, all_tokens, config.repeat_penalty, config.temperature, 0.9)
                if stop_ids[next_token] then
                    break
                end

                table.insert(all_tokens, next_token)
                local tok_bytes = engine.tokenizer:token_bytes(next_token)

                -- Detecção de tags de raciocínio CoT (Qwen3 e Gemma 4)
                if tok_bytes:find("<think>") or tok_bytes:find("<|think|>") then
                    in_think_block = true
                    io.write(ANSI.GRAY .. ANSI.ITALIC .. "\n[Pensamento: ")
                    io.flush()
                elseif tok_bytes:find("</think>") or (in_think_block and (tok_bytes:find("<|channel>") or tok_bytes:find("<channel|>"))) then
                    in_think_block = false
                    io.write("]\n\n" .. ANSI.RESET)
                    io.flush()
                else
                    if in_think_block then
                        think_tokens = think_tokens + 1
                        table.insert(think_buffer, tok_bytes)
                        io.write(ANSI.GRAY .. ANSI.ITALIC .. tok_bytes .. ANSI.RESET)

                        -- Salvaguarda: se o modelo refletir além do limite seguro, força encerramento do pensamento
                        if think_tokens >= config.max_think_tokens then
                            in_think_block = false
                            io.write("] (tempo de pensamento concluído)\n\n" .. ANSI.RESET)
                            io.flush()
                        end
                    else
                        answer_tokens = answer_tokens + 1
                        table.insert(answer_buffer, tok_bytes)
                        io.write(tok_bytes)

                        -- Atingiu o orçamento estipulado de resposta
                        if answer_tokens >= config.max_tokens then
                            break
                        end
                    end
                    io.flush()

                    if config.char_delay > 0 then
                        sleep_ms(config.char_delay * 1000)
                    end
                end

                logits = engine:forward_token(next_token, cur_pos, true)
                cur_pos = cur_pos + 1
            end

            local now_t = wall_clock_sec()
            local elapsed = now_t - gen_t0
            local decode_elapsed = now_t - decode_t0
            local total_tokens = think_tokens + answer_tokens
            local tps = total_tokens / (decode_elapsed > 0 and decode_elapsed or 0.001)

            last_thought = table.concat(think_buffer)
            last_answer = table.concat(answer_buffer)

            local telemetria_msg
            if think_tokens > 0 then
                telemetria_msg = string.format("[GNC-X LuaJIT: %d resposta + %d pensamento (%d total) | %.2f s | %.2f tok/s | RAM: %.1f MB]",
                    answer_tokens, think_tokens, total_tokens, elapsed, tps, get_process_memory_mb())
            else
                telemetria_msg = string.format("[GNC-X LuaJIT: %d tokens | %.2f s | %.2f tok/s | RAM: %.1f MB]",
                    answer_tokens, elapsed, tps, get_process_memory_mb())
            end

            io.write("\n" .. ANSI.DIM .. ANSI.GRAY .. "\n" .. telemetria_msg .. "\n" .. ANSI.RESET)
            io.flush()
        end
    end

    if engine then
        engine:close()
    end
end

run_chat()
