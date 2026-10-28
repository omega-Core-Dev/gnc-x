--[[
  GNC-X LLM — Assistente & Copiloto de Sessão Linux (Fedora GNOME)
  Fase 6: Agente Autônomo Leve com Inicialização no Sistema,
  Telemetria de Boot, Lançador de Aplicativos e Execução de Comandos do SO.
--]]

local ffi = require("ffi")
local streamer = require("lua.gncx_streamer")

-- Declarações C para chamadas de sistema e relógio
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
    WHITE     = "\27[97m",
}

-- Funções utilitárias de coleta de dados do Linux
local function exec_cmd_capture(cmd)
    local handle = io.popen(cmd .. " 2>&1")
    if not handle then return "" end
    local output = handle:read("*a") or ""
    handle:close()
    return output:gsub("^%s*(.-)%s*$", "%1")
end

local function exec_gui_app(cmd)
    -- Executa aplicação gráfica desanexada sem bloquear o terminal
    local full_cmd = string.format("nohup %s >/dev/null 2>&1 &", cmd)
    os.execute(full_cmd)
end

local function get_system_health()
    local hour = tonumber(os.date("%H"))
    local saudacao = "Olá"
    if hour >= 5 and hour < 12 then
        saudacao = "Bom dia"
    elseif hour >= 12 and hour < 18 then
        saudacao = "Boa tarde"
    else
        saudacao = "Boa noite"
    end

    -- RAM
    local mem_total_mb = 0
    local mem_avail_mb = 0
    local f_mem = io.open("/proc/meminfo", "r")
    if f_mem then
        for line in f_mem:lines() do
            local k, v = line:match("^(%w+):%s+(%d+)%s+kB")
            if k == "MemTotal" then mem_total_mb = tonumber(v) / 1024.0 end
            if k == "MemAvailable" then mem_avail_mb = tonumber(v) / 1024.0 end
        end
        f_mem:close()
    end

    -- Uptime
    local uptime_str = exec_cmd_capture("uptime -p 2>/dev/null")
    if uptime_str == "" then
        uptime_str = "sistema ativo"
    else
        uptime_str = uptime_str:gsub("^up%s+", "")
    end

    -- Disco /
    local disk_avail = exec_cmd_capture("df -h / | awk 'NR==2 {print $4}' 2>/dev/null")
    local user_name = os.getenv("USER") or "Zaqueu"

    return {
        saudacao = saudacao,
        user = user_name,
        mem_total_gb = mem_total_mb / 1024.0,
        mem_avail_gb = mem_avail_mb / 1024.0,
        mem_used_pct = mem_total_mb > 0 and ((mem_total_mb - mem_avail_mb) / mem_total_mb * 100.0) or 0,
        uptime = uptime_str,
        disk_avail = disk_avail,
        date_str = os.date("%d/%m/%Y %H:%M:%S")
    }
end

-- Banner e Apresentação Inicial Compacta (68 Colunas)
local function show_welcome_banner(h)
    print(ANSI.CYAN .. ANSI.BOLD)
    print(" ╭──────────────────────────────────────────────────────────────────╮")
    print(" │               GNC-X COPILOTO AUTÔNOMO DE SESSÃO                  │")
    print(" │         Alimentado por Qwen3 1.7B | C++20 Geometric MM           │")
    print(" ╰──────────────────────────────────────────────────────────────────╯" .. ANSI.RESET)
    
    print(string.format("%s%s, %s!%s %s(%s)%s", 
        ANSI.BOLD .. ANSI.GREEN, h.saudacao, h.user, ANSI.RESET, ANSI.GRAY, h.date_str, ANSI.RESET))
    print(string.format("%s  • RAM livre:   %s%.2f GB de %.2f GB (%d%% em uso)%s",
        ANSI.CYAN, ANSI.WHITE, h.mem_avail_gb, h.mem_total_gb, math.floor(h.mem_used_pct), ANSI.RESET))
    print(string.format("%s  • Disco (/):   %s%s livres%s",
        ANSI.CYAN, ANSI.WHITE, h.disk_avail, ANSI.RESET))
    print(string.format("%s  • Uptime:      %s%s%s",
        ANSI.CYAN, ANSI.WHITE, h.uptime, ANSI.RESET))
    print(ANSI.GRAY .. " ──────────────────────────────────────────────────────────────────" .. ANSI.RESET)
    print(ANSI.YELLOW .. " Como posso te ajudar na sua sessão hoje?" .. ANSI.RESET)
    print(ANSI.GRAY .. " (Ex: 'abrir navegador', 'arquivos', 'status' ou pergunte algo)" .. ANSI.RESET)
    print(ANSI.GRAY .. " Comandos: /ajuda ou /sair\n" .. ANSI.RESET)
end

-- Comandos rápidos mapeados para ação instantânea (< 1ms)
local QUICK_ACTIONS = {
    {
        patterns = {"^abrir navegador", "^abre o navegador", "^navegador", "^abrir internet", "^internet", "^firefox"},
        desc = "Abrir o navegador Web (Firefox)",
        action = function()
            print(ANSI.GREEN .. "  [Ação] Iniciando Firefox em segundo plano..." .. ANSI.RESET)
            exec_gui_app("firefox")
            print(ANSI.CYAN .. "  [OK] Navegador disparado na interface gráfica!" .. ANSI.RESET)
            return true
        end
    },
    {
        patterns = {"^abrir arquivos", "^abre os arquivos", "^arquivos", "^pastas", "^meus arquivos", "^gerenciador de arquivos", "^nautilus"},
        desc = "Abrir o Gerenciador de Arquivos (Nautilus)",
        action = function()
            print(ANSI.GREEN .. "  [Ação] Abrindo Gerenciador de Arquivos..." .. ANSI.RESET)
            exec_gui_app("nautilus ~")
            print(ANSI.CYAN .. "  [OK] Pasta pessoal aberta!" .. ANSI.RESET)
            return true
        end
    },
    {
        patterns = {"^abrir editor", "^editor de texto", "^bloco de notas", "^editor", "^gnome%-text%-editor", "^gedit"},
        desc = "Abrir o Editor de Texto",
        action = function()
            print(ANSI.GREEN .. "  [Ação] Abrindo Editor de Texto..." .. ANSI.RESET)
            exec_gui_app("gnome-text-editor")
            print(ANSI.CYAN .. "  [OK] Editor de texto aberto!" .. ANSI.RESET)
            return true
        end
    },
    {
        patterns = {"^abrir terminal", "^novo terminal", "^terminal", "^ptyxis"},
        desc = "Abrir uma nova janela de terminal",
        action = function()
            print(ANSI.GREEN .. "  [Ação] Abrindo novo terminal..." .. ANSI.RESET)
            exec_gui_app("ptyxis")
            print(ANSI.CYAN .. "  [OK] Janela de terminal aberta!" .. ANSI.RESET)
            return true
        end
    },
    {
        patterns = {"^status", "^saude", "^diagnostico", "^relatorio"},
        desc = "Exibir saúde atual do sistema",
        action = function()
            local h = get_system_health()
            print(ANSI.CYAN .. "\n=== Diagnóstico Rápido do Sistema ===" .. ANSI.RESET)
            print(string.format("  RAM Total:       %.2f GB", h.mem_total_gb))
            print(string.format("  RAM Disponível:  %.2f GB (%.1f%% ocupada)", h.mem_avail_gb, h.mem_used_pct))
            print(string.format("  Espaço em Disco: %s livres em /", h.disk_avail))
            print(string.format("  Uptime:          %s", h.uptime))
            print(ANSI.CYAN .. "====================================\n" .. ANSI.RESET)
            return true
        end
    },
    {
        patterns = {"^compilar", "^compilar linux", "^build"},
        desc = "Recompilar kernels C++20 do GNC-X",
        action = function()
            print(ANSI.YELLOW .. "  [Ação] Executando compilação do núcleo nativo C++20..." .. ANSI.RESET)
            local res = exec_cmd_capture("./compilar_linux.sh")
            print(res)
            print(ANSI.GREEN .. "  [OK] Processo de compilação concluído." .. ANSI.RESET)
            return true
        end
    },
}

-- Motor de Embedding para Roteamento Semântico Instantâneo (Fase 2)
local emb_mod = nil
local embedding_engine = nil

local function get_embedding_engine()
    if not embedding_engine then
        local ok, mod = pcall(require, "lua.gncx_embedding")
        if ok then
            local ok_init, instance = pcall(mod.new, "models/embeddinggemma-2-Q8_0.gguf", 256)
            if ok_init then
                embedding_engine = instance
                -- Pré-vetoriza ferramentas para tempo de inferência ultrarrápido
                for _, item in ipairs(QUICK_ACTIONS) do
                    if not item.vec then
                        item.vec = embedding_engine:encode(item.desc)
                    end
                end
            end
        end
    end
    return embedding_engine
end

local function try_quick_action(text)
    local clean = text:lower():gsub("^%s*(.-)%s*$", "%1")

    -- 1. Rota Ultrarrápida por Regex (< 0.1 ms)
    for _, item in ipairs(QUICK_ACTIONS) do
        for _, pat in ipairs(item.patterns) do
            if clean:find(pat) then
                return item.action()
            end
        end
    end

    -- 2. Rota Semântica via EmbeddingGemma 2 (Fase 2: < 5 ms com cache)
    local emb = get_embedding_engine()
    if emb then
        local t0 = wall_clock_sec()
        local query_vec = emb:encode(clean)
        local best_item = nil
        local best_sim = -1.0

        for _, item in ipairs(QUICK_ACTIONS) do
            if item.vec then
                local sim = emb:similarity(query_vec, item.vec)
                if sim > best_sim then
                    best_sim = sim
                    best_item = item
                end
            end
        end

        local decision_time_ms = (wall_clock_sec() - t0) * 1000

        -- Limiar semântico para despacho determinístico sem consumir tokens de LLM
        if best_item and best_sim >= 0.88 then
            print(string.format("%s  [Roteador Semântico GNC-X] Ferramenta: \"%s\" (Similaridade: %.3f | Decisão: %.1f ms)%s",
                ANSI.CYAN, best_item.desc, best_sim, decision_time_ms, ANSI.RESET))
            return best_item.action()
        end
    end

    return false
end

-- Prompt de Sistema Especializado do Copiloto
local AGENT_SYSTEM_PROMPT = [[
Você é o Copiloto Autônomo GNC-X rodando no Fedora Linux (Intel Celeron, 6GB RAM).
Sua missão é ajudar o usuário nas tarefas do dia a dia do sistema operacional e do projeto.
Diretrizes fundamentais:
1. Responda em português do Brasil de forma prestativa, direta e objetiva.
2. Quando a solicitação demandar executar um comando no Linux (inspeção, busca, verificação), forneça o comando dentro de tags: <cmd>comando_aqui</cmd>.
3. Para abrir programas da interface, sugira comandos seguros como firefox, nautilus, gnome-text-editor.
4. Mantenha explicações curtas e concentre-se na utilidade prática.
]]

-- Loop Principal do Agente
local function run_agent()
    -- Solicita redimensionamento compacto para o terminal (22 linhas x 70 colunas) e limpa a tela
    io.write("\27[8;22;70t\27[2J\27[H")
    io.flush()

    local h = get_system_health()
    show_welcome_banner(h)

    -- Configuração do Motor GNC-X
    local profile_id = "3" -- Padrão: Qwen3 1.7B (melhor seguimento de comandos)
    local max_seq_len = 2048
    local pin_ram = true
    local enable_thinking = false -- Pensamento desativado para resposta rápida em comandos de sistema

    local engine = nil
    local function ensure_engine()
        if not engine then
            print(ANSI.YELLOW .. "  [GNC-X] Inicializando motor neural Qwen3 1.7B..." .. ANSI.RESET)
            local t0 = wall_clock_sec()
            engine = streamer.new(profile_id, max_seq_len, pin_ram, 3)
            local ms = (wall_clock_sec() - t0) * 1000
            print(string.format("%s  [GNC-X] Motor pronto em %.1f ms!%s\n", ANSI.GREEN, ms, ANSI.RESET))
        end
        return engine
    end

    while true do
        io.write(ANSI.BOLD .. ANSI.GREEN .. "Zaqueu: " .. ANSI.RESET)
        io.flush()
        local line = io.read()
        if line == nil then break end
        line = line:gsub("^%s*(.-)%s*$", "%1")

        if line == "" then
            -- linha vazia: ignora
        elseif line == "/sair" or line == "/exit" or line == "sair" or line == "exit" then
            print(ANSI.YELLOW .. "\nTenha um excelente dia! Sessão do GNC-X finalizada.\n" .. ANSI.RESET)
            if engine then
                engine:close()
                engine = nil
            end
            break
        elseif line == "/ajuda" or line == "/help" then
            print(ANSI.CYAN .. "\n=== Capacidades do Assistente GNC-X ===" .. ANSI.RESET)
            print("  Comandos Rápidos Imediatos:")
            print("    'abrir navegador'  -> Lança o Firefox")
            print("    'arquivos'         -> Abre o gerenciador de pastas")
            print("    'editor'           -> Abre o editor de texto")
            print("    'status'           -> Relatório de RAM e disco")
            print("    'compilar'         -> Compila os kernels nativos C++")
            print("\n  Comandos de Controle:")
            print("    /app <comando>     -> Executa qualquer app gráfico em segundo plano")
            print("    /sh <comando>      -> Executa um comando bash diretamente e exibe o resultado")
            print("    /raciocinio [on/off]-> Ativa/desativa reflexão profunda <think>")
            print("    /memoria           -> Exibe a telemetria do motor GNC-X")
            print("    /sair              -> Encerra o assistente\n" .. ANSI.RESET)
        elseif line:find("^/app%s+") then
            local app_cmd = line:match("^/app%s+(.+)$")
            if app_cmd then
                print(ANSI.GREEN .. "  [Lançador] Executando: " .. app_cmd .. ANSI.RESET)
                exec_gui_app(app_cmd)
                print(ANSI.CYAN .. "  [OK] Aplicativo disparado com sucesso!" .. ANSI.RESET)
            end
        elseif line:find("^/sh%s+") then
            local sh_cmd = line:match("^/sh%s+(.+)$")
            if sh_cmd then
                print(ANSI.YELLOW .. "  [Terminal] Executando: " .. sh_cmd .. ANSI.RESET)
                local out = exec_cmd_capture(sh_cmd)
                if out ~= "" then
                    print(ANSI.WHITE .. out .. ANSI.RESET)
                else
                    print(ANSI.GRAY .. "  (Comando finalizado sem saída de texto)" .. ANSI.RESET)
                end
            end
        elseif line:find("^/raciocinio") then
            local sub = line:match("^/%w+%s*(.*)$")
            sub = sub and sub:lower():gsub("^%s*(.-)%s*$", "%1") or ""
            if sub == "on" or sub == "1" or sub == "sim" then
                enable_thinking = true
                print(ANSI.GREEN .. "  [Raciocínio] Modo de pensamento aprofundado ATIVADO." .. ANSI.RESET)
            elseif sub == "off" or sub == "0" or sub == "nao" then
                enable_thinking = false
                print(ANSI.YELLOW .. "  [Raciocínio] Modo de pensamento DESATIVADO (máxima velocidade)." .. ANSI.RESET)
            else
                enable_thinking = not enable_thinking
                print(ANSI.CYAN .. "  [Raciocínio] Alternado para: " .. (enable_thinking and "ATIVADO" or "DESATIVADO") .. ANSI.RESET)
            end
        elseif line == "/memoria" then
            if engine then
                local geo = engine:get_geo_telemetry()
                if geo then
                    print(string.format("%s  [GMM] Quadros Usados: %d/%d | Fragmentação: %.4f%%%s",
                        ANSI.MAGENTA, geo.used_frames, geo.capacity_frames, geo.fragmentation * 100.0, ANSI.RESET))
                end
            else
                print(ANSI.GRAY .. "  (Motor neural ainda não instanciado na memória)" .. ANSI.RESET)
            end
        else
            -- 1. Tenta Ação Imediata (< 1ms)
            local handled = try_quick_action(line)
            
            -- 2. Se não for comando imediato, consulta o motor neural
            if not handled then
                local eng = ensure_engine()
                local formatted = eng.tokenizer:format_chat(line, AGENT_SYSTEM_PROMPT, enable_thinking)

                io.write(ANSI.BOLD .. ANSI.BLUE .. "\nGNC-X: " .. ANSI.RESET)
                io.flush()

                local prompt_tokens = eng.tokenizer:encode(formatted)
                local logits = eng:prefill(prompt_tokens)

                local all_tokens = {}
                for _, t in ipairs(prompt_tokens) do table.insert(all_tokens, t) end
                local cur_pos = #prompt_tokens

                local stop_ids = { [eng.tokenizer.eos_id] = true }
                for _, st in ipairs({"<end_of_turn>", "<turn|>", "<|turn>", "<|im_end|>", "<|endoftext|>", "<|im_start|>"}) do
                    local tid = eng.tokenizer.token_to_id[st]
                    if tid then stop_ids[tid] = true end
                end

                local answer_buffer = {}
                local in_think = false
                local max_tokens = 512
                local answer_count = 0

                while cur_pos < eng.max_seq_len - 1 do
                    local next_token = eng:sample(logits, all_tokens, 1.10, 0.65, 0.9)
                    if stop_ids[next_token] then break end

                    table.insert(all_tokens, next_token)
                    local tok_bytes = eng.tokenizer:token_bytes(next_token)

                    if tok_bytes:find("<think>") or tok_bytes:find("<|think|>") then
                        in_think = true
                        io.write(ANSI.GRAY .. ANSI.ITALIC .. "\n[Pensamento: ")
                        io.flush()
                    elseif tok_bytes:find("</think>") or (in_think and tok_bytes:find("<|im_end|>")) then
                        in_think = false
                        io.write("]\n\n" .. ANSI.RESET)
                        io.flush()
                    else
                        if in_think then
                            io.write(ANSI.GRAY .. ANSI.ITALIC .. tok_bytes .. ANSI.RESET)
                        else
                            answer_count = answer_count + 1
                            table.insert(answer_buffer, tok_bytes)
                            io.write(tok_bytes)
                            if answer_count >= max_tokens then break end
                        end
                        io.flush()
                    end

                    logits = eng:forward_token(next_token, cur_pos, true)
                    cur_pos = cur_pos + 1
                end

                io.write("\n")
                io.flush()

                -- Verifica se o modelo sugeriu um comando na tag <cmd>...</cmd>
                local full_answer = table.concat(answer_buffer)
                local suggested_cmd = full_answer:match("<cmd>(.-)</cmd>")
                if not suggested_cmd then
                    -- Tenta capturar bloco ```bash ... ```
                    suggested_cmd = full_answer:match("```bash%s*(.-)%s*```")
                end

                if suggested_cmd and suggested_cmd ~= "" then
                    suggested_cmd = suggested_cmd:gsub("^%s*(.-)%s*$", "%1")
                    print(string.format("\n%s  [Ação Sugerida Detectada]: %s%s%s", 
                        ANSI.YELLOW, ANSI.BOLD .. ANSI.WHITE, suggested_cmd, ANSI.RESET))
                    io.write(ANSI.YELLOW .. "  Deseja executar este comando no sistema agora? (s/N): " .. ANSI.RESET)
                    io.flush()
                    local resp = io.read() or ""
                    resp = resp:lower():gsub("^%s*(.-)%s*$", "%1")

                    if resp == "s" or resp == "sim" or resp == "y" then
                        print(ANSI.GREEN .. "  Executando no Linux...\n" .. ANSI.RESET)
                        local cmd_out = exec_cmd_capture(suggested_cmd)
                        if cmd_out ~= "" then
                            print(ANSI.CYAN .. "─── [Resultado da Máquina] ───" .. ANSI.RESET)
                            print(ANSI.WHITE .. cmd_out .. ANSI.RESET)
                            print(ANSI.CYAN .. "──────────────────────────────\n" .. ANSI.RESET)
                        else
                            print(ANSI.GREEN .. "  [OK] Comando executado com sucesso (sem retorno de texto).\n" .. ANSI.RESET)
                        end
                    else
                        print(ANSI.GRAY .. "  Execução cancelada pelo usuário.\n" .. ANSI.RESET)
                    end
                end
            end
        end
    end
end

run_agent()
