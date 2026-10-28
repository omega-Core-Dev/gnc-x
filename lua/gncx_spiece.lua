--[[
  GNC-X LLM — Tokenizador SentencePiece BPE em LuaJIT
  Suporte a modelos Google (Gemma 2B, Gemma 4 E2B, LLaMA SPiece)
--]]

local GNCXSentencePiece = {}
GNCXSentencePiece.__index = GNCXSentencePiece

local SPIECE_UNDERLINE = "\226\150\129" -- ' ' (U+2581 em UTF-8)

function GNCXSentencePiece.new(tokens, scores, token_types, bos_id, eos_id, unk_id)
    local self = setmetatable({}, GNCXSentencePiece)

    self.id_to_token = tokens
    self.vocab_size = #tokens
    self.scores = scores or {}
    self.token_types = token_types or {}
    self.bos_id = bos_id or 2
    self.eos_id = eos_id or 1
    self.unk_id = unk_id or 3

    -- Mapa token -> id
    self.token_to_id = {}
    for idx = 1, #tokens do
        local tok = tokens[idx]
        local tid = idx - 1
        if self.token_to_id[tok] == nil then
            self.token_to_id[tok] = tid
        end
    end

    -- Tokens de controle especiais
    self.special_tokens = {
        "<start_of_turn>",
        "<end_of_turn>",
        "<|turn>",
        "<turn|>",
        "<|think|>",
        "<|channel>",
        "<channel|>",
        "<|tool>",
        "<tool|>",
        "<|tool_call>",
        "<tool_call|>",
        "<|tool_response>",
        "<tool_response|>",
        "<bos>",
        "<eos>",
        "<pad>",
    }

    self.special_ids = {}
    for _, sp in ipairs(self.special_tokens) do
        local tid = self.token_to_id[sp]
        if tid then self.special_ids[tid] = true end
    end

    self.is_gemma4_chat = (self.token_to_id["<|turn>"] ~= nil and self.token_to_id["<turn|>"] ~= nil)
    self.is_chatml = false
    self.has_native_thinking = (self.token_to_id["<|think|>"] ~= nil or self.token_to_id["<think>"] ~= nil)

    -- Mapeamento de fallback de bytes <0x00> .. <0xFF>
    self.byte_to_id = {}
    self.id_to_byte = {}
    for b = 0, 255 do
        local hex_tok = string.format("<0x%02X>", b)
        local tid = self.token_to_id[hex_tok]
        if tid then
            self.byte_to_id[b] = tid
            self.id_to_byte[tid] = b
        end
    end

    self.token_bytes_cache = {}
    self.seg_cache = {}

    return self
end

-- Codifica um segmento sem tokens especiais
function GNCXSentencePiece:_encode_segment(text)
    if not text or text == "" then return {} end

    local cached = self.seg_cache[text]
    if cached then return cached end

    -- Substitui espaços normais pelo caractere sublinhado SentencePiece
    local normalized = text:gsub(" ", SPIECE_UNDERLINE)

    -- Divide a string em caracteres UTF-8 individuais ou tokens de byte <0xXX>
    local symbols = {}
    local p = 1
    local len = #normalized

    while p <= len do
        local b = normalized:byte(p)
        local clen = 1
        if b >= 240 then clen = 4
        elseif b >= 224 then clen = 3
        elseif b >= 192 then clen = 2
        end

        local ch = normalized:sub(p, p + clen - 1)
        if self.token_to_id[ch] ~= nil then
            table.insert(symbols, ch)
        else
            -- Caractere fora do vocabulário direto: emite bytes <0xXX>
            for i = 1, #ch do
                local byte_val = ch:byte(i)
                local hex_tok = string.format("<0x%02X>", byte_val)
                table.insert(symbols, hex_tok)
            end
        end
        p = p + clen
    end

    -- Fusão iterativa BPE guiada pelos scores máximos do vocabulário GGUF
    local scores = self.scores
    local token_to_id = self.token_to_id

    while #symbols > 1 do
        local best_score = -1e30
        local best_idx = -1

        for i = 1, #symbols - 1 do
            local merged = symbols[i] .. symbols[i + 1]
            local tid = token_to_id[merged]
            if tid ~= nil then
                local sc = scores[tid + 1] or 0.0
                if sc > best_score then
                    best_score = sc
                    best_idx = i
                end
            end
        end

        if best_idx == -1 then break end

        symbols[best_idx] = symbols[best_idx] .. symbols[best_idx + 1]
        table.remove(symbols, best_idx + 1)
    end

    local ids = {}
    for _, s in ipairs(symbols) do
        table.insert(ids, token_to_id[s] or self.unk_id)
    end

    if #self.seg_cache < 50000 then
        self.seg_cache[text] = ids
    end
    return ids
end

-- Codifica texto completo
function GNCXSentencePiece:encode(text, add_bos)
    if add_bos == nil then add_bos = true end
    local ids = {}
    if add_bos and self.bos_id >= 0 then
        table.insert(ids, self.bos_id)
    end

    local cursor = 1
    local text_len = #text

    while cursor <= text_len do
        local earliest_pos = nil
        local matched_special = nil

        for _, sp in ipairs(self.special_tokens) do
            local pos = text:find(sp, cursor, true)
            if pos then
                if earliest_pos == nil or pos < earliest_pos then
                    earliest_pos = pos
                    matched_special = sp
                end
            end
        end

        if earliest_pos == nil then
            local rest = text:sub(cursor)
            if #rest > 0 then
                local s_ids = self:_encode_segment(rest)
                for _, id in ipairs(s_ids) do table.insert(ids, id) end
            end
            break
        else
            if earliest_pos > cursor then
                local segment = text:sub(cursor, earliest_pos - 1)
                local s_ids = self:_encode_segment(segment)
                for _, id in ipairs(s_ids) do table.insert(ids, id) end
            end
            local sp_id = self.token_to_id[matched_special]
            if sp_id ~= nil then
                table.insert(ids, sp_id)
            end
            cursor = earliest_pos + #matched_special
        end
    end

    return ids
end

-- Transforma 1 token em string de bytes para o typewriter
function GNCXSentencePiece:token_bytes(token_id)
    local cached = self.token_bytes_cache[token_id]
    if cached ~= nil then return cached end

    if token_id < 0 or token_id >= self.vocab_size then
        return ""
    end

    local tok = self.id_to_token[token_id + 1]
    if not tok then return "" end

    -- Se for token de controle especial, oculta no terminal
    if self.special_ids[token_id] or tok == "<bos>" or tok == "<eos>" or tok == "<pad>"
       or tok == "<start_of_turn>" or tok == "<end_of_turn>"
       or tok == "<|turn>" or tok == "<turn|>" or tok == "<|channel>" or tok == "<channel|>" then
        self.token_bytes_cache[token_id] = ""
        return ""
    end

    -- Se for fallback de byte <0xXX>
    local byte_val = self.id_to_byte[token_id]
    if byte_val ~= nil then
        local res = string.char(byte_val)
        self.token_bytes_cache[token_id] = res
        return res
    end

    -- Substitui ' ' (SPIECE_UNDERLINE) por espaço regular
    local res = tok:gsub(SPIECE_UNDERLINE, " ")
    self.token_bytes_cache[token_id] = res
    return res
end

-- Decodifica lista de tokens em texto limpo
function GNCXSentencePiece:decode(token_ids, skip_special)
    if skip_special == nil then skip_special = true end
    local parts = {}
    local pending_bytes = {}

    local function flush_bytes()
        if #pending_bytes > 0 then
            table.insert(parts, string.char(unpack(pending_bytes)))
            pending_bytes = {}
        end
    end

    for _, tid in ipairs(token_ids) do
        if tid >= 0 and tid < self.vocab_size then
            local tok = self.id_to_token[tid + 1]
            if tok then
                if skip_special and (self.special_ids[tid] or tok == "<bos>" or tok == "<eos>" or tok == "<pad>"
                   or tok == "<start_of_turn>" or tok == "<end_of_turn>"
                   or tok == "<|turn>" or tok == "<turn|>") then
                    -- pula token de controle
                else
                    local b = self.id_to_byte[tid]
                    if b ~= nil then
                        table.insert(pending_bytes, b)
                    else
                        flush_bytes()
                        local clean = tok:gsub(SPIECE_UNDERLINE, " ")
                        table.insert(parts, clean)
                    end
                end
            end
        end
    end

    flush_bytes()
    return table.concat(parts)
end

-- Formatação oficial do chat para modelos Google Gemma
function GNCXSentencePiece:format_chat(user_prompt, system_prompt, enable_thinking)
    system_prompt = system_prompt or "Você é o assistente GNC-X. Responda em português do Brasil de forma completa, detalhada e explicativa, desenvolvendo suas explicações com clareza e profundidade."

    if self.is_gemma4_chat then
        local parts = {}
        if system_prompt ~= "" then
            table.insert(parts, "<|turn>system\n" .. system_prompt:match("^%s*(.-)%s*$") .. "<turn|>\n")
        end
        table.insert(parts, "<|turn>user\n" .. user_prompt:match("^%s*(.-)%s*$") .. "<turn|>\n")
        table.insert(parts, "<|turn>model\n")
        return table.concat(parts)
    else
        local full_user = (system_prompt and system_prompt ~= "") and (system_prompt .. "\n\n" .. user_prompt) or user_prompt
        return string.format("<start_of_turn>user\n%s<end_of_turn>\n<start_of_turn>model\n", full_user:match("^%s*(.-)%s*$"))
    end
end

return GNCXSentencePiece
