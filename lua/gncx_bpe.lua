--[[
  GNC-X LLM — Tokenizador Byte-Level BPE em LuaJIT
  Fase 3: Suporte a Qwen3, ChatML, Llama 3 e Decodificação Incremental UTF-8
--]]

local ffi = require("ffi")

-- Tentativa de carregar PCRE2 nativo para pre-tokenização ultrarrápida
local pcre2 = nil
pcall(function()
    ffi.cdef[[
        typedef struct pcre2_real_code_8 pcre2_code_8;
        typedef struct pcre2_real_match_data_8 pcre2_match_data_8;
        pcre2_code_8 *pcre2_compile_8(const uint8_t *, size_t, uint32_t, int *, size_t *, void *);
        void pcre2_code_free_8(pcre2_code_8 *);
        pcre2_match_data_8 *pcre2_match_data_create_from_pattern_8(const pcre2_code_8 *, void *);
        void pcre2_match_data_free_8(pcre2_match_data_8 *);
        int pcre2_match_8(const pcre2_code_8 *, const uint8_t *, size_t, size_t, uint32_t, pcre2_match_data_8 *, void *);
        size_t *pcre2_get_ovector_pointer_8(pcre2_match_data_8 *);
    ]]
    pcre2 = ffi.load("libpcre2-8.so.0")
end)

local GNCXBPE = {}
GNCXBPE.__index = GNCXBPE

-- Tipos de token GGUF
local TOKEN_TYPE_NORMAL       = 1
local TOKEN_TYPE_UNKNOWN      = 2
local TOKEN_TYPE_CONTROL      = 3
local TOKEN_TYPE_USER_DEFINED = 4
local TOKEN_TYPE_UNUSED       = 5
local TOKEN_TYPE_BYTE         = 6

-- Função auxiliar: Converte codepoint Unicode para sequência UTF-8 em Lua
local function utf8_chr(cp)
    if cp < 128 then
        return string.char(cp)
    elseif cp < 2048 then
        return string.char(
            192 + math.floor(cp / 64),
            128 + (cp % 64)
        )
    elseif cp < 65536 then
        return string.char(
            224 + math.floor(cp / 4096),
            128 + (math.floor(cp / 64) % 64),
            128 + (cp % 64)
        )
    else
        return string.char(
            240 + math.floor(cp / 262144),
            128 + (math.floor(cp / 4096) % 64),
            128 + (math.floor(cp / 64) % 64),
            128 + (cp % 64)
        )
    end
end

-- Constrói a tabela reversível byte -> caractere GPT-2 / tiktoken
local function build_bytes_to_unicode()
    local bs = {}
    local in_bs = {}

    local function add_range(start_c, end_c)
        for b = start_c, end_c do
            table.insert(bs, b)
            in_bs[b] = true
        end
    end

    add_range(string.byte("!"), string.byte("~"))
    add_range(161, 172)
    add_range(174, 255)

    local cs = {}
    for i, b in ipairs(bs) do
        cs[i] = b
    end

    local n = 0
    for b = 0, 255 do
        if not in_bs[b] then
            table.insert(bs, b)
            table.insert(cs, 256 + n)
            n = n + 1
        end
    end

    local byte_to_char = {}
    local char_to_byte = {}
    for i = 1, #bs do
        local b = bs[i]
        local c_str = utf8_chr(cs[i])
        byte_to_char[b] = c_str
        char_to_byte[c_str] = b
    end

    return byte_to_char, char_to_byte
end

local BYTE_ENCODER, BYTE_DECODER = build_bytes_to_unicode()

function GNCXBPE.new(tokens, merges, token_types, bos_id, eos_id)
    local self = setmetatable({}, GNCXBPE)

    self.id_to_token = tokens
    self.vocab_size = #tokens
    self.token_types = token_types or {}
    self.eos_id = eos_id or 0
    self.bos_id = bos_id or -1

    self.token_to_id = {}
    for idx = 1, #tokens do
        local tok = tokens[idx]
        local tid = idx - 1
        if self.token_to_id[tok] == nil then
            self.token_to_id[tok] = tid
        end
    end

    self.bpe_ranks = {}
    if merges then
        for rank = 1, #merges do
            local m = merges[rank]
            local sp = m:find(" ", 1, true)
            if sp then
                local a = m:sub(1, sp - 1)
                local b = m:sub(sp + 1)
                self.bpe_ranks[a .. "\0" .. b] = rank - 1
            end
        end
    end

    self.special_tokens = {}
    self.control_ids = {}
    for idx = 1, self.vocab_size do
        local tid = idx - 1
        local tt = self.token_types[idx]
        if tt == TOKEN_TYPE_CONTROL or tt == TOKEN_TYPE_USER_DEFINED then
            table.insert(self.special_tokens, self.id_to_token[idx])
        end
        if tt == TOKEN_TYPE_CONTROL then
            self.control_ids[tid] = true
        end
    end

    -- Ordena tokens especiais do mais longo ao mais curto para matching correto
    table.sort(self.special_tokens, function(a, b) return #a > #b end)

    self.is_chatml = (self.token_to_id["<|im_start|>"] ~= nil and self.token_to_id["<|im_end|>"] ~= nil)
    self.has_native_thinking = (self.token_to_id["<think>"] ~= nil and self.token_to_id["</think>"] ~= nil)

    self.token_bytes_cache = {}
    self.bpe_cache = {}

    -- Inicializa regex PCRE2 se disponível
    if pcre2 then
        local pat = "(?i:\\x27s|\\x27t|\\x27re|\\x27ve|\\x27m|\\x27ll|\\x27d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
        local err_code = ffi.new("int[1]")
        local err_off = ffi.new("size_t[1]")
        local PCRE2_UTF = 0x00080000
        local PCRE2_UCP = 0x00020000
        self.pcre_code = pcre2.pcre2_compile_8(pat, #pat, PCRE2_UTF + PCRE2_UCP, err_code, err_off, nil)
        if self.pcre_code ~= nil then
            self.pcre_md = pcre2.pcre2_match_data_create_from_pattern_8(self.pcre_code, nil)
            self.pcre_ovec = pcre2.pcre2_get_ovector_pointer_8(self.pcre_md)
        end
    end

    return self
end

-- Algoritmo BPE para uma palavra mapeada com caracteres GPT-2
function GNCXBPE:_bpe(word)
    local cached = self.bpe_cache[word]
    if cached ~= nil then
        return cached
    end

    local direct = self.token_to_id[word]
    if direct ~= nil then
        self.bpe_cache[word] = { direct }
        return { direct }
    end

    -- Divide a palavra em caracteres UTF-8
    local symbols = {}
    local p = 1
    local len = #word
    while p <= len do
        local b = word:byte(p)
        local clen = 1
        if b >= 240 then clen = 4
        elseif b >= 224 then clen = 3
        elseif b >= 192 then clen = 2
        end
        table.insert(symbols, word:sub(p, p + clen - 1))
        p = p + clen
    end

    local ranks = self.bpe_ranks
    while #symbols > 1 do
        local best_rank = nil
        local best_i = -1
        for i = 1, #symbols - 1 do
            local r = ranks[symbols[i] .. "\0" .. symbols[i + 1]]
            if r ~= nil and (best_rank == nil or r < best_rank) then
                best_rank = r
                best_i = i
            end
        end
        if best_i < 1 then break end

        local a = symbols[best_i]
        local b = symbols[best_i + 1]
        local merged = a .. b
        local new_symbols = {}
        local i = 1
        local n = #symbols
        while i <= n do
            if i < n and symbols[i] == a and symbols[i + 1] == b then
                table.insert(new_symbols, merged)
                i = i + 2
            else
                table.insert(new_symbols, symbols[i])
                i = i + 1
            end
        end
        symbols = new_symbols
    end

    local ids = {}
    for _, s in ipairs(symbols) do
        local tid = self.token_to_id[s]
        if tid ~= nil then
            table.insert(ids, tid)
        else
            -- Fallback caractere a caractere
            local sp = 1
            local slen = #s
            while sp <= slen do
                local sb = s:byte(sp)
                local sclen = (sb >= 240 and 4) or (sb >= 224 and 3) or (sb >= 192 and 2) or 1
                local ch = s:sub(sp, sp + sclen - 1)
                local cid = self.token_to_id[ch]
                if cid ~= nil then table.insert(ids, cid) end
                sp = sp + sclen
            end
        end
    end

    if #self.bpe_cache < 200000 then
        self.bpe_cache[word] = ids
    end
    return ids
end

-- Pre-tokenização e codificação de um segmento de texto puro
function GNCXBPE:_encode_segment(text)
    local ids = {}
    local enc = BYTE_ENCODER

    if self.pcre_code ~= nil then
        local text_len = #text
        local offset = 0
        local ovec = self.pcre_ovec
        local md = self.pcre_md
        local code = self.pcre_code

        while offset < text_len do
            local rc = pcre2.pcre2_match_8(code, text, text_len, offset, 0, md, nil)
            if rc < 1 then
                -- Restante como uma única peça se houver bytes não casados
                local rest = text:sub(offset + 1)
                local mapped = {}
                for i = 1, #rest do
                    table.insert(mapped, enc[rest:byte(i)])
                end
                local piece_ids = self:_bpe(table.concat(mapped))
                for _, id in ipairs(piece_ids) do table.insert(ids, id) end
                break
            end

            local s_pos = tonumber(ovec[0])
            local e_pos = tonumber(ovec[1])
            if e_pos <= s_pos then break end

            local piece = text:sub(s_pos + 1, e_pos)
            local mapped = {}
            for i = 1, #piece do
                table.insert(mapped, enc[piece:byte(i)])
            end
            local piece_ids = self:_bpe(table.concat(mapped))
            for _, id in ipairs(piece_ids) do table.insert(ids, id) end
            offset = e_pos
        end
    else
        -- Fallback simples sem regex (byte a byte)
        local mapped = {}
        for i = 1, #text do
            table.insert(mapped, enc[text:byte(i)])
        end
        local piece_ids = self:_bpe(table.concat(mapped))
        for _, id in ipairs(piece_ids) do table.insert(ids, id) end
    end

    return ids
end

-- Codifica texto completo (incluindo tokens especiais de controle)
function GNCXBPE:encode(text)
    local ids = {}
    local cursor = 1
    local text_len = #text

    while cursor <= text_len do
        -- Procura o próximo token especial mais próximo
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
            -- Nenhum token especial restante; codifica todo o restante
            local rest = text:sub(cursor)
            if #rest > 0 then
                local s_ids = self:_encode_segment(rest)
                for _, id in ipairs(s_ids) do table.insert(ids, id) end
            end
            break
        else
            -- Codifica segmento anterior ao token especial
            if earliest_pos > cursor then
                local segment = text:sub(cursor, earliest_pos - 1)
                local s_ids = self:_encode_segment(segment)
                for _, id in ipairs(s_ids) do table.insert(ids, id) end
            end
            -- Insere o ID do token especial diretamente
            local sp_id = self.token_to_id[matched_special]
            if sp_id ~= nil then
                table.insert(ids, sp_id)
            end
            cursor = earliest_pos + #matched_special
        end
    end

    return ids
end

-- Decodifica 1 token em sequência de bytes brutos (string Lua)
function GNCXBPE:token_bytes(token_id)
    local cached = self.token_bytes_cache[token_id]
    if cached ~= nil then
        return cached
    end

    if token_id < 0 or token_id >= self.vocab_size or self.control_ids[token_id] then
        self.token_bytes_cache[token_id] = ""
        return ""
    end

    local tok = self.id_to_token[token_id + 1]
    if not tok then
        self.token_bytes_cache[token_id] = ""
        return ""
    end

    local ttype = self.token_types[token_id + 1] or TOKEN_TYPE_NORMAL
    if ttype == TOKEN_TYPE_USER_DEFINED then
        self.token_bytes_cache[token_id] = tok
        return tok
    end

    -- Decodificação de caracteres GPT-2 byte a byte
    local out = {}
    local p = 1
    local len = #tok
    local dec = BYTE_DECODER

    while p <= len do
        local b1 = tok:byte(p)
        local ch
        local ch_len = 1
        if b1 < 128 then
            ch = tok:sub(p, p)
            ch_len = 1
        elseif b1 < 224 then
            ch = tok:sub(p, p + 1)
            ch_len = 2
        elseif b1 < 240 then
            ch = tok:sub(p, p + 2)
            ch_len = 3
        else
            ch = tok:sub(p, p + 3)
            ch_len = 4
        end

        local original_byte = dec[ch]
        if original_byte ~= nil then
            table.insert(out, string.char(original_byte))
        else
            table.insert(out, ch)
        end
        p = p + ch_len
    end

    local res = table.concat(out)
    self.token_bytes_cache[token_id] = res
    return res
end

-- Decodifica lista de IDs em string legível
function GNCXBPE:decode(token_ids, skip_special)
    if skip_special == nil then skip_special = true end
    local out = {}
    for _, tid in ipairs(token_ids) do
        if self.control_ids[tid] then
            if not skip_special then
                local tok = self.id_to_token[tid + 1]
                if tok then table.insert(out, tok) end
            end
        else
            table.insert(out, self:token_bytes(tid))
        end
    end
    return table.concat(out)
end

-- Formata prompt no template ChatML (Qwen3)
function GNCXBPE:format_chat(user_prompt, system_prompt, enable_thinking)
    system_prompt = system_prompt or "Responda sempre em português do Brasil de forma clara e direta."
    if enable_thinking == nil then enable_thinking = true end

    local parts = {}
    if system_prompt ~= "" then
        table.insert(parts, "<|im_start|>system\n" .. system_prompt:match("^%s*(.-)%s*$") .. "<|im_end|>\n")
    end
    table.insert(parts, "<|im_start|>user\n" .. user_prompt:match("^%s*(.-)%s*$") .. "<|im_end|>\n")
    table.insert(parts, "<|im_start|>assistant\n")

    if self.has_native_thinking and not enable_thinking then
        table.insert(parts, "<think>\n\n</think>\n\n")
    end

    return table.concat(parts)
end

return GNCXBPE
