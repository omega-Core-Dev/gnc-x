# PROJETO PRODUÇÃO — MOTOR GEOMÉTRICO GNC-X LLM (LUAJIT NATIVO)

Diretório isolado para execução em produção do **Motor Geométrico GNC-X LLM** via **LuaJIT FFI Zero-Overhead**, sem dependências externas (Zero Python, Zero Llama-Server, Zero PyTorch).

---

## 📁 Estrutura Isolada de Produção

```
projeto_producao/
├── iniciar_gncx_lua.sh     # Inicializador principal de produção
├── bin/
│   └── gncx_llm_core.so    # Biblioteca C++20 compilada nativa (-O3 -march=native)
├── lua/
│   ├── gncx_chat.lua       # Interface CLI de chat interativo
│   ├── gncx_streamer.lua   # Pipeline GLA e orquestração de camadas
│   ├── gncx_gguf.lua       # Parser GGUF e mmap Zero-Copy
│   ├── gncx_bpe.lua        # Tokenizador Byte-Level BPE (Qwen3 / ChatML)
│   ├── gncx_spiece.lua     # Tokenizador SentencePiece BPE (Gemma 2B / Gemma 4)
│   └── gncx_agent.lua      # Agente autônomo com chamada de ferramentas
├── models/
│   ├── qwen3-1.7b-q4_k_m.gguf      # Perfil 3: Qwen3 1.7B Alibaba (1.05 GB)
│   ├── gemma-2b-it-q4_0.gguf       # Perfil 1: Gemma 2B Instruct (1.60 GB)
│   ├── gemma4-e2b-it-q4_k.gguf     # Perfil 2: Gemma 4 E2B Omni (6.83 GB)
│   ├── embeddinggemma-2-Q8_0.gguf  # Modelo Multimodal de Embeddings
│   └── mmproj-embeddinggemma-2-Q8_0.gguf # Projetor CLIP Multimodal
├── csrc/
│   ├── gncx_llm_kernels.cpp        # Código-fonte C++20 dos kernels
│   ├── gncx_genome_vm.hpp          # Mapeamento cartesiano centrado na Origem Central
│   └── gncx_geometric_mm.hpp       # GeometricMemoryManager (GMM)
└── compilar_linux.sh       # Script para recompilação caso necessário
```

---

## 🚀 Como Iniciar

Dentro do diretório de produção, execute:

```bash
./iniciar_gncx_lua.sh
```

Ou a partir da pasta raiz:

```bash
cd "projeto produção" && ./iniciar_gncx_lua.sh
```

---

## ⚙️ Modelos Suportados no Chat

1. **Perfil 3 (Padrão):** `Qwen3 1.7B Instruct` (28 Camadas | Q4_K + Q6_K | GQA + SwiGLU | 1.05 GB)
2. **Perfil 1:** `Gemma 2B Instruct` (18 Camadas | Q4_0 + Q8_0 | GeGLU | 1.60 GB)
3. **Perfil 2:** `Gemma 4 E2B Omni` (35 Camadas | Q4_K + Q6_K + PLE | 6.83 GB)

Para alternar de modelo em tempo de execução, digite `/perfis` durante a conversa.

---

## 🎮 Comandos Interativos

| Comando | Função |
| :--- | :--- |
| `/raciocinio [on/off]` | Ativa ou desativa o pensamento interno `<think>` |
| `/tokens <numero>` | Ajusta o limite de tokens da resposta final (ex: `/tokens 1024`) |
| `/sistema <texto>` | Altera a personalidade/instrução do sistema |
| `/perfis` | Alterna entre Gemma 2B, Gemma 4 e Qwen3 |
| `/memoria` | Exibe o consumo de memória RAM do processo em tempo real |
| `/geometria` | Exibe a telemetria do GeometricMemoryManager (GMM) |
| `/salvar [arq.md]` | Exporta a conversa e o raciocínio gerado para Markdown |
| `/sair` | Encerra a sessão liberando todos os tensores da RAM |
