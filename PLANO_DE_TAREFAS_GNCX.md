# 📋 Plano de Tarefas: Upgrade GNC-X v2.6 e Módulo EmbeddingGemma

Este documento define o roteiro técnico de execução dividido em duas fases estratégicas:
- **Fase 1:** Correções arquiteturais, estabilidade de memória, vetorização SSE e destravamento de concorrência (Upgrade v2.6).
- **Fase 2:** Implementação do módulo de pré-inferência e busca semântica com o modelo compacto `embeddinggemma-2-Q8_0.gguf`.

---

## 🏗️ FASE 1: Alicerce, Estabilidade e Otimização do Core (Upgrade v2.6)

### Tarefa 1.1: Alocador HugePage Slab Compartilhado de 4 MB
- **Arquivo Alvo:** [`csrc/gncx_llm_kernels.cpp`](file:///home/oriebir/Documentos/projeto_producao/csrc/gncx_llm_kernels.cpp)
- **Status:** ✅ Concluído
- **Subtarefas:**
  - [x] Implementar a estrutura `GNCXHugePageSlab` com alinhamento a 2 MB (`HUGEPAGE_2MB`) e `MADV_HUGEPAGE`.
  - [x] Atualizar a struct `GLARamBufferHandle` para incluir ponteiro `shared_slab` e ponteiros de mapeamento.
  - [x] Refatorar `gncx_ram_alloc_colored`: buffers `<= 1 MB` passam a ser fatiados dentro da arena de 4 MB mantendo os offsets de cor DDR3 (`bank_slot` bits 16..14 e `l1_slot` bits 11..6).
  - [x] Atualizar `gncx_ram_free` com liberação atômica/referenciada (`ref_count`) do Slab HugePage.
- **Critério de Sucesso:** 11 scratch buffers alocados com `gncx_ram_alloc_colored` residem dentro do mesmo bloco contíguo de 4 MB com correspondência exata nos seletores de banco DDR3 e sets L1. (Validado: span de 882.3 KB < 1 MB).

---

### Tarefa 1.2: Destravar GNCXThreadPool e Paralelizar Atenção MQA/GQA
- **Arquivo Alvo:** [`csrc/gncx_llm_kernels.cpp`](file:///home/oriebir/Documentos/projeto_producao/csrc/gncx_llm_kernels.cpp)
- **Status:** ✅ Concluído
- **Subtarefas:**
  - [x] Ajustar `GNCXThreadPool::parallel_for` para aceitar `int32_t min_grain = 128` e usar `std::remove_reference_t<Fn>` no contexto de despacho de threads (eliminar erro de referência em lambdas).
  - [x] Eliminar `static std::vector<float> scores;` em `gncx_mqa_attention_step_ext` e adotar `static thread_local std::vector<float> tl_scores` (eliminar *data race* concorrente).
  - [x] Vetorizar com SSE (`_mm_loadu_ps`, `_mm_mul_ps`, `_mm_add_ps`) o produto escalar $Q \cdot K^T$ e acumulação $w \cdot V$.
  - [x] Despachar cabeças de atenção via `get_global_pool().parallel_for(n_heads, compute_heads, 2)` quando `win_len >= 8 && n_heads >= 2`.
  - [x] Paralelizar as ativações `gncx_geglu_activation` e `gncx_swiglu_activation` com granulação de 2048 elementos.
- **Critério de Sucesso:** Execução paralela sem colisões de memória entre threads em CPUs multicore.

---

### Tarefa 1.3: Cache RoPE de 2 Slots e Fast-Path no RMSNorm
- **Arquivo Alvo:** [`csrc/gncx_llm_kernels.cpp`](file:///home/oriebir/Documentos/projeto_producao/csrc/gncx_llm_kernels.cpp)
- **Status:** ✅ Concluído
- **Subtarefas:**
  - [x] Criar estrutura `RoPECacheSlot` alinhada a 64 bytes com `static thread_local RoPECacheSlot tl_rope_cache[2];` (slot 0: Global / slot 1: SWA).
  - [x] Reutilizar tabelas de `cos` e `sin` para a mesma posição `pos`, evitando recalculá-las nas 28 ou 35 camadas da rede.
  - [x] Vetorizar rotação do RoPE com SSE2 em blocos de 4 floats.
  - [x] Vetorizar soma de quadrados e normalização em `gncx_rmsnorm_and_encode_4phase`.
  - [x] No `gncx_rmsnorm_and_encode_4phase`, se `out_tokens == nullptr`, executar apenas o caminho ultra-rápido de normalização fp32 sem crivagem 4-fases desnecessária.
- **Critério de Sucesso:** Eliminação de 100% dos cálculos repetidos de `std::pow/std::cos/std::sin` durante a passagem pelas camadas de um mesmo token; aceleração de 5.04x no Fast-Path do RMSNorm.

---

### Tarefa 1.4: Primitivas de Treino LoRA + MeZO e Scanner 3D do Equador
- **Arquivo Alvo:** [`csrc/gncx_llm_kernels.cpp`](file:///home/oriebir/Documentos/projeto_producao/csrc/gncx_llm_kernels.cpp)
- **Status:** ✅ Concluído
- **Subtarefas:**
  - [x] Implementar gerador pseudoaleatório de registrador `splitmix64_next`.
  - [x] Implementar e exportar funções LoRA: `gncx_lora_init_adapter`, `gncx_lora_forward_add`, `gncx_lora_backward_step`.
  - [x] Implementar funções de otimização MeZO: `gncx_mezo_perturb_params`, `gncx_mezo_update_params` (com gradient clipping `[-10, 10]`) e `gncx_cross_entropy_loss`.
  - [x] Exportar struct `GNCXLayer3DPoint` e função `gncx_project_layer_3d_point` centrada no Equador (`Y = 0.0f` na camada mediana).
- **Critério de Sucesso:** Convergência verificável da loss em CPU sem necessidade de GPU ou armazenamento massivo de gradientes (LoRA loss < 0.001 em 40 passos; MeZO bit-exact diff < 1e-6 e redução monotônica).

---

### Tarefa 1.5: Atualização do Pipeline LuaJIT e Prefill em Lote
- **Arquivo Alvo:** [`lua/gncx_streamer.lua`](file:///home/oriebir/Documentos/projeto_producao/lua/gncx_streamer.lua)
- **Status:** ✅ Concluído
- **Subtarefas:**
  - [x] Atualizar o bloco `ffi.cdef` para incluir as novas assinaturas C (`GNCXLayer3DPoint`, `gncx_lora_*`, `gncx_mezo_*`, `gncx_cross_entropy_loss`, `gncx_project_layer_3d_point`).
  - [x] No método `prefill(token_ids)`, remover o laço sequencial token-a-token quando `self.ple_dim > 0` (Gemma 4).
  - [x] Implementar projeção em lote (`_matbatch`) para tensores PLE (`b_ple_tok`, `b_ple_proj`, `b_ple_gate`).
  - [x] Passar `nil` para `out_tokens` nas chamadas de `gncx_rmsnorm_and_encode_4phase` durante o `prefill` para acionar o fast-path.
  - [x] Criar o método `GNCXStreamer:scan_layer_3d_point(l_idx)`.
- **Critério de Sucesso:** Prefill executado em lote contíguo com uma única leitura de pesos da RAM.

---

### Tarefa 1.6: Compilação e Bateria de Verificação Automatizada
- **Arquivos:** [`compilar_linux.sh`](file:///home/oriebir/Documentos/projeto_producao/compilar_linux.sh), `testes_v26.lua` e `verify_gncx_fixes.py`
- **Status:** ✅ Concluído
- **Subtarefas:**
  - [x] Recompilar biblioteca nativa `bin/gncx_llm_core.so` com flags `-O3 -std=c++20 -march=native`.
  - [x] Executar testes de alocador de HugePages, concorrência, RoPE cache, fast-path RMSNorm e projeção 3D.
- **Critério de Sucesso:** Compilação limpa sem warnings de tipo e 100% dos testes passando em Python e LuaJIT.

---

## 🧠 FASE 2: Módulo de Pré-Inferência com EmbeddingGemma 2

### Tarefa 2.1: Leitor e Mapeador do Modelo EmbeddingGemma
- **Arquivo Alvo:** `lua/gncx_embedding.lua` e `models/embeddinggemma-2-Q8_0.gguf`
- **Status:** ✅ Concluído
- **Subtarefas:**
  - [x] Criar classe `GNCXEmbedding` para mapeamento mmap zero-copy de `embeddinggemma-2-Q8_0.gguf` (296 MB).
  - [x] Inicializar scratch buffers usando o `gncx_ram_alloc_colored` otimizado da Fase 1.
  - [x] Integrar tokenizador SentencePiece correspondente ao vocabulário do Gemma sem duplicação de `<bos>`.
- **Critério de Sucesso:** Carregamento do modelo em RAM consumindo < 350 MB com tempo de carregamento < 500 ms.

---

### Tarefa 2.2: Pipeline de Extração de Vetores e Similaridade
- **Arquivo Alvo:** `lua/gncx_embedding.lua` e `csrc/gncx_llm_kernels.cpp`
- **Status:** ✅ Concluído
- **Subtarefas:**
  - [x] Implementar extração do vetor representativo da sequência (Mean Pooling + projeção linear `output.weight` 512->768).
  - [x] Implementar kernel SIMD SSE2 `gncx_cosine_similarity` e `gncx_bidirectional_attention_gqa` em $O(D)$ direto na CPU.
  - [x] Normalização $L_2$ estrita dos vetores gerados ($\|v\|_2 = 1.000000$).
- **Critério de Sucesso:** Geração de embedding para textos curtos com similaridade angular discriminativa.

---

### Tarefa 2.3: Roteamento de Ações no Agente Autônomo
- **Arquivo Alvo:** [`lua/gncx_agent.lua`](file:///home/oriebir/Documentos/projeto_producao/lua/gncx_agent.lua)
- **Status:** ✅ Concluído
- **Subtarefas:**
  - [x] Vetorizar as descrições das ferramentas disponíveis (`QUICK_ACTIONS`: navegador, arquivos, terminal, status, compilação).
  - [x] Ao receber comando do usuário, calcular similaridade entre a entrada e as ferramentas antes de acionar o LLM.
  - [x] Disparar chamadas determinísticas de ferramenta com tempo de decisão < 5 ms usando cache pré-vetorizado.
- **Critério de Sucesso:** Despacho instantâneo de ferramentas sem gastar tokens de raciocínio no LLM principal.

---

### Tarefa 2.4: Filtro de Contexto Semântico (RAG Local)
- **Arquivo Alvo:** [`lua/gncx_chat.lua`](file:///home/oriebir/Documentos/projeto_producao/lua/gncx_chat.lua)
- **Status:** ✅ Concluído
- **Subtarefas:**
  - [x] Fatiar histórico de conversa e documentos de memória em blocos semânticos (`GNCXSemanticRAG`).
  - [x] Recuperar apenas os top-$K$ fragmentos mais relevantes com base na proximidade angular de cosseno.
  - [x] Montar prompt compacto contendo apenas os dados essenciais para o prefill do Qwen3 / Gemma 2B.
- **Critério de Sucesso:** Redução de 70% ou mais no volume de tokens inseridos no prefill do modelo principal (80.14% atingido em teste).

---

## 🔮 FASE 3: Mecanismo de Raciocínio sobre Contexto e Aprendizado Contínuo (Próximo Objetivo)

> **Status:** 🎯 **EM PLANEJAMENTO E ARQUITETURA CONCEITUAL**

### Tarefa 3.1: Mecanismo Proprietário de Raciocínio sobre Contexto Existente
- **Objetivo:** Substituir a operação *stateless* (sem memória) por uma arquitetura inovadora de raciocínio contextual persistente baseada no insight proprietário do projeto.
- **Subtarefas:**
  - [ ] Estruturação formal do algoritmo e topologia de retenção e síntese de contexto.
  - [ ] Integração nativa do mecanismo com o `GNCXStreamer` e o alocador de memória geométrica (`GMM`).
  - [ ] Validação de raciocínio contínuo multi-turno mantendo consumo de RAM < 2.0 GB.

### Tarefa 3.2: Acoplamento do Aprendizado Contínuo Online (LoRA + MeZO no Chat)
- **Objetivo:** Conectar as primitivas analíticas existentes em C++20 (`gncx_lora_train_step_cpu` e `gncx_mezo_step`) diretamente ao loop interativo do chat para adaptação em tempo real.
- **Subtarefas:**
  - [ ] Ativação do otimizador MeZO de ordem zero (sem backpropagation) para ajuste leve de pesos durante o uso.
  - [ ] Atualização incremental de adaptadores LoRA diretamente em CPU legada.
  - [ ] Persistência dos estados aprendidos em disco sem modificar os pesos brutos originais do GGUF.
