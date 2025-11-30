# <h1 align="center"><font color="red">RAG AI Agent usando LangGraph</font></h1>

<font color="pink">Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro</font>

Este projeto demonstra como construir um **Agente RAG (Retrieval-Augmented Generation)** usando o framework **LangGraph**. O objetivo é criar um exemplo didático e bem explicado que ilustra os conceitos fundamentais do LangGraph na prática.

## 📚 O que é LangGraph?

**LangGraph** é uma biblioteca Python desenvolvida pela LangChain para construir aplicações de agentes de IA com **lógica complexa e fluxos de decisão customizáveis**. Diferentemente de frameworks que fornecem uma "caixa preta" de agentes, o LangGraph oferece **controle total** sobre o comportamento do agente através de uma arquitetura baseada em grafos.

### Por que usar LangGraph?

- ✅ **Controle total**: Defina exatamente como seu agente deve se comportar
- ✅ **Fluxos complexos**: Suporte para múltiplos agentes, decisões condicionais e loops
- ✅ **Memória integrada**: Sistema de persistência de estado e memória de conversação
- ✅ **Human-in-the-loop**: Fácil integração de aprovações humanas no fluxo
- ✅ **Observabilidade**: Visualize e depure o fluxo de execução do agente
- ✅ **Open Source**: Licença MIT, totalmente gratuito

## 🔑 Conceitos Fundamentais

### 1. Grafos (Graphs)

Um **grafo** no LangGraph é uma estrutura que define o **fluxo de execução** do seu agente. Ele é composto por:

- **Nós (Nodes)**: Representam operações, ações ou agentes individuais
- **Arestas (Edges)**: Definem as transições e conexões entre os nós

```
┌─────────────┐
│    START    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  agent_node │◄─┐
└──────┬──────┘  │
       │         │
       ▼         │
   ┌───────┐     │
   │ route │     │
   └───┬───┘     │
       │         │
    ┌──┴──┐      │
    │     │      │
    ▼     ▼      │
┌────┐  ┌────┐   │
│tool│  │END │   │
└─┬──┘  └────┘   │
  │              │
  └──────────────┘
```

### 2. Nós (Nodes)

Cada **nó** é uma função que:
- Recebe o **estado** atual
- Executa alguma operação (chamar LLM, executar ferramenta, etc.)
- Retorna atualizações para o **estado**

**Exemplo:**

```python
def agent_node(state: MessagesState):
    """Nó que processa a mensagem do usuário e decide a próxima ação"""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}
```

### 3. Arestas (Edges)

Existem dois tipos de arestas:

#### a) Arestas Diretas (Fixed Edges)
Transições fixas de um nó para outro:

```python
graph.add_edge("tool_node", "agent_node")
# Sempre vai de tool_node para agent_node
```

#### b) Arestas Condicionais (Conditional Edges)
Lógica de roteamento baseada no estado:

```python
def should_continue(state: MessagesState) -> Literal["tools", "end"]:
    """Decide se deve chamar ferramentas ou terminar"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "end"

graph.add_conditional_edges(
    "agent_node",
    should_continue,
    {"tools": "tool_node", "end": END}
)
```

### 4. Estado (State)

O **estado** é um dicionário compartilhado entre todos os nós do grafo. Ele persiste informações ao longo da execução.

**MessagesState**: Estado padrão para conversações

```python
from langgraph.graph import MessagesState

class MessagesState(TypedDict):
    messages: Annotated[list, add_messages]
```

A anotação `Annotated[list, add_messages]` define como atualizar o estado:
- `add_messages`: Adiciona novas mensagens à lista existente (não substitui)

### 5. Memória de Curto Prazo

LangGraph oferece **checkpointing** integrado para manter o contexto da conversação:

```python
from langgraph.checkpoint.memory import MemorySaver

# Cria um sistema de memória
memory = MemorySaver()

# Compila o grafo com memória
graph = workflow.compile(checkpointer=memory)

# Usa um thread_id para identificar a sessão
config = {"configurable": {"thread_id": "user_123"}}
response = graph.invoke(input, config=config)
```

**Benefícios:**
- ✅ Mantém histórico de mensagens entre invocações
- ✅ Permite "time-travel" (voltar a estados anteriores)
- ✅ Suporta múltiplas sessões simultâneas (thread_id diferente)

## 🏗️ Arquitetura do RAG Agent

Este projeto implementa um agente RAG com a seguinte arquitetura:

```
1. Usuário faz uma pergunta
   ↓
2. Agent Node decide se precisa buscar contexto
   ↓
3. Tool Node busca documentos relevantes no Chroma
   ↓
4. Agent Node gera resposta usando contexto + pergunta
   ↓
5. Resposta é retornada ao usuário
```

### Componentes:

1. **Banco Vetorial (Chroma)**: Armazena embeddings dos documentos
2. **Tool retrieve_context**: Busca documentos relevantes
3. **LLM (OpenAI GPT)**: Processa perguntas e gera respostas
4. **StateGraph**: Orquestra o fluxo do agente
5. **MemorySaver**: Mantém histórico da conversação

## 📁 Estrutura do Projeto

```
2_RAG_AI_Agent_using_LangGraph/
├── README.md                  # Este arquivo
├── CONCEPTS.md               # Aprofundamento teórico
├── .env.example              # Template de configuração
├── config.py                 # Configurações do projeto
├── ingest_data.py           # Script para ingestão de documentos
├── rag_agent.py             # Agente RAG principal
└── data/                    # Seus documentos (PDFs, TXT, etc.)
```

## 🚀 Como Usar

### 1. Instalar Dependências

```bash
# Atualizar dependências do projeto
uv sync
```

### 2. Configurar Variáveis de Ambiente

Copie o arquivo `.env.example` para `.env` e adicione suas API keys:

```bash
cp .env.example .env
```

Edite o arquivo `.env`:

```env
OPENAI_API_KEY=sua-chave-aqui
```

### 3. Preparar Dados

Coloque seus documentos na pasta `data/`:

```bash
mkdir -p data
# Copie seus PDFs, TXTs, etc. para a pasta data/
```

### 4. Ingerir Documentos

Execute o script de ingestão para criar o banco vetorial:

```bash
uv run ingest_data.py
```

Este script irá:
- Carregar documentos da pasta `data/`
- Dividir em chunks
- Criar embeddings
- Armazenar no banco vetorial Chroma

### 5. Executar o Agente

Inicie o agente RAG:

```bash
uv run rag_agent.py
```

Agora você pode fazer perguntas sobre seus documentos!

## 💡 Exemplos de Uso

```
Usuário: Qual é o tema principal dos documentos?
Agente: [Busca contexto] [Analisa] [Responde com base nos documentos]

Usuário: Me dê mais detalhes sobre X
Agente: [Usa memória da conversa anterior] [Busca mais contexto] [Responde]
```

## 🎯 Diferenças: LangChain vs LangGraph

| Aspecto | LangChain | LangGraph |
|---------|-----------|-----------|
| **Controle** | Alto nível, mais abstrato | Baixo nível, controle total |
| **Fluxos** | Lineares (chains) | Grafos complexos com ciclos |
| **Memória** | Implementação manual | Checkpointing integrado |
| **Debugging** | Limitado | Visualização completa do grafo |
| **Uso** | Casos simples e rápidos | Agentes complexos e production-ready |

## 📖 Recursos e Referências

### Documentação Oficial
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)

### Tutoriais
- [Tutorial FutureSmart AI](https://blog.futuresmart.ai/langgraph-rag-agent-tutorial-basics-to-advanced-multi-agent-ai-chatbot)
- [LangChain Academy - Introduction to LangGraph](https://www.langchain.com/langgraph)

### Código Original
- [Notebook de Referência](https://github.com/PradipNichite/Youtube-Tutorials/blob/main/RAG_AI_Agent_using_LangGraph.ipynb)

### Artigos em Português
- [LangGraph para Construção de Agentes de IA](https://blog.dsacademy.com.br/langgraph-para-construcao-de-agentes-de-ia-arquitetura-orquestracao-e-casos-de-uso/)

## 🔍 Para Aprender Mais

Consulte o arquivo [`CONCEPTS.md`](CONCEPTS.md) para:

- Comparação detalhada: ``StateGraph`` vs ``MessageGraph``
- Padrões avançados de ``Human-in-the-loop``
- Estratégias de persistência e ``checkpointing``
- Exemplos de arquiteturas ``multi-agente``
- Melhores práticas para produção

## 📝 Notas

- Este é um projeto educacional para aprender LangGraph
- O código contém comentários detalhados em português
- Sinta-se livre para modificar e experimentar!

---

**Autor**: ``Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro``  
**Data**: ``Novembro 2025``
