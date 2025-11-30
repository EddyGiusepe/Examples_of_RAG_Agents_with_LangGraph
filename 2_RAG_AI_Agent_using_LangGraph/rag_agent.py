#!/usr/bin/env python3
"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

RAG Agent usando LangGraph
===========================
Este script implementa um agente RAG (Retrieval-Augmented Generation) 
usando o framework LangGraph para orquestração.

O agente:
1. Recebe perguntas do usuário
2. Busca contexto relevante no banco vetorial
3. Gera respostas baseadas no contexto encontrado
4. Mantém memória da conversação

Arquitetura:
- StateGraph: Define o fluxo de execução
- Nós: agent_node (decide ações) e tool_node (executa ferramentas)
- Arestas condicionais: Roteamento baseado em decisões do agente
- MemorySaver: Mantém histórico da conversação

Uso:
----
uv run rag_agent.py
"""

import sys
from typing import Literal


from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool


from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode


from config import (
    VECTOR_DB_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    SYSTEM_PROMPT,
    RETRIEVAL_K,
)


# ============================================================================
# Definição de Ferramentas (Tools)
# ============================================================================

@tool
def retrieve_context(query: str) -> str:
    """
    Busca documentos relevantes no banco vetorial Chroma.
    
    Esta ferramenta é chamada pelo agente quando ele precisa de contexto
    adicional para responder uma pergunta do usuário.
    
    Args:
        query: A pergunta ou tópico para buscar documentos relevantes
        
    Returns:
        String contendo os documentos relevantes encontrados
    """
    print(f"\n Buscando contexto relevante para: '{query}'...")
    
    # Verifica se o banco vetorial existe
    if not VECTOR_DB_DIR.exists():
        error_msg = f"""
            ❌ ERRO: Banco vetorial não encontrado em {VECTOR_DB_DIR}

            💡 SOLUÇÃO:
            1. Execute primeiro o script de ingestão:
            uv run ingest_data.py

            2. Adicione seus documentos na pasta 'data/'
            3. Execute o script de ingestão novamente
            4. Depois execute este agente
            """
        print(error_msg)
        return error_msg
    
    try:
        # Cria o embedding model
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        
        # Carrega o banco vetorial existente
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(VECTOR_DB_DIR),
        )
        
        # Busca documentos similares
        results = vector_store.similarity_search(query, k=RETRIEVAL_K)
        
        if not results:
            print("   ⚠️  Nenhum documento relevante encontrado.")
            return "Nenhum documento relevante encontrado para esta pergunta."
        
        # Formata os resultados
        context_parts = []
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get('source', 'Unknown')
            content = doc.page_content.strip()
            context_parts.append(f"[Documento {i} - Fonte: {source}]\n{content}")
        
        context = "\n\n".join(context_parts)
        
        print(f"   ✅ Encontrados {len(results)} documento(s) relevante(s)\n")
        
        return context
        
    except Exception as e:
        error_msg = f"❌ Erro ao buscar contexto: {str(e)}"
        print(error_msg)
        return error_msg


# Lista de ferramentas disponíveis para o agente
tools = [retrieve_context]


# ============================================================================
# Criação do Modelo LLM com Ferramentas
# ============================================================================

def create_llm_with_tools():
    """
    Cria e configura o modelo LLM com as ferramentas disponíveis.
    
    Returns:
        Modelo LLM configurado com ferramentas
    """
    # Cria o modelo base
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
    )
    
    # Vincula as ferramentas ao modelo
    # Isso permite que o modelo possa "chamar" as ferramentas
    llm_with_tools = llm.bind_tools(tools)
    
    return llm_with_tools


# ============================================================================
# Definição dos Nós do Grafo
# ============================================================================

def agent_node(state: MessagesState):
    """
    Nó do agente que processa mensagens e decide quais ações tomar.
    
    Este nó:
    1. Recebe o estado atual (incluindo histórico de mensagens)
    2. Envia as mensagens para o LLM
    3. O LLM decide se precisa chamar ferramentas ou pode responder diretamente
    4. Retorna a resposta do LLM (que pode incluir chamadas de ferramentas)
    
    Args:
        state: Estado atual contendo as mensagens
        
    Returns:
        Dicionário com as mensagens atualizadas
    """
    # Cria o LLM com ferramentas
    llm_with_tools = create_llm_with_tools()
    
    # Adiciona o system prompt se for a primeira mensagem
    messages = state["messages"]
    if not any(isinstance(msg, SystemMessage) for msg in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    
    # Invoca o modelo
    response = llm_with_tools.invoke(messages)
    
    # Retorna a resposta (MessagesState automaticamente adiciona à lista)
    return {"messages": [response]}


# O ToolNode já está pré-construído no LangGraph
# Ele automaticamente executa as ferramentas solicitadas pelo agente
tool_node = ToolNode(tools)


# ============================================================================
# Função de Roteamento (Arestas Condicionais)
# ============================================================================

def should_continue(state: MessagesState) -> Literal["tools", "end"]:
    """
    Determina se o agente deve continuar chamando ferramentas ou terminar.
    
    Esta é uma ARESTA CONDICIONAL que decide o próximo passo baseado no
    estado atual.
    
    Lógica:
    - Se a última mensagem contém chamadas de ferramentas (tool_calls),
      retorna "tools" para executar as ferramentas
    - Caso contrário, retorna "end" para finalizar
    
    Args:
        state: Estado atual contendo as mensagens
        
    Returns:
        "tools" para continuar ou "end" para finalizar
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # Se o LLM fez chamadas de ferramentas, vai para o nó de ferramentas
    if last_message.tool_calls:
        return "tools"
    
    # Caso contrário, termina
    return "end"


# ============================================================================
# Construção do Grafo
# ============================================================================

def create_graph():
    """
    Constrói o grafo do agente RAG.
    
    Estrutura do grafo:
    
        START
          ↓
      agent_node ←─┐
          ↓        │
      [decisão]    │
       ↙    ↘     │
    tools   END    │
      ↓            │
      └────────────┘
    
    Fluxo:
    1. START → agent_node: Processa a mensagem do usuário
    2. agent_node → decisão: Verifica se precisa chamar ferramentas
       - Se sim: vai para tool_node
       - Se não: vai para END
    3. tool_node → agent_node: Após executar ferramentas, volta para o agente
    4. Repete até o agente decidir terminar
    
    Returns:
        Grafo compilado e pronto para uso
    """
    # Cria um novo grafo com MessagesState
    workflow = StateGraph(MessagesState)
    
    # Adiciona os nós ao grafo
    workflow.add_node("agent", agent_node)  # Nó do agente
    workflow.add_node("tools", tool_node)   # Nó de ferramentas
    
    # Define a aresta de entrada (START → agent)
    workflow.add_edge(START, "agent")
    
    # Adiciona aresta condicional do agente
    # Decide se vai para ferramentas ou termina
    workflow.add_conditional_edges(
        "agent",              # Nó de origem
        should_continue,      # Função de decisão
        {
            "tools": "tools", # Se retornar "tools", vai para o nó "tools"
            "end": END        # Se retornar "end", termina
        }
    )
    
    # Adiciona aresta fixa: tools → agent
    # Após executar ferramentas, sempre volta para o agente
    workflow.add_edge("tools", "agent")
    
    # Compila o grafo com memória
    # MemorySaver permite manter o histórico da conversação
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
    
    return graph


# ============================================================================
# Interface de Chat
# ============================================================================

def print_welcome():
    """Exibe mensagem de boas-vindas"""
    print("\n" + "="*70)
    print("🤖 RAG AGENT COM LANGGRAPH")
    print("="*70)
    print("\n💡 Dicas:")
    print("   - Faça perguntas sobre os documentos no banco vetorial")
    print("   - Digite 'sair' ou 'exit' para encerrar")
    print("   - Digite 'limpar' para iniciar uma nova conversa")
    print("\n" + "="*70 + "\n")


def chat_loop(graph):
    """
    Loop principal do chat interativo.
    
    Args:
        graph: Grafo compilado do agente
    """
    print_welcome()
    
    # ID da thread para manter a memória da conversa
    thread_id = "user_session_1"
    config = {"configurable": {"thread_id": thread_id}}
    
    while True:
        try:
            # Lê input do usuário
            user_input = input("👤 Você: ").strip()
            
            # Comandos especiais
            if user_input.lower() in ['sair', 'exit', 'quit']:
                print("\n👋 Até logo!\n")
                break
            
            if user_input.lower() == 'limpar':
                # Cria uma nova thread para limpar o histórico
                import uuid
                thread_id = f"user_session_{uuid.uuid4()}"
                config = {"configurable": {"thread_id": thread_id}}
                print("\n🔄 Conversa limpa! Iniciando nova sessão.\n")
                continue
            
            if not user_input:
                continue
            
            # Cria a mensagem do usuário
            input_message = {"messages": [HumanMessage(content=user_input)]}
            
            print("\n🤖 Agente: ", end="", flush=True)
            
            # Invoca o grafo com streaming
            # O grafo processa a mensagem e mantém o histórico
            response_text = ""
            for event in graph.stream(input_message, config, stream_mode="values"):
                # Pega a última mensagem
                last_message = event["messages"][-1]
                
                # Se for uma mensagem de IA (não ferramenta), exibe
                if hasattr(last_message, 'content') and last_message.content:
                    # Evita duplicação mostrando apenas a mensagem final
                    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
                        response_text = last_message.content
            
            # Exibe a resposta final
            print(response_text)
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Até logo!\n")
            break
        except Exception as e:
            print(f"\n❌ Erro: {e}\n")


# ============================================================================
# Função Principal
# ============================================================================

def main():
    """
    Função principal que inicia o agente RAG.
    """
    # Verifica se o banco vetorial existe
    if not VECTOR_DB_DIR.exists():
        print("\n" + "="*70)
        print("❌ BANCO VETORIAL NÃO ENCONTRADO")
        print("="*70)
        print(f"\n💡 O banco vetorial não foi encontrado em: {VECTOR_DB_DIR}")
        print("\nPara resolver:")
        print("  1. Execute o script de ingestão primeiro:")
        print("     uv run ingest_data.py")
        print("\n  2. Adicione seus documentos na pasta 'data/'")
        print("\n  3. Execute este script novamente")
        print("\n" + "="*70 + "\n")
        sys.exit(1)
    
    # Cria o grafo
    print("\n⚙️  Inicializando o agente RAG...")
    graph = create_graph()
    print("✅ Agente inicializado com sucesso!")
    
    # Inicia o chat
    chat_loop(graph)


# ============================================================================
# Ponto de Entrada
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Programa interrompido pelo usuário.\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

