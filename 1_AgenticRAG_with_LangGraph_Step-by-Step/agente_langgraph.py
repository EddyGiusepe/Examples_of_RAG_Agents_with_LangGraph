#! /usr/bin/env python3
"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

Script para AGENTE RAG - Relações com Investidores da VIX
=========================================================
Este script é um agente RAG que usa o banco vetorial Chroma para buscar 
documentos relevantes e responder perguntas sobre Relações com Investidores 
da VIX.

Run
---
uv run agente_langgraph.py
"""
from typing import Literal
from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai.chat_models import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Define the tool for context retrieval
@tool
def retrieve_context(query: str):
    """
    Busca documentos relevantes no banco vetorial Chroma.
    IMPORTANTE: O banco vetorial deve ser criado primeiro com o script ingerir_dados.py
    """
    
    print("🔍 Buscando no banco vetorial...")
    
    # Caminho do banco vetorial persistido
    chroma_db_path = "./chroma_db_ri_vix"
    
    # Verificar se o banco vetorial existe
    if not Path(chroma_db_path).exists():
        error_msg = """
❌ ERRO: Banco vetorial não encontrado!

Para resolver:
1. Execute primeiro o script de ingestão:
   uv run ingerir_dados.py

2. Depois execute este agente novamente:
   uv run agente_langgraph.py
"""
        print(error_msg)
        return error_msg
    
    try:
        # Carregar o banco vetorial existente (SEM recriar embeddings!)
        vectorstore = Chroma(
            collection_name="ri_vix_docs",
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
            persist_directory=chroma_db_path
        )
        
        # Criar retriever com configurações otimizadas
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 6  # Retornar top 6 documentos mais relevantes
            }
        )
        
        # Buscar documentos relevantes
        print(f"🔎 Query: '{query}'")
        results = retriever.invoke(query)
        print(f"✅ {len(results)} documentos relevantes encontrados")
        
        # Retornar o conteúdo dos documentos encontrados
        return "\n\n---\n\n".join([doc.page_content for doc in results])
        
    except Exception as e:
        error_msg = f"❌ Erro ao buscar no banco vetorial: {str(e)}"
        print(error_msg)
        return error_msg

tools = [retrieve_context]
tool_node = ToolNode(tools)

# OpenAI LLM model
model = ChatOpenAI(model="gpt-5-nano", temperature=0).bind_tools(tools)

# Function to decide whether to continue or stop the workflow
def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    # If the LLM makes a tool call, go to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, finish the workflow
    return END

# Function that invokes the model
def call_model(state: MessagesState):
    """
    Função que invoca o modelo LLM com instruções específicas.
    Adiciona um SystemMessage com contexto sobre como responder.
    """
    messages = state['messages']
    
    # Adicionar system message se ainda não existir
    if not messages or not isinstance(messages[0], SystemMessage):
        system_msg = SystemMessage(content="""Você é um assistente especializado em Relações com Investidores da VIX Logística.

INSTRUÇÕES IMPORTANTES:
1. Leia CUIDADOSAMENTE todo o contexto recuperado dos documentos
2. Extraia informações ESPECÍFICAS e DETALHADAS do texto fornecido
3. Sempre responda em português brasileiro (pt-BR) de forma clara e profissional
4. Base suas respostas EXCLUSIVAMENTE no contexto recuperado
5. Seja DIRETO, PRECISO e OBJETIVO nas respostas
6. Se encontrar a informação no contexto, responda com confiança e detalhes
7. Se NÃO encontrar a informação no contexto, diga explicitamente: "Não há informação disponível sobre isso nos documentos fornecidos"
8. Organize respostas longas com marcadores ou numeração quando apropriado
9. Use terminologia técnica adequada para o contexto de Relações com Investidores

FORMATO DE RESPOSTA:
- Para perguntas objetivas: responda diretamente
- Para perguntas complexas: estruture a resposta em tópicos
- Sempre cite informações relevantes do contexto quando disponível

Use o contexto recuperado para responder de forma completa, precisa e profissional.

Ademais, se o usário dar saudações ou despedidas, responda com uma saudação ou despedida apropriada.
""")
        messages = [system_msg] + messages
    
    response = model.invoke(messages)
    return {"messages": [response]}

# Define the workflow with LangGraph
workflow = StateGraph(MessagesState)

# Add nodes to the graph
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Connect nodes
workflow.add_edge(START, "agent")  # Initial entry
workflow.add_conditional_edges("agent", should_continue)  # Decision after the "agent" node
workflow.add_edge("tools", "agent")  # Cycle between tools and agent

# Configure memory to persist the state
checkpointer = MemorySaver()

# Compile the graph into a LangChain Runnable application
app = workflow.compile(checkpointer=checkpointer)

# ========================================
# MODO INTERATIVO - Chat com o Agente
# ========================================

def chat_interativo():
    """
    Função para interagir com o agente de forma contínua.
    """
    print("\n" + "=" * 60)
    print("🤖 AGENTE RAG - Relações com Investidores da VIX")
    print("=" * 60)
    print("📚 Sistema pronto para responder suas perguntas!")
    print("💡 Dica: Digite 'sair' ou 'exit' para encerrar")
    print("=" * 60 + "\n")
    
    # ID da thread (sessão) para manter o histórico da conversa
    thread_id = 42
    
    while True:
        try:
            # Solicitar pergunta do usuário
            pergunta = input("👤 Você: ").strip()
            
            # Verificar se o usuário quer sair
            if pergunta.lower() in ['sair', 'exit', 'quit', 'q']:
                print("\n👋 Encerrando... Até logo!")
                break
            
            # Ignorar entradas vazias
            if not pergunta:
                continue
            
            print("\n🤔 Processando...\n")
            
            # Executar o workflow com a pergunta do usuário
            final_state = app.invoke(
                {"messages": [HumanMessage(content=pergunta)]},
                config={"configurable": {"thread_id": thread_id}}
            )
            
            # Mostrar a resposta do agente
            resposta = final_state["messages"][-1].content
            print(f"🤖 Agente: {resposta}\n")
            print("-" * 60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrompido pelo usuário. Até logo!")
            break
        except Exception as e:
            print(f"\n❌ Erro: {str(e)}\n")
            print("💡 Tente fazer outra pergunta.\n")


# Iniciar o chat interativo
if __name__ == "__main__":
    chat_interativo()

