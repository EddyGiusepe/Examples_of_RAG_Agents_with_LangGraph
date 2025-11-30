#!/usr/bin/env python3
"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

Configurações do Projeto RAG Agent com LangGraph
=================================================
Este módulo centraliza todas as configurações do projeto, incluindo:
- Carregamento de variáveis de ambiente
- Configurações do modelo LLM
- Configurações de embeddings
- Parâmetros do banco vetorial
- Configurações de chunking de documentos
"""

import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# ============================================================================
# Carregamento de Variáveis de Ambiente
# ============================================================================

# Tenta encontrar e carregar o arquivo .env
_ = load_dotenv(find_dotenv())

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Validação básica
if not OPENAI_API_KEY:
    raise ValueError(
        "❌ OPENAI_API_KEY não encontrada!\n"
        "Por favor, crie um arquivo .env com sua chave da OpenAI.\n"
        "Veja o arquivo .env.example para referência."
    )

# ============================================================================
# Configurações do Projeto
# ============================================================================

# Diretório raiz do projeto (2_RAG_AI_Agent_using_LangGraph/)
PROJECT_ROOT = Path(__file__).parent

# Diretório para armazenar dados
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# Diretório para o banco vetorial
VECTOR_DB_DIR = PROJECT_ROOT / "chroma_db"

# ============================================================================
# Configurações do Modelo LLM
# ============================================================================

# Modelo da OpenAI a ser usado
LLM_MODEL = "gpt-5-nano"  # Mais rápido e econômico
# LLM_MODEL = "gpt-4o"  # Mais poderoso, mas mais caro

# Temperatura do modelo (0.0 = mais determinístico, 1.0 = mais criativo)
LLM_TEMPERATURE = 0.0

# Número máximo de tokens na resposta
LLM_MAX_TOKENS = 1000

# ============================================================================
# Configurações de Embeddings
# ============================================================================

# Modelo de embeddings da OpenAI
EMBEDDING_MODEL = "text-embedding-3-small"  # Mais rápido e barato
# EMBEDDING_MODEL = "text-embedding-3-large"  # Maior qualidade

# Dimensão dos embeddings (apenas para text-embedding-3-*)
# Valores menores são mais rápidos, valores maiores têm melhor qualidade
EMBEDDING_DIMENSION = 1536  # Padrão do text-embedding-3-small

# ============================================================================
# Configurações do Vector Store (Chroma)
# ============================================================================

# Nome da coleção no Chroma
COLLECTION_NAME = "rag_documents"

# Número de documentos a retornar em cada busca
RETRIEVAL_K = 4  # Top 4 documentos mais relevantes

# Tipo de busca: "similarity" (padrão) ou "mmr" (Maximum Marginal Relevance)
SEARCH_TYPE = "similarity"

# ============================================================================
# Configurações de Chunking (Divisão de Documentos)
# ============================================================================

# Tamanho de cada chunk em caracteres
CHUNK_SIZE = 1000

# Overlap entre chunks (para manter contexto)
CHUNK_OVERLAP = 200

# ============================================================================
# Configurações do Agente
# ============================================================================

# System prompt para o agente
SYSTEM_PROMPT = """Você é um assistente especializado em responder perguntas com base em
                   documentos fornecidos.

INSTRUÇÕES:
1. Use a ferramenta 'retrieve_context' para buscar informações relevantes nos documentos
2. Base suas respostas APENAS nas informações encontradas nos documentos
3. Se não encontrar informação relevante nos documentos, diga que não tem informação suficiente
4. Seja claro, conciso e objetivo nas respostas
5. Cite trechos dos documentos quando apropriado

Sempre priorize a precisão sobre a criatividade."""

# Máximo de iterações do agente (evita loops infinitos)
MAX_ITERATIONS = 3

# ============================================================================
# Tipos de Arquivos Suportados
# ============================================================================

SUPPORTED_EXTENSIONS = {
    ".txt": "text",
    ".pdf": "pdf",
    ".md": "markdown",
    ".json": "json",
}

# ============================================================================
# Configurações de Logging
# ============================================================================

# Nível de logging
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Exibir mensagens de progresso
VERBOSE = True

# ============================================================================
# Funções Auxiliares
# ============================================================================

def print_config():
    """Exibe as configurações atuais do projeto"""
    print("\n" + "="*60)
    print("⚙️  CONFIGURAÇÕES DO PROJETO")
    print("="*60)
    print(f"📁 Diretório de dados: {DATA_DIR}")
    print(f"🗄️  Banco vetorial: {VECTOR_DB_DIR}")
    print(f"🤖 Modelo LLM: {LLM_MODEL}")
    print(f"📊 Modelo Embeddings: {EMBEDDING_MODEL}")
    print(f"📑 Tamanho do chunk: {CHUNK_SIZE} caracteres")
    print(f"🔄 Overlap: {CHUNK_OVERLAP} caracteres")
    print(f"🔍 Top-K retrieval: {RETRIEVAL_K}")
    print("="*60 + "\n")


def validate_config():
    """Valida se todas as configurações necessárias estão presentes"""
    errors = []
    
    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY não configurada")
    
    if CHUNK_SIZE <= CHUNK_OVERLAP:
        errors.append("CHUNK_SIZE deve ser maior que CHUNK_OVERLAP")
    
    if RETRIEVAL_K < 1:
        errors.append("RETRIEVAL_K deve ser pelo menos 1")
    
    if errors:
        raise ValueError(
            "❌ Erros de configuração encontrados:\n" + 
            "\n".join(f"  - {error}" for error in errors)
        )
    
    return True


# Executa validação ao importar
try:
    validate_config()
except ValueError as e:
    print(f"\n{e}\n")
    raise


# ============================================================================
# Uso
# ============================================================================

if __name__ == "__main__":
    # Exibe as configurações quando executado diretamente
    print_config()
    print("✅ Configurações validadas com sucesso!")

