#!/usr/bin/env python3
"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

Script de Ingestão de Dados para RAG Agent
==========================================
Este script processa documentos da pasta 'data/' e cria um banco vetorial
usando Chroma para posterior recuperação de contexto pelo agente RAG.

Processo:
1. Carrega documentos da pasta data/
2. Divide documentos em chunks
3. Cria embeddings usando OpenAI
4. Armazena no banco vetorial Chroma

Uso:
----
uv run ingest_data.py
"""

import sys
from pathlib import Path
from typing import List

# Importações do LangChain
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import (
    DATA_DIR,
    VECTOR_DB_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    COLLECTION_NAME,
    SUPPORTED_EXTENSIONS,
    VERBOSE,
    OPENAI_API_KEY,
)


# ============================================================================
# Funções de Carregamento de Documentos
# ============================================================================

def load_documents_from_directory(directory: Path) -> List[Document]:
    """
    Carrega todos os documentos suportados de um diretório.
    
    Args:
        directory: Caminho do diretório contendo os documentos
        
    Returns:
        Lista de documentos carregados
    """
    print(f"\n📂 Carregando documentos de: {directory}")
    
    # Verifica se o diretório existe
    if not directory.exists():
        print(f"❌ Diretório não encontrado: {directory}")
        print(f"💡 Criando diretório: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        return []
    
    all_documents = []
    
    # Carrega cada tipo de arquivo suportado
    for extension, file_type in SUPPORTED_EXTENSIONS.items():
        pattern = f"*{extension}"
        files = list(directory.glob(pattern))
        
        if files:
            print(f"Encontrados {len(files)} arquivo(s) {extension}")
            
            for file_path in files:
                try:
                    # Seleciona o loader apropriado
                    if extension == ".pdf":
                        loader = PyPDFLoader(str(file_path))
                    else:  # .txt, .md, .json
                        loader = TextLoader(str(file_path), encoding='utf-8')
                    
                    # Carrega o documento
                    docs = loader.load()
                    all_documents.extend(docs)
                    
                    if VERBOSE:
                        print(f"    ✅ {file_path.name} - {len(docs)} página(s)/seção(ões)")
                        
                except Exception as e:
                    print(f"    ❌ Erro ao carregar {file_path.name}: {e}")
    
    print(f"\n✅ Total de documentos carregados: {len(all_documents)}\n")
    return all_documents


# ============================================================================
# Funções de Chunking (Divisão de Documentos)
# ============================================================================

def split_documents(documents: List[Document]) -> List[Document]:
    """
    Divide documentos em chunks menores para melhor recuperação.
    
    O RecursiveCharacterTextSplitter tenta dividir o texto de forma inteligente:
    1. Primeiro tenta dividir por parágrafos (\n\n)
    2. Se muito grande, divide por frases (\n)
    3. Se ainda muito grande, divide por palavras (espaços)
    4. Como último recurso, divide por caracteres
    
    Args:
        documents: Lista de documentos a serem divididos
        
    Returns:
        Lista de chunks de documentos
    """
    print("✂️  Dividindo documentos em chunks...")
    print(f"   Tamanho do chunk: {CHUNK_SIZE} caracteres")
    print(f"   Overlap: {CHUNK_OVERLAP} caracteres\n")
    
    # Cria o text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
        separators=[
            "\n\n",  # Parágrafos
            "\n",    # Linhas
            " ",     # Palavras
            "",      # Caracteres
        ]
    )
    
    # Divide os documentos
    chunks = text_splitter.split_documents(documents)
    
    print(f"✅ Documentos divididos em {len(chunks)} chunks\n")
    
    # Estatísticas dos chunks
    if chunks and VERBOSE:
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        print("📊 Estatísticas dos chunks:")
        print(f"   Mínimo: {min(chunk_sizes)} caracteres")
        print(f"   Máximo: {max(chunk_sizes)} caracteres")
        print(f"   Média: {sum(chunk_sizes) / len(chunk_sizes):.0f} caracteres\n")
    
    return chunks


# ============================================================================
# Funções de Criação do Vector Store
# ============================================================================

def create_vector_store(chunks: List[Document]) -> Chroma:
    """
    Cria um banco vetorial Chroma a partir dos chunks de documentos.
    
    Processo:
    1. Cria embeddings para cada chunk usando OpenAI
    2. Armazena os embeddings no Chroma
    3. Persiste o banco vetorial em disco
    
    Args:
        chunks: Lista de chunks de documentos
        
    Returns:
        Instância do Chroma vector store
    """
    print("    Criando embeddings e armazenando no Chroma...")
    print(f"   Modelo: {EMBEDDING_MODEL}")
    print(f"   Coleção: {COLLECTION_NAME}")
    print(f"   Destino: {VECTOR_DB_DIR}\n")
    
    # Cria o embedding model
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
        # chunk_size=1000,  # Número de textos a embeddar por vez
    )
    
    # Remove banco vetorial antigo se existir
    if VECTOR_DB_DIR.exists():
        print("🗑️  Removendo banco vetorial anterior...")
        import shutil
        shutil.rmtree(VECTOR_DB_DIR)
    
    # Cria o vector store
    print("⚡ Criando embeddings (isso pode demorar alguns segundos)...\n")
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTOR_DB_DIR),
    )
    
    print("✅ Banco vetorial criado com sucesso!")
    print(f"   📍 Localização: {VECTOR_DB_DIR}")
    print(f"   📊 Total de chunks armazenados: {len(chunks)}\n")
    
    return vector_store


# ============================================================================
# Funções de Teste
# ============================================================================

def test_vector_store(vector_store: Chroma):
    """
    Testa o banco vetorial com uma busca de exemplo.
    
    Args:
        vector_store: Instância do Chroma vector store
    """
    print("🧪 Testando o banco vetorial...")
    
    # Pergunta de teste
    test_query = "Qual é o tema principal deste documento?"
    
    print(f"   Pergunta de teste: '{test_query}'")
    
    # Busca documentos similares
    results = vector_store.similarity_search(test_query, k=2)
    
    if results:
        print(f"\n✅ Teste bem-sucedido! Encontrados {len(results)} resultado(s):\n")
        
        for i, doc in enumerate(results, 1):
            content_preview = doc.page_content[:200].replace('\n', ' ')
            print(f"   {i}. {content_preview}...")
            print(f"      Fonte: {doc.metadata.get('source', 'Unknown')}\n")
    else:
        print("\n⚠️  Nenhum resultado encontrado. Verifique se os documentos foram carregados corretamente.")


# ============================================================================
# Função Principal
# ============================================================================

def main():
    """
    Função principal que orquestra todo o processo de ingestão.
    """
    print("\n" + "="*70)
    print("🚀 INICIANDO PROCESSO DE INGESTÃO DE DADOS")
    print("="*70)
    
    # 1. Verifica se há documentos
    if not DATA_DIR.exists() or not any(DATA_DIR.iterdir()):
        print(f"\n❌ ERRO: Nenhum documento encontrado em {DATA_DIR}")
        print("\n💡 SOLUÇÃO:")
        print(f"   1. Crie a pasta: mkdir -p {DATA_DIR}")
        print("   2. Adicione seus documentos (PDF, TXT, MD, JSON)")
        print("   3. Execute este script novamente\n")
        print(f"📚 Extensões suportadas: {', '.join(SUPPORTED_EXTENSIONS.keys())}")
        print("="*70 + "\n")
        sys.exit(1)
    
    # 2. Carrega documentos
    documents = load_documents_from_directory(DATA_DIR)
    
    if not documents:
        print("❌ Nenhum documento foi carregado com sucesso!")
        print(f"   Verifique se os arquivos em {DATA_DIR} estão em formatos suportados.")
        sys.exit(1)
    
    # 3. Divide em chunks
    chunks = split_documents(documents)
    
    if not chunks:
        print("❌ Erro ao dividir documentos em chunks!")
        sys.exit(1)
    
    # 4. Cria vector store
    vector_store = create_vector_store(chunks)
    
    # 5. Testa o vector store
    test_vector_store(vector_store)
    
    print("="*70)
    print("✅ PROCESSO DE INGESTÃO CONCLUÍDO COM SUCESSO!")
    print("="*70)
    print("\n💡 Próximos passos:")
    print("   1. Execute o agente RAG: uv run rag_agent.py")
    print("   2. Faça perguntas sobre seus documentos!")
    print("="*70 + "\n")


# ============================================================================
# Ponto de Entrada
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Processo interrompido pelo usuário.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

