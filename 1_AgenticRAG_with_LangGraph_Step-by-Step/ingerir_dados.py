#! /usr/bin/env python3
"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

Script para INGESTÃO de dados - Executar UMA VEZ
Este script carrega os arquivos Markdown e cria o banco vetorial Chroma.
NÃO precisa executar novamente a menos que adicione novos documentos.
"""
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
import os
from dotenv import load_dotenv, find_dotenv

# Carregar variáveis de ambiente
_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


def ingerir_documentos():
    """
    Função para ingerir documentos Markdown e criar o banco vetorial Chroma.
    Executa UMA VEZ para criar a base de dados.
    """
    print("\n" + "=" * 70)
    print("📚 INGESTÃO DE DADOS - Criando Banco Vetorial Chroma")
    print("=" * 70 + "\n")
    
    # Definir caminhos
    resultados_dir = Path("/home/eddygiusepe/2_GitHub/Examples_of_RAG_Agents_with_LangGraph/1_AgenticRAG_with_LangGraph_Step-by-Step/Scraping_For_ri-vix/markdown_result_ri_vix")
    chroma_db_path = "./chroma_db_ri_vix"
    
    # Verificar se o diretório de resultados existe
    if not resultados_dir.exists():
        print(f"❌ Erro: Diretório não encontrado: {resultados_dir.absolute()}")
        print("💡 Execute primeiro o script de raspagem: python 1_AgenticRAG_with_LangGraph_Step-by-Step/Scraping_For_ri-vix/ri_vix_scraping.py")
        return False
    
    # Verificar se já existe um banco vetorial
    chroma_path = Path(chroma_db_path)
    if chroma_path.exists():
        resposta = input(f"\n⚠️  Já existe um banco vetorial em '{chroma_db_path}'.\n   Deseja RECRIAR? Isso vai deletar o banco existente. (s/n): ")
        if resposta.lower() not in ['s', 'sim', 'y', 'yes']:
            print("✅ Mantendo banco vetorial existente. Nada foi alterado.")
            return True
        else:
            print("🗑️  Removendo banco vetorial antigo...")
            import shutil
            shutil.rmtree(chroma_db_path)
    
    # Contar arquivos Markdown
    markdown_files = list(resultados_dir.glob("*.md"))
    print(f"📄 Encontrados {len(markdown_files)} arquivos Markdown:")
    for i, file in enumerate(markdown_files, 1):
        print(f"   {i}. {file.name}")
    
    if len(markdown_files) == 0:
        print("❌ Erro: Nenhum arquivo Markdown encontrado!")
        print("💡 Execute primeiro o script de raspagem: python 1_AgenticRAG_with_LangGraph_Step-by-Step/Scraping_For_ri-vix/ri_vix_scraping.py")
        return False
    
    print("\n" + "-" * 70)
    print("🔄 ETAPA 1: Carregando documentos...")
    print("-" * 70)
    
    # Carregar documentos
    loader = DirectoryLoader(
        path=str(resultados_dir),
        glob="*.md",
        loader_cls=UnstructuredMarkdownLoader,
        show_progress=True
    )
    
    docs = loader.load()
    print(f"✅ {len(docs)} documentos carregados com sucesso!")
    
    # Mostrar estatísticas dos documentos
    total_chars = sum(len(doc.page_content) for doc in docs)
    print(f"📊 Total de caracteres: {total_chars:,}")
    
    print("\n" + "-" * 70)
    print("🔄 ETAPA 2: Dividindo documentos em chunks...")
    print("-" * 70)
    
    # Dividir documentos em chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, 
        chunk_overlap=300
    )
    doc_splits = text_splitter.split_documents(docs)
    print(f"✅ {len(doc_splits)} chunks criados")
    
    print("\n" + "-" * 70)
    print("🔄 ETAPA 3: Criando embeddings e salvando no Chroma...")
    print("-" * 70)
    print("⚠️  IMPORTANTE: Esta etapa usa a API da OpenAI e pode levar alguns minutos.")
    print("⚠️  Os embeddings serão persistidos e NÃO precisarão ser recriados.")
    
    # Criar embeddings e salvar no Chroma
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="ri_vix_docs",
        embedding=embeddings,
        persist_directory=chroma_db_path
    )
    
    print(f"✅ Banco vetorial criado e persistido em: {chroma_db_path}")
    
    # Testar o banco vetorial
    print("\n" + "-" * 70)
    print("🔄 ETAPA 4: Testando o banco vetorial...")
    print("-" * 70)
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}  # Número de documentos a retornar
    )
    test_query = "O que é a VIX Logística?"
    results = retriever.invoke(test_query)
    
    print(f"✅ Teste concluído: {len(results)} documentos recuperados para query de teste")
    
    print("\n" + "=" * 70)
    print("🎉 INGESTÃO CONCLUÍDA COM SUCESSO!")
    print("=" * 70)
    print(f"\n📁 Banco vetorial salvo em: {Path(chroma_db_path).absolute()}")
    print(f"📊 Total de chunks indexados: {len(doc_splits)}")
    print(f"💾 Espaço em disco usado: ~{(total_chars / 1024):.2f} KB (aproximado)")
    print("\n💡 Agora você pode executar o agente sem recriar os embeddings:")
    print("   uv run agente_langgraph.py")
    print("\n⚠️  DICA: Só execute este script novamente se adicionar novos documentos!")
    print("=" * 70 + "\n")
    
    return True


if __name__ == "__main__":
    try:
        sucesso = ingerir_documentos()
        if not sucesso:
            exit(1)
    except KeyboardInterrupt:
        print("\n\n👋 Interrompido pelo usuário.")
        exit(1)
    except Exception as e:
        print(f"\n❌ Erro durante a ingestão: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)

