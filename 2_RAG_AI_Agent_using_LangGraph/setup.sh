#!/bin/bash
# Script de Setup para RAG Agent com LangGraph
# =============================================
# Este script facilita a configuração inicial do projeto

set -e  # Para em caso de erro

echo ""
echo "=========================================="
echo "🚀 Setup do RAG Agent com LangGraph"
echo "=========================================="
echo ""

# Cores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 1. Verificar se está no diretório correto
if [ ! -f "config.py" ]; then
    echo -e "${RED}❌ Erro: Execute este script a partir da pasta 2_RAG_AI_Agent_using_LangGraph${NC}"
    exit 1
fi

# 2. Criar arquivo .env se não existir
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}📝 Criando arquivo .env...${NC}"
    cp env.example .env
    echo -e "${GREEN}✅ Arquivo .env criado!${NC}"
    echo -e "${YELLOW}⚠️  IMPORTANTE: Edite o arquivo .env e adicione sua OPENAI_API_KEY${NC}"
    echo ""
else
    echo -e "${GREEN}✅ Arquivo .env já existe${NC}"
fi

# 3. Verificar se a pasta data existe
if [ ! -d "data" ]; then
    echo -e "${YELLOW}📁 Criando pasta data...${NC}"
    mkdir -p data
    echo -e "${GREEN}✅ Pasta data criada!${NC}"
else
    echo -e "${GREEN}✅ Pasta data já existe${NC}"
fi

# 4. Verificar se há documentos na pasta data
echo ""
echo -e "${YELLOW}📚 Verificando documentos...${NC}"
file_count=$(find data -type f \( -name "*.txt" -o -name "*.pdf" -o -name "*.md" \) | wc -l)

if [ "$file_count" -eq 0 ]; then
    echo -e "${YELLOW}⚠️  Nenhum documento encontrado na pasta data/${NC}"
    echo ""
    echo "Para adicionar documentos:"
    echo "  1. Copie seus arquivos (PDF, TXT, MD) para a pasta data/"
    echo "  2. Execute: uv run ingest_data.py"
    echo ""
else
    echo -e "${GREEN}✅ Encontrados $file_count documento(s)${NC}"
fi

# 5. Exibir próximos passos
echo ""
echo "=========================================="
echo "📋 Próximos Passos"
echo "=========================================="
echo ""
echo "1. Configure sua API Key:"
echo "   nano .env"
echo ""
echo "2. (Opcional) Adicione mais documentos:"
echo "   cp seus-documentos.pdf data/"
echo ""
echo "3. Ingira os documentos:"
echo "   uv run ingest_data.py"
echo ""
echo "4. Execute o agente:"
echo "   uv run rag_agent.py"
echo ""
echo "=========================================="
echo -e "${GREEN}✅ Setup concluído!${NC}"
echo "=========================================="
echo ""
