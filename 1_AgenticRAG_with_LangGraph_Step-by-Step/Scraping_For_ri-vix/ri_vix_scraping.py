#! /usr/bin/env python3
"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

Run
===
uv run ri_vix_scraping.py
"""
import asyncio
import re
#from datetime import datetime
from pathlib import Path
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode


def limpar_markdown(texto: str) -> str:
    """
    Remove elementos indesejados do markdown gerado.
    """
    # Remover linhas com javascript:void
    linhas = texto.split('\n')
    linhas_limpas = []
    
    for linha in linhas:
        # Pular linhas com javascript:void
        if 'javascript:void' in linha:
            continue
        # Remover links vazios com formato []()
        linha = re.sub(r'\[\]\([^)]*\)', '', linha)
        # Remover múltiplas linhas em branco consecutivas
        if linha.strip() or (linhas_limpas and linhas_limpas[-1].strip()):
            linhas_limpas.append(linha)
    
    # Juntar as linhas e remover espaços extras
    texto_limpo = '\n'.join(linhas_limpas)
    
    # Remover múltiplas linhas em branco (mais de 2 seguidas)
    texto_limpo = re.sub(r'\n{3,}', '\n\n', texto_limpo)
    
    return texto_limpo.strip()


def gerar_nome_arquivo(url: str) -> str:
    """
    Gera um nome de arquivo baseado na URL.
    """
    # Extrair a última parte da URL como nome
    nome = url.rstrip('/').split('/')[-1]
    # Remover caracteres especiais
    nome = re.sub(r'[^\w\-]', '_', nome)
    # Limitar tamanho
    return nome[:50] if nome else "pagina"


async def raspar_url(crawler, url: str, output_dir: Path, config: CrawlerRunConfig) -> dict:
    """
    Raspa uma única URL e salva em arquivo.
    Retorna um dicionário com informações sobre o resultado.
    """
    try:
        print(f"  🔍 Raspando: {url}")
        
        result = await crawler.arun(url=url, config=config)
        
        # Limpar o markdown
        markdown_limpo = limpar_markdown(result.markdown)
        
        # Gerar nome do arquivo baseado na URL
        nome_base = gerar_nome_arquivo(url)
        #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{nome_base}.md"  #f"{nome_base}_{timestamp}.md"
        filepath = output_dir / filename
        
        # Salvar o markdown em arquivo:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# {nome_base.replace('_', ' ').title()}\n\n")
            f.write(f"**URL:** {url}\n\n")
            #f.write(f"**Data:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")
            f.write("---\n\n")
            f.write(markdown_limpo)
        
        return {
            "url": url,
            "sucesso": True,
            "arquivo": filepath,
            "tamanho": len(markdown_limpo),
            "erro": None
        }
        
    except Exception as e:
        print(f"  ❌ Erro ao raspar {url}: {str(e)}")
        return {
            "url": url,
            "sucesso": False,
            "arquivo": None,
            "tamanho": 0,
            "erro": str(e)
        }


async def main():
    # Lista de URLs para fazer a raspagem
    urls = [
            # PÁGINA INICIAL
            "https://ri.vix.com.br/",
            
            # A COMPANHIA
            "https://ri.vix.com.br/a-companhia/perfil/",
            "https://ri.vix.com.br/a-companhia/missao-visao-e-valores/",
            "https://ri.vix.com.br/a-companhia/estrategia-e-vantagens-competitivas/",
            "https://ri.vix.com.br/a-companhia/premios-e-certificacoes/",
            #"https://ri.vix.com.br/a-companhia/na-midia/",
            
            # GOVERNANÇA CORPORATIVA
            #"https://ri.vix.com.br/governanca-corporativa/estrutura-societaria/",
            #"https://ri.vix.com.br/governanca-corporativa/conselhos-e-diretoria/",
            #"https://ri.vix.com.br/governanca-corporativa/comites/",
            #"https://ri.vix.com.br/governanca-corporativa/estatuto-codigos-e-politicas/",
            #"https://ri.vix.com.br/governanca-corporativa/atas-de-reuniao/",
            #"https://ri.vix.com.br/governanca-corporativa/sustentabilidade-e-inovacao/",
            #"https://ri.vix.com.br/governanca-corporativa/programa-de-integridade/",
            #"https://ri.vix.com.br/governanca-corporativa/canal-de-denuncia/",
            #"https://ri.vix.com.br/governanca-corporativa/informe-de-governanca/",
            
            # SUSTENTABILIDADE
            #"https://ri.vix.com.br/sustentabilidade/",
            
            # INFORMAÇÕES FINANCEIRAS
            #"https://ri.vix.com.br/informacoes-financeiras/central-de-resultados/",
            #"https://ri.vix.com.br/informacoes-financeiras/rating/",
            #"https://ri.vix.com.br/informacoes-financeiras/dividendos-e-jcp/",
            #"https://ri.vix.com.br/informacoes-financeiras/dividas/",
            #"https://ri.vix.com.br/informacoes-financeiras/planilhas-interativas/",
            #"https://ri.vix.com.br/informacoes-financeiras/apresentacoes-institucionais/",
            
            # DOCUMENTOS CVM
            #"https://ri.vix.com.br/documentos-cvm/fatos-relevantes/",
            #"https://ri.vix.com.br/documentos-cvm/formulario-de-referencia/",
            #"https://ri.vix.com.br/documentos-cvm/outros-documentos/",
            
            # SERVIÇOS AOS INVESTIDORES
            #"https://ri.vix.com.br/servicos-aos-investidores/fatores-de-risco/",
            #"https://ri.vix.com.br/servicos-aos-investidores/central-de-downloads/",
            #"https://ri.vix.com.br/servicos-aos-investidores/perguntas-frequentes/",
            #"https://ri.vix.com.br/servicos-aos-investidores/glossario/",
            #"https://ri.vix.com.br/servicos-aos-investidores/calendario-de-eventos/",
        ]
    
    # Criar diretório para salvar os resultados
    output_dir = Path("markdown_result_ri_vix")
    output_dir.mkdir(exist_ok=True)
    
    # Configurações do navegador
    browser_config = BrowserConfig(
        headless=True,
        verbose=False
    )
    
    # Configurações do crawler
    crawler_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        wait_for_images=False,
        js_code=[
            "document.querySelectorAll('nav, header, footer, .menu, .nav').forEach(el => el.remove());"
        ]
    )
    
    print(f"🚀 Iniciando raspagem de {len(urls)} URLs")
    print("=" * 60)
    
    resultados = []
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        # Raspar todas as URLs
        for i, url in enumerate(urls, 1):
            print(f"\n[{i}/{len(urls)}]")
            resultado = await raspar_url(crawler, url, output_dir, crawler_config)
            resultados.append(resultado)
    
    # Mostrar resumo
    print("\n" + "=" * 30)
    print("📊 RESUMO DA RASPAGEM")
    print("=" * 30)
    
    sucesso = sum(1 for r in resultados if r["sucesso"])
    falhas = len(resultados) - sucesso
    total_caracteres = sum(r["tamanho"] for r in resultados if r["sucesso"])
    
    print(f"✅ Sucessos: {sucesso}/{len(urls)}")
    print(f"❌ Falhas: {falhas}")
    print(f"📄 Total de caracteres: {total_caracteres:,}")
    print(f"📁 Arquivos salvos em: {output_dir.absolute()}")
    
    # Listar arquivos criados
    if sucesso > 0:
        print("\n📝 Arquivos criados:")
        for resultado in resultados:
            if resultado["sucesso"]:
                print(f"  • {resultado['arquivo'].name} ({resultado['tamanho']:,} caracteres)")
    
    # Listar erros, se houver
    if falhas > 0:
        print("\n⚠️  URLs com erro:")
        for resultado in resultados:
            if not resultado["sucesso"]:
                print(f"  • {resultado['url']}")
                print(f"    Erro: {resultado['erro']}")


if __name__ == "__main__":
    asyncio.run(main())