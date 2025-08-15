import os
import json
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from main import PDFAgent
import streamlit as st


@dataclass
class TestConfig:
    """Configuração para teste de otimização."""
    chunk_size: int
    chunk_overlap: int
    k_results: int = 5


@dataclass
class TestResult:
    """Resultado de um teste de configuração."""
    config: TestConfig
    processing_time: float
    result_count: int
    avg_result_length: float
    coverage_score: float
    relevance_score: float
    overall_score: float


class PDFOptimizerAgent:
    """Agent responsável por otimizar configurações do PDF Agent."""
    
    def __init__(self, api_key: str, model_type: str = "gemini"):
        self.api_key = api_key
        self.model_type = model_type
        self.test_results: List[TestResult] = []
        
    def generate_test_configs(self) -> List[TestConfig]:
        """Gera diferentes configurações para teste."""
        configs = []
        
        # Chunk sizes variando de 500 a 2000
        chunk_sizes = [500, 750, 1000, 1250, 1500, 1750, 2000]
        
        # Overlaps variando de 0 a 500
        overlaps = [0, 100, 200, 300, 400, 500]
        
        # Combinações mais promissoras
        for chunk_size in chunk_sizes:
            for overlap in overlaps:
                # Evita overlap maior que chunk_size
                if overlap < chunk_size:
                    configs.append(TestConfig(chunk_size, overlap))
        
        return configs
    
    def test_configuration(self, pdf_path: str, config: TestConfig, test_query: str) -> TestResult:
        """Testa uma configuração específica."""
        start_time = time.time()
        
        try:
            # Inicializa o agent
            agent = PDFAgent(self.api_key, self.model_type)
            
            # Carrega o PDF
            agent.load_pdf(pdf_path, config.chunk_size, config.chunk_overlap)
            
            # Cria o vectorstore
            agent.create_vectorstore()
            
            # Faz a busca
            results = agent.similarity_search(test_query, config.k_results)
            
            processing_time = time.time() - start_time
            
            # Calcula métricas
            result_count = len(results)
            avg_result_length = sum(len(doc.page_content) for doc in results) / max(result_count, 1)
            
            # Score de cobertura (quanto do conteúdo foi encontrado)
            total_content_length = sum(len(doc.page_content) for doc in results)
            coverage_score = min(total_content_length / 10000, 1.0)  # Normalizado
            
            # Score de relevância (baseado no tamanho dos resultados)
            relevance_score = min(avg_result_length / 1000, 1.0)  # Normalizado
            
            # Score geral (combinação de velocidade, cobertura e relevância)
            speed_score = max(0, 1 - (processing_time / 10))  # Penaliza tempos longos
            overall_score = (coverage_score * 0.4 + relevance_score * 0.3 + speed_score * 0.3)
            
            return TestResult(
                config=config,
                processing_time=processing_time,
                result_count=result_count,
                avg_result_length=avg_result_length,
                coverage_score=coverage_score,
                relevance_score=relevance_score,
                overall_score=overall_score
            )
            
        except Exception as e:
            # Retorna resultado com score baixo em caso de erro
            return TestResult(
                config=config,
                processing_time=time.time() - start_time,
                result_count=0,
                avg_result_length=0,
                coverage_score=0,
                relevance_score=0,
                overall_score=0
            )
    
    def run_optimization(self, pdf_path: str, test_query: str, max_workers: int = 4) -> List[TestResult]:
        """Executa a otimização com múltiplas configurações."""
        configs = self.generate_test_configs()
        
        st.info(f"Testando {len(configs)} configurações diferentes...")
        
        # Usa ThreadPoolExecutor para testes paralelos
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submete todos os testes
            future_to_config = {
                executor.submit(self.test_configuration, pdf_path, config, test_query): config
                for config in configs
            }
            
            # Coleta resultados
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result()
                    self.test_results.append(result)
                    
                    # Atualiza progresso
                    progress = len(self.test_results) / len(configs)
                    st.progress(progress)
                    
                except Exception as e:
                    st.error(f"Erro ao testar configuração {config}: {e}")
        
        # Ordena por score geral
        self.test_results.sort(key=lambda x: x.overall_score, reverse=True)
        
        return self.test_results
    
    def get_best_config(self) -> TestResult:
        """Retorna a melhor configuração encontrada."""
        if not self.test_results:
            raise ValueError("Nenhum teste foi executado ainda.")
        return self.test_results[0]
    
    def get_recommendations(self) -> Dict[str, List[TestResult]]:
        """Retorna recomendações categorizadas."""
        if not self.test_results:
            return {}
        
        recommendations = {
            "melhor_geral": self.test_results[:3],
            "mais_rapido": sorted(self.test_results, key=lambda x: x.processing_time)[:3],
            "melhor_cobertura": sorted(self.test_results, key=lambda x: x.coverage_score, reverse=True)[:3],
            "melhor_relevancia": sorted(self.test_results, key=lambda x: x.relevance_score, reverse=True)[:3]
        }
        
        return recommendations


class OptimizationCrew:
    """Crew que coordena múltiplos agents de otimização."""
    
    def __init__(self, api_key: str, model_type: str = "gemini"):
        self.optimizer = PDFOptimizerAgent(api_key, model_type)
        self.results = []
    
    def optimize_pdf(self, pdf_path: str, test_queries: List[str]) -> Dict:
        """Otimiza um PDF usando múltiplas queries de teste."""
        st.header("🤖 Crew de Otimização de PDF")
        st.write("Testando diferentes configurações para encontrar os melhores parâmetros...")
        
        all_results = []
        
        for i, query in enumerate(test_queries, 1):
            st.subheader(f"Teste {i}: {query[:50]}...")
            
            results = self.optimizer.run_optimization(pdf_path, query)
            all_results.extend(results)
            
            # Mostra melhor resultado para esta query
            best = self.optimizer.get_best_config()
            st.success(f"Melhor configuração para esta query: Chunk={best.config.chunk_size}, Overlap={best.config.chunk_overlap}")
        
        # Agrega resultados de todas as queries
        self.results = all_results
        
        # Calcula configuração geral recomendada
        recommendations = self.optimizer.get_recommendations()
        
        return {
            "results": all_results,
            "recommendations": recommendations,
            "best_overall": self.optimizer.get_best_config()
        }


def main():
    st.set_page_config(
        page_title="PDF Optimizer Crew",
        page_icon="🔧",
        layout="wide"
    )
    
    st.title("🔧 PDF Optimizer Crew")
    st.markdown("Encontre os melhores parâmetros para seu PDF Agent!")
    
    with st.sidebar:
        st.header("Configurações")
        
        model_type = st.selectbox(
            "Modelo de IA",
            ["gemini", "openai"],
            help="Escolha entre Gemini (Google) ou OpenAI"
        )
        
        api_key = st.text_input(
            "API Key",
            type="password",
            placeholder="API Key",
            help=f"Cole sua {model_type.upper()} API key aqui"
        )
        
        max_workers = st.slider(
            "Paralelização",
            min_value=1,
            max_value=8,
            value=4,
            help="Número de testes simultâneos"
        )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload do PDF")
        uploaded_file = st.file_uploader(
            "Escolha um arquivo PDF para otimizar",
            type=['pdf'],
            help="Faça upload do PDF que você quer otimizar"
        )
    
    with col2:
        st.header("Queries de Teste")
        st.write("Digite algumas perguntas para testar diferentes aspectos do PDF:")
        
        test_queries = []
        for i in range(3):
            query = st.text_area(
                f"Query de teste {i+1}",
                placeholder="Ex: Qual é o tema principal do documento?",
                height=80
            )
            if query.strip():
                test_queries.append(query.strip())
        
        # Queries padrão se nenhuma for fornecida
        if not test_queries:
            test_queries = [
                "Qual é o tema principal do documento?",
                "Quais são os pontos mais importantes?",
                "Resuma o conteúdo principal"
            ]
    
    if st.button("🚀 Iniciar Otimização", type="primary") and uploaded_file and api_key:
        with st.spinner("Executando otimização..."):
            try:
                # Salva o PDF temporariamente
                with open("temp_optimize.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Cria o crew e executa otimização
                crew = OptimizationCrew(api_key, model_type)
                results = crew.optimize_pdf("temp_optimize.pdf", test_queries)
                
                # Mostra resultados
                st.success("✅ Otimização concluída!")
                
                # Melhor configuração geral
                best = results["best_overall"]
                st.header("🏆 Melhor Configuração Encontrada")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Chunk Size", best.config.chunk_size)
                with col2:
                    st.metric("Overlap", best.config.chunk_overlap)
                with col3:
                    st.metric("Score Geral", f"{best.overall_score:.3f}")
                with col4:
                    st.metric("Tempo", f"{best.processing_time:.2f}s")
                
                # Recomendações detalhadas
                st.header("📊 Recomendações Detalhadas")
                
                recommendations = results["recommendations"]
                
                tabs = st.tabs(["Melhor Geral", "Mais Rápido", "Melhor Cobertura", "Melhor Relevância"])
                
                with tabs[0]:
                    st.subheader("Top 3 - Melhor Geral")
                    for i, result in enumerate(recommendations["melhor_geral"], 1):
                        with st.expander(f"#{i}: Chunk={result.config.chunk_size}, Overlap={result.config.chunk_overlap} (Score: {result.overall_score:.3f})"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Tempo:** {result.processing_time:.2f}s")
                                st.write(f"**Resultados:** {result.result_count}")
                            with col2:
                                st.write(f"**Cobertura:** {result.coverage_score:.3f}")
                                st.write(f"**Relevância:** {result.relevance_score:.3f}")
                
                with tabs[1]:
                    st.subheader("Top 3 - Mais Rápido")
                    for i, result in enumerate(recommendations["mais_rapido"], 1):
                        with st.expander(f"#{i}: {result.processing_time:.2f}s (Chunk={result.config.chunk_size}, Overlap={result.config.chunk_overlap})"):
                            st.write(f"**Score Geral:** {result.overall_score:.3f}")
                            st.write(f"**Cobertura:** {result.coverage_score:.3f}")
                            st.write(f"**Relevância:** {result.relevance_score:.3f}")
                
                with tabs[2]:
                    st.subheader("Top 3 - Melhor Cobertura")
                    for i, result in enumerate(recommendations["melhor_cobertura"], 1):
                        with st.expander(f"#{i}: Cobertura {result.coverage_score:.3f} (Chunk={result.config.chunk_size}, Overlap={result.config.chunk_overlap})"):
                            st.write(f"**Score Geral:** {result.overall_score:.3f}")
                            st.write(f"**Tempo:** {result.processing_time:.2f}s")
                            st.write(f"**Relevância:** {result.relevance_score:.3f}")
                
                with tabs[3]:
                    st.subheader("Top 3 - Melhor Relevância")
                    for i, result in enumerate(recommendations["melhor_relevancia"], 1):
                        with st.expander(f"#{i}: Relevância {result.relevance_score:.3f} (Chunk={result.config.chunk_size}, Overlap={result.config.chunk_overlap})"):
                            st.write(f"**Score Geral:** {result.overall_score:.3f}")
                            st.write(f"**Tempo:** {result.processing_time:.2f}s")
                            st.write(f"**Cobertura:** {result.coverage_score:.3f}")
                
                # Configuração recomendada para copiar
                st.header("📋 Configuração Recomendada")
                st.code(f"""
# Configuração otimizada para seu PDF:
chunk_size = {best.config.chunk_size}
chunk_overlap = {best.config.chunk_overlap}
k_results = {best.config.k_results}

# Use estes valores no seu PDF Agent!
""")
                
            except Exception as e:
                st.error(f"Erro durante a otimização: {str(e)}")
    
    elif st.button("🚀 Iniciar Otimização", type="primary"):
        if not uploaded_file:
            st.error("Por favor, faça upload de um PDF.")
        if not api_key:
            st.error("Por favor, insira sua API key.")


if __name__ == "__main__":
    main()

