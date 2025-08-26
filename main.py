import streamlit as st
import os
import time
import re
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

# Carregar variáveis de ambiente
load_dotenv()

# Configurações
API_KEY = os.getenv("GOOGLE_API_KEY")
VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "./chroma_db")
K_ = int(os.getenv("K_DOCUMENTS", "5"))

# Script de avaliação
evaluation_script = [
    {
        "id": 1,
        "question": "What are the three different statuses of a well once downhole activities or production are discontinued?",
        "ideal_answer": (
            "Os três status são:\n"
            "- **Suspensão:** O equipamento de controle do poço não é removido.\n"
            "- **Abandono Temporário:** O equipamento de controle do poço é removido com a intenção de reentrada posterior ou abandono permanente.\n"
            "- **Abandono Permanente:** O poço, ou parte dele, é tamponado e abandonado com a intenção de nunca mais ser reutilizado ou reentrado."
        ),
        "keywords": ["suspensão", "abandono temporário", "abandono permanente", "equipamento", "tamponado"]
    },
    {
        "id": 2,
        "question": "What is the primary objective of a plug and abandonment (P&A) operation and why is it necessary?",
        "ideal_answer": (
            "O principal objetivo de uma operação de P&A é restaurar a funcionalidade da rocha de capeamento para garantir a integridade do poço permanentemente. "
            "A operação é necessária para estabelecer barreiras que impeçam o fluxo de fluidos perigosos para os arredores, como o ambiente marinho, águas subterrâneas, o solo ou a atmosfera."
        ),
        "keywords": ["objetivo", "integridade", "barreiras", "fluidos perigosos", "ambiente"],
    },
    {
        "id": 3,
        "question": "Explain the 'two-barrier' philosophy (or 'hat-over-hat' principle) in the context of well P&A.",
        "ideal_answer": (
            "A filosofia de duas barreiras estipula que o poço deve ser equipado com duas barreiras de poço independentes: uma barreira primária e uma secundária. "
            "A barreira primária é o primeiro invólucro que impede o fluxo de uma fonte potencial. "
            "A barreira secundária atua como um backup para a barreira primária caso ela falhe. "
            "Este conceito também é conhecido como o princípio 'chapéu sobre chapéu'."
        ),
        "keywords": ["duas barreiras", "primária", "secundária", "backup", "hat-over-hat"],
    },
    {
        "id": 4,
        "question": "List five common challenges associated with P&A operations.",
        "ideal_answer": (
            "Cinco desafios comuns mencionados são: altas temperaturas, formações não consolidadas, alterações na resistência da formação devido à depleção, "
            "estresse tectônico (como cisalhamento e subsidência), pressão sustentada no revestimento (SCP), falta de dados de poços antigos e a dificuldade "
            "de verificar o cimento atrás da segunda coluna de revestimento."
        ),
        "keywords": ["altas temperaturas", "formações não consolidadas", "depleção", "estresse tectônico", "SCP"],
    },
    {
        "id": 5,
        "question": "Compare Portland cement and thermosetting polymers as permanent plugging materials, mentioning two advantages and two limitations of each.",
        "ideal_answer": (
            "**Cimento Portland:**\n"
            "- **Vantagens:** É o principal material de barreira usado na indústria do petróleo, sendo, portanto, bem conhecido e estudado. "
            "Existem várias classes API disponíveis para se adequar a diferentes condições de poço.\n"
            "- **Limitações:** Pode sofrer retração, o que pode criar microanéis. Pode ser frágil e degradar a longo prazo devido à exposição a altas temperaturas e substâncias químicas como H₂S e CO₂.\n\n"
            "**Polímeros Termofixos (Resinas):**\n"
            "- **Vantagens:** Possuem permeabilidade muito baixa (estanques a gás), forte adesão à formação e ao aço, e boas propriedades mecânicas.\n"
            "- **Limitações:** Geralmente são frágeis no estado sólido. A durabilidade a longo prazo é parcialmente desconhecida e pode haver retração química."
        ),
        "keywords": ["portland", "polímeros termofixos", "vantagens", "limitações", "permeabilidade", "adesão"],
    }
]


@st.cache_resource
def load_vectorstore():
    """Carrega o vectorstore do Chroma"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=API_KEY,
            model="models/embedding-001"
        )
        
        vectorstore = Chroma(
            persist_directory=VECTORSTORE_PATH,
            embedding_function=embeddings
        )
        return vectorstore
    except Exception as e:
        st.error(f"Erro ao carregar vectorstore: {e}")
        return None


@st.cache_resource
def load_llm():
    """Carrega o modelo LLM"""
    return ChatGoogleGenerativeAI(
        google_api_key=API_KEY,
        model="models/gemini-2.0-flash",
        temperature=0.3
    )


def get_answer(query: str, k: int = K_):
    """Função para obter resposta usando similarity search"""
    vectorstore = load_vectorstore()
    llm = load_llm()
    
    if not vectorstore or not llm:
        return None, []
    
    relevant_docs = vectorstore.similarity_search(query, k=k)
    
    context_text = "\n\n".join([
        f"Fonte: {doc.metadata.get('filename', 'Unknown')} - Página {doc.metadata.get('page', 'N/A')}\nConteúdo: {doc.page_content}"
        for doc in relevant_docs
    ])

    system_prompt = """
    Você é um assistente especializado em analisar documentos PDF. 
    Sua tarefa é responder perguntas dos usuários baseando-se APENAS no contexto fornecido.

    Regras importantes:
    1. Responda APENAS com base nas informações fornecidas no contexto
    2. Se a informação não estiver no contexto, diga claramente que não tem essa informação
    3. Seja preciso e direto nas respostas
    4. Cite as fontes quando relevante
    5. Responda em português brasileiro
    6. Se houver informações conflitantes, mencione isso
    7. Organize a resposta de forma clara e estruturada

    Contexto dos documentos:
    {context}

    Pergunta do usuário: {question}

    Responda de forma clara, útil e bem estruturada:
    """
            
    prompt = ChatPromptTemplate.from_template(system_prompt)

    messages = prompt.format_messages(
        context=context_text,
        question=query
    )

    response = llm.invoke(messages)
    return response, relevant_docs


def clean_text(text: str) -> str:
    """Limpa o texto para comparação"""
    text = re.sub(r'\[cite:\s*\d+\]', '', text)
    text = re.sub(r'\[cite_start\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\-\.\,\;\:\!\?]', '', text)
    return text.strip().lower()


def get_evaluation(question, ideal_answer, agent_answer):
    """Função para avaliar a qualidade da resposta"""
    
    evaluation_prompt = """
    Você é um especialista em avaliação de qualidade de respostas. Sua tarefa é comparar uma resposta do agente com uma resposta ideal e dar um score baseado na similaridade semântica e factual.

    CRITÉRIOS DE AVALIAÇÃO:
    - **Score 0 (Totalmente Diferente)**: A resposta do agente não aborda o tema da pergunta ou fornece informações completamente incorretas/irrelevantes
    - **Score 1 (Pouco Diferente)**: A resposta aborda o tema mas com informações limitadas, imprecisas ou muito superficiais
    - **Score 2 (Similar)**: A resposta aborda bem o tema com informações corretas, mas pode estar incompleta ou ter pequenas diferenças
    - **Score 3 (Igual)**: A resposta é muito similar à ideal, cobrindo os mesmos pontos principais com precisão

    CONSIDERE:
    - Precisão factual das informações
    - Cobertura dos tópicos principais
    - Clareza e estrutura da resposta
    - Uso correto de terminologia técnica
    - Completude da resposta

    PERGUNTA: {question}

    RESPOSTA IDEAL: {ideal_answer}

    RESPOSTA DO AGENTE: {agent_answer}

    INSTRUÇÕES:
    1. Analise cuidadosamente ambas as respostas
    2. Compare o conteúdo semântico e factual
    3. Atribua um score de 0 a 3
    4. Forneça uma justificativa breve para o score

    FORMATO DA RESPOSTA:
    Score: [0-3]
    Justificativa: [explicação do score]

    Responda apenas com o score e justificativa:
    """
    
    llm = load_llm()
    if not llm:
        return "Erro: LLM não disponível"
    
    evaluation_template = ChatPromptTemplate.from_template(evaluation_prompt)
    
    messages = evaluation_template.format_messages(
        question=question,
        ideal_answer=ideal_answer,
        agent_answer=agent_answer
    )
        
    response = llm.invoke(messages)
    return response.content.strip()


def main():
    st.set_page_config(
        page_title="Decomissioning Agent Interface",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("Decomissioning Agent Interface")
    
    # Verificar se a API Key está configurada
    if not API_KEY:
        st.error("❌ API Key não configurada. Crie um arquivo .env com GOOGLE_API_KEY=your_key_here")
        st.stop()
    
    # Verificar se o vectorstore existe
    if not os.path.exists(VECTORSTORE_PATH):
        st.error("Vectorstore não encontrado. Execute primeiro o script create_embeddings.py")
        return
    
    # Criar abas
    tab1, tab2 = st.tabs(["💬 Chat", "📊 Evaluation"])
    
    with tab1:
        st.header("💬 Chat com Similarity Search")
        
        # Input da pergunta
        query = st.text_area(
            "Digite sua pergunta:",
            placeholder="Ex: What are the types of cement?",
            height=100
        )
        
        # Parâmetros e botão na mesma linha
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            k = st.slider("Número de documentos relevantes (K)", 1, 20, K_)
        with col2:
            st.write("")  # Espaçamento
        with col3:
            search_button = st.button("🔍 Buscar Resposta", type="primary", use_container_width=True)
        
        # Resposta ocupando toda a largura
        if search_button:
            if query.strip():
                with st.spinner("Buscando resposta..."):
                    start_time = time.time()
                    result, relevant_docs = get_answer(query, k=k)
                    end_time = time.time()
                    
                    if result:
                        st.success(f"Resposta encontrada em {end_time - start_time:.2f}s")
                        
                        # Mostrar resposta ocupando toda a largura
                        st.subheader("🤖 Resposta:")
                        st.markdown(result.content)
                        
                        # Mostrar fontes
                        if relevant_docs:
                            st.subheader(f"📚 Fontes ({len(relevant_docs)} documentos):")
                            for i, source in enumerate(relevant_docs, 1):
                                with st.expander(f"{i}. {source.metadata.get('filename', 'Unknown')} - Página {source.metadata.get('page', 'N/A')}"):
                                    st.text(source.page_content[:500] + "..." if len(source.page_content) > 500 else source.page_content)
                    else:
                        st.error("Erro ao buscar resposta")
            else:
                st.warning("Digite uma pergunta")
    
    with tab2:
        st.header("📊 Evaluation")
        
        # Card explicativo sobre a avaliação
        with st.container():
            st.markdown("### ℹ️ Como funciona a avaliação")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("""
                **📋 Processo de Avaliação:**
                
                Esta seção avalia automaticamente a qualidade das respostas do agente usando um conjunto de **5 perguntas pré-definidas** com **respostas ideais** conhecidas. O sistema:
                
                • **Busca** documentos relevantes para cada pergunta
                • **Gera** uma resposta usando similarity search
                • **Compara** com a resposta ideal esperada
                • **Avalia** a qualidade usando IA especializada
                • **Conta** palavras-chave encontradas vs. esperadas
                
                **🎯 Métricas Avaliadas:**
                • Tempo de resposta
                • Número de documentos encontrados
                • Palavras-chave identificadas
                • Score de similaridade semântica
                """)
            
            with col2:
                st.markdown("""
                **📊 Sistema de Scores:**
                
                **0 - Totalmente Diferente**
                Resposta não aborda o tema ou fornece informações incorretas
                
                **1 - Pouco Diferente**
                Aborda o tema mas com informações limitadas/superficiais
                
                **2 - Similar**
                Aborda bem o tema mas pode estar incompleta
                
                **3 - Igual**
                Muito similar à ideal, cobrindo os mesmos pontos principais
                """)
        
        st.divider()
        
        if st.button("🚀 Executar Avaliação Completa", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            
            for i, evaluation in enumerate(evaluation_script):
                status_text.text(f"Processando pergunta {i+1}/{len(evaluation_script)}: {evaluation['question'][:50]}...")
                
                # Buscar resposta
                start_time = time.time()
                result, relevant_docs = get_answer(evaluation['question'], k=10)
                end_time = time.time()
                
                if result:
                    agent_answer = result.content
                    response_time = end_time - start_time
                    
                    # Contar palavras-chave encontradas
                    found_keywords = []
                    for keyword in evaluation['keywords']:
                        if keyword.lower() in agent_answer:
                            found_keywords.append(keyword)
                    
                    # Avaliar similaridade
                    evaluation_result = get_evaluation(
                        evaluation['question'], 
                        evaluation['ideal_answer'], 
                        agent_answer
                    )
                    
                    results.append({
                        'id': evaluation['id'],
                        'question': evaluation['question'],
                        'response_time': response_time,
                        'docs_found': len(relevant_docs),
                        'keywords_found': len(found_keywords),
                        'total_keywords': len(evaluation['keywords']),
                        'evaluation': evaluation_result,
                        'agent_answer': agent_answer,
                        'ideal_answer': evaluation['ideal_answer']
                    })
                
                progress_bar.progress((i + 1) / len(evaluation_script))
            
            status_text.text("Avaliação concluída!")
            
            # Mostrar resultados
            st.subheader("📈 Resultados da Avaliação")
            
            for result in results:
                with st.expander(f"Pergunta {result['id']}: {result['question']}"):
                    # Pergunta completa
                    st.markdown(f"**❓ Pergunta:** {result['question']}")
                    
                    st.divider()
                    
                    # Métricas em cards
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("⏱️ Tempo", f"{result['response_time']:.2f}s")
                    
                    with col2:
                        st.metric("📄 Documentos", result['docs_found'])
                    
                    with col3:
                        st.metric("🎯 Palavras-chave", f"{result['keywords_found']}/{result['total_keywords']}")
                    
                    with col4:
                        # Extrair score da avaliação
                        evaluation_text = result['evaluation']
                        if "Score:" in evaluation_text:
                            score = evaluation_text.split("Score:")[1].split("\n")[0].strip()
                            
                            # Descrição do score
                            score_descriptions = {
                                "0": "Totalmente Diferente",
                                "1": "Pouco Diferente", 
                                "2": "Similar",
                                "3": "Igual"
                            }
                            
                            score_value = score
                            score_desc = score_descriptions.get(score_value, "N/A")
                            
                            st.metric("📊 Score", f"{score_value} - {score_desc}")
                        else:
                            st.metric("📊 Score", "N/A")
                    
                    st.divider()
                    
                    # Palavras-chave encontradas vs necessárias
                    st.markdown("**🔑 Palavras-chave:**")
                    
                    # Encontrar palavras-chave na resposta
                    found_keywords = []
                    for keyword in evaluation_script[result['id']-1]['keywords']:
                        if keyword.lower() in result['agent_answer'].lower():
                            found_keywords.append(keyword)
                    
                    # Criar visualização das palavras-chave
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**✅ Encontradas:**")
                        if found_keywords:
                            for keyword in found_keywords:
                                st.markdown(f"• {keyword}")
                        else:
                            st.markdown("*Nenhuma palavra-chave encontrada*")
                    
                    with col2:
                        st.markdown("**❌ Não encontradas:**")
                        missing_keywords = [kw for kw in evaluation_script[result['id']-1]['keywords'] if kw not in found_keywords]
                        if missing_keywords:
                            for keyword in missing_keywords:
                                st.markdown(f"• {keyword}")
                        else:
                            st.markdown("*Todas as palavras-chave foram encontradas!*")
                    
                    st.divider()
                    
                    # Avaliação completa
                    st.markdown("**📋 Avaliação Detalhada:**")
                    st.info(result['evaluation'])
                    
                    st.divider()
                    
                    # Comparar respostas
                    st.markdown("**📝 Comparação de Respostas:**")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🧠 Resposta do Agente:**")
                        st.markdown(result['agent_answer'])
                    
                    with col2:
                        st.markdown("**✅ Resposta Ideal:**")
                        st.markdown(result['ideal_answer'])
            
            # Resumo geral
            st.subheader("📊 Resumo Geral")
            avg_time = sum(r['response_time'] for r in results) / len(results)
            avg_keywords = sum(r['keywords_found'] / r['total_keywords'] for r in results) / len(results)
            
            # Calcular score médio
            scores = []
            for result in results:
                evaluation_text = result['evaluation']
                if "Score:" in evaluation_text:
                    score = evaluation_text.split("Score:")[1].split("\n")[0].strip()
                    try:
                        scores.append(float(score))
                    except:
                        pass
            
            avg_score = sum(scores) / len(scores) if scores else 0
            
            # Descrição do score médio
            score_descriptions = {
                (0, 0.5): "Baixo",
                (0.5, 1.5): "Regular", 
                (1.5, 2.5): "Bom",
                (2.5, 3.0): "Excelente"
            }
            
            score_category = "N/A"
            for (min_score, max_score), desc in score_descriptions.items():
                if min_score <= avg_score < max_score:
                    score_category = desc
                    break
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("⏱️ Tempo Médio", f"{avg_time:.2f}s")
            with col2:
                st.metric("🎯 Palavras-chave Médias", f"{avg_keywords:.1%}")
            with col3:
                st.metric("📊 Score Médio", f"{avg_score:.1f} - {score_category}")


if __name__ == "__main__":
    main()
