from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document

import streamlit as st


class PDFAgent:
    def __init__(self, api_key: str, model_type: str = "gemini"):
        """Inicializa o agente PDF."""
        self.api_key = api_key
        self.model_type = model_type.lower()
        
        if self.model_type == "gemini":
            self.embeddings = GoogleGenerativeAIEmbeddings(
                google_api_key=api_key,
                model="models/embedding-001"
            )
        elif self.model_type == "openai":
            from langchain_openai import OpenAIEmbeddings
            self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        else:
            raise ValueError("model_type deve ser 'gemini' ou 'openai'")
            
        self.vectorstore = None
        self.documents = []
        
    def load_pdf(self, pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """Carrega e processa um arquivo PDF."""
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        self.documents = text_splitter.split_documents(documents)
        return self.documents
            
    def create_vectorstore(self) -> bool:
        """Cria o vectorstore com os documentos carregados."""
        if not self.documents:
            raise Exception("Nenhum documento carregado.")
        
        self.vectorstore = Chroma.from_documents(
            documents=self.documents,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        return True
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Realiza busca por similaridade no vectorstore."""
        try:
            if not self.vectorstore:
                raise Exception("Vectorstore não criado.")
            
            results = self.vectorstore.similarity_search(query, k=k)
            return results
            
        except Exception as e:
            raise Exception(f"Erro na busca por similaridade: {e}")

def main():
    st.set_page_config(
        page_title="PDF Agent",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("PDF Agent - Busca por Similaridade")
    st.markdown("Faça upload de um PDF e faça perguntas sobre seu conteúdo usando IA!")
    
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
        
        st.subheader("Configurações de Texto")
        
        chunk_size = st.slider(
            "Tamanho do Chunk",
            min_value=500,
            max_value=2000,
            value=1000,
            step=100,
            help="Tamanho de cada pedaço de texto (caracteres)"
        )
        
        chunk_overlap = st.slider(
            "Sobreposição do Chunk",
            min_value=0,
            max_value=500,
            value=200,
            step=50,
            help="Sobreposição entre chunks consecutivos"
        )
        
        k_results = st.slider(
            "Número de Resultados",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            help="Quantidade de resultados a retornar na busca"
        )
        
        st.subheader("Configurações Atuais")
        st.write(f"**Chunk Size:** {chunk_size} caracteres")
        st.write(f"**Chunk Overlap:** {chunk_overlap} caracteres")
        st.write(f"**Número de Resultados:** {k_results}")
        st.write(f"**Modelo:** {model_type.upper()}")
        
        if model_type == "gemini":
            st.info("Pegar API Key: https://makersuite.google.com/app/apikey")
        else:
            st.info("Pegar API Key: https://platform.openai.com/api-keys")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload do PDF")
        uploaded_file = st.file_uploader(
            "Escolha um arquivo PDF",
            type=['pdf'],
            help="Faça upload do PDF que você quer analisar"
        )
    
    with col2:
        st.header("Buscar no PDF")
        query = st.text_area(
            "Digite o texto para buscar no PDF",
            placeholder="Ex: sobre o descomissionamento...",
            height=100
        )
        
        search_button = st.button("Buscar", type="primary")
    
    if search_button and uploaded_file and query and api_key:
        with st.spinner("Processando PDF..."):
            try:
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                agent = PDFAgent(api_key, model_type)
                
                agent.load_pdf("temp.pdf", chunk_size, chunk_overlap)
                
                agent.create_vectorstore()
                
                results = agent.similarity_search(query, k=k_results)
                
                if results:
                    st.success(f"Encontrados {len(results)} resultados!")
                    
                    for i, doc in enumerate(results, 1):
                        with st.expander(f"Resultado {i} (Página {doc.metadata.get('page', 'N/A')})"):
                            st.write(doc.page_content)
                else:
                    st.warning("Nenhum resultado encontrado.")
                    
            except Exception as e:
                st.error(f"Erro: {str(e)}")
    
    elif search_button:
        if not uploaded_file:
            st.error("Por favor, faça upload de um PDF.")
        if not query:
            st.error("Por favor, digite uma pergunta.")
        if not api_key:
            st.error("Por favor, insira sua API key.")

if __name__ == "__main__":
    main()
