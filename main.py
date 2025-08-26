import os
import time
from pathlib import Path
import streamlit as st

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from markitdown import MarkItDown


# ==============================
# CONFIGURAÇÕES
# ==============================
PDFS_DIR = Path("./pdfs")  # pasta de PDFs
VECTORSTORE_PATH = "./chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
K_ = 5
BATCH_SIZE = 50  # controla quantos chunks são enviados por vez (ajuste conforme sua quota)


# ==============================
# STREAMLIT APP
# ==============================
st.set_page_config(page_title="PDF Agent", layout="wide")
st.title("📑 PDF Agent - RAG com LangChain + MarkItDown")

api_key = st.text_input("🔑 Google API Key:", type="password")

if not api_key:
    st.warning("Insira sua API Key para continuar.")
    st.stop()


# ==============================
# FUNÇÕES AUXILIARES
# ==============================
def load_and_split_pdfs(pdf_dir: Path):
    """Carrega e divide PDFs em chunks usando MarkItDown + RecursiveSplitter"""
    all_chunks = []
    md = MarkItDown()

    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        st.error("Nenhum PDF encontrado na pasta ./pdfs")
        return []

    with st.spinner("Carregando e processando PDFs..."):
        for pdf_file in pdf_files:
            st.write(f"📄 Processando: `{pdf_file.name}`")

            try:
                result = md.convert(str(pdf_file))
                text = result.text_content
            except Exception as e:
                st.error(f"Erro ao processar {pdf_file.name}: {e}")
                continue

            documents = [Document(page_content=text, metadata={"source": str(pdf_file), "filename": pdf_file.name})]

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
            )
            chunks = splitter.split_documents(documents)

            for chunk in chunks:
                chunk.metadata['source'] = str(pdf_file)
                chunk.metadata['filename'] = pdf_file.name

            all_chunks.extend(chunks)

    return all_chunks


@st.cache_resource(show_spinner="🔍 Preparando Vectorstore...")
def build_vectorstore(api_key: str):
    """
    Cria o vectorstore a partir dos PDFs, processando em lotes de 100
    documentos por minuto para respeitar os limites da API gratuita.
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key=api_key,
        model="models/embedding-001"
    )

    docs = load_and_split_pdfs(PDFS_DIR)
    if not docs:
        st.error("Nenhum documento válido encontrado em ./pdfs")
        return None

    st.info(f"📝 Criando novo vectorstore com {len(docs)} chunks...")

    # Inicializa o ChromaDB. Ele será populado de forma incremental no loop abaixo.
    vectorstore = Chroma(
        persist_directory=VECTORSTORE_PATH,
        embedding_function=embeddings
    )

    # Itera sobre os documentos em lotes de BATCH_SIZE
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i:i + BATCH_SIZE]
        
        try:
            # Adiciona o lote atual de documentos ao vectorstore
            vectorstore.add_documents(batch)
            processed_count = i + len(batch)
            st.write(f"✅ Lote processado: {processed_count}/{len(docs)} chunks...")

        except Exception as e:
            st.error(f"⚠️ Erro ao processar o lote: {e}")
            st.warning("Aguardando 60 segundos para tentar novamente...")
            time.sleep(60) # Espera a cota resetar
            try:
                # Tenta adicionar o mesmo lote novamente
                vectorstore.add_documents(batch)
                processed_count = i + len(batch)
                st.write(f"✅ Lote processado com sucesso após nova tentativa.")
            except Exception as e2:
                st.error(f"❌ Falha definitiva no lote. Abortando. Erro: {e2}")
                # Retorna o que foi processado até o momento
                return vectorstore

        # --- Ponto Chave da Lógica ---
        # Se este NÃO for o último lote, faz uma pausa de 60 segundos.
        # Isso garante que você não envie mais de 100 documentos por minuto.
        if i + BATCH_SIZE < len(docs):
            st.write("⏳ Aguardando 60 segundos para respeitar o limite da API...")
            time.sleep(60)

    st.success("✅ Vectorstore criado e persistido com sucesso!")
    return vectorstore


def get_answer(vectorstore, llm, query: str, k: int = K_):
    relevant_docs = vectorstore.similarity_search(query, k=k)

    context_text = "\n\n".join([
        f"Fonte: {doc.metadata.get('filename', 'Unknown')}\nConteúdo: {doc.page_content}"
        for doc in relevant_docs
    ])

    system_prompt = """
    Você é um assistente especializado em analisar documentos PDF. 
    Responda APENAS com base no contexto fornecido.

    Regras:
    1. Não invente nada fora do contexto.
    2. Seja preciso e direto.
    3. Cite as fontes.
    4. Responda em português brasileiro.
    5. Se houver conflito, mencione.
    6. Estruture a resposta claramente.

    Contexto dos documentos:
    {context}

    Pergunta do usuário: {question}
    """

    prompt = ChatPromptTemplate.from_template(system_prompt)

    messages = prompt.format_messages(
        context=context_text,
        question=query
    )

    response = llm.invoke(messages)
    return response, relevant_docs


# ==============================
# PIPELINE PRINCIPAL
# ==============================
vectorstore = build_vectorstore(api_key)

if vectorstore is None:
    st.stop()

llm = ChatGoogleGenerativeAI(
    google_api_key=api_key,
    model="models/gemini-2.5-flash",
    temperature=0.3
)

query = st.text_area("❓ Pergunta:", placeholder="Digite sua pergunta sobre os PDFs...")
if st.button("Buscar resposta") and query.strip():
    with st.spinner("Consultando..."):
        answer, sources = get_answer(vectorstore, llm, query)

    st.subheader("📌 Resposta")
    st.write(answer.content)

    if sources:
        st.subheader("📂 Fontes")
        for i, src in enumerate(sources, 1):
            with st.expander(f"{i}. {src.metadata.get('filename')}"):
                st.write(src.page_content[:500] + "...")
