# app_ollama_rag.py
import os
import time
import pickle
import streamlit as st
from pathlib import Path
from tqdm import tqdm

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS

# === IMPORTS DO OLLAMA / LANGCHAIN-OLLAMA ===
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import ChatPromptTemplate
from markitdown import MarkItDown

# ==============================
# CONFIGURAÇÕES
# ==============================
PDFS_DIR = Path("./pdfs")
VECTORSTORE_PATH = "./chroma_db_ollama"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
DEFAULT_K = 8
BATCH_SIZE = 200  # se quiser indexar em lotes (útil para muitos chunks)


# ==============================
# STREAMLIT UI
# ==============================
st.set_page_config(page_title="PDF Agent (Ollama) - RAG local", layout="wide")
st.title("📑 PDF Agent — RAG local com Ollama (embeddings + chat)")

st.sidebar.markdown("### Configuração Ollama")
embed_model = st.sidebar.selectbox(
    "Modelo de embeddings (instalado no Ollama)",
    options=["nomic-embed-text", "mxbai-embed-large", "all-minilm"],
    index=0,
    help="Modelo usado para gerar embeddings. Baixe-o com `ollama pull <nome-do-modelo>`."
)
chat_model = st.sidebar.selectbox(
    "Modelo de chat (instalado no Ollama)",
    options=["llama3", "llama3.1", "gemma3", "gpt-oss"],
    index=0,
    help="Modelo de geração de texto. Baixe-o com `ollama pull <nome-do-modelo>`."
)
k_results = st.sidebar.number_input("k (número de chunks retornados pelo retriever)", min_value=1, max_value=50, value=DEFAULT_K)
force_reindex = st.sidebar.button("🔁 Recriar vectorstore (forçar reindex)")

st.sidebar.markdown("---")
st.sidebar.markdown("⚠️ **Importante**: antes de rodar este app, instale e execute o Ollama localmente e puxe os modelos desejados. Veja instruções no painel abaixo.")

# ==============================
# FUNÇÕES AUXILIARES
# ==============================
def load_and_split_pdfs(pdf_dir: Path, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Carrega PDFs da pasta e divide em chunks usando MarkItDown."""
    all_chunks = []
    md = MarkItDown()

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
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

            # cria um Document com todo o texto do PDF (vai ser chunked depois)
            docs = [Document(page_content=text, metadata={"source": str(pdf_file), "filename": pdf_file.name})]

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
            )
            chunks = splitter.split_documents(docs)

            for chunk in chunks:
                chunk.metadata['source'] = str(pdf_file)
                chunk.metadata['filename'] = pdf_file.name

            all_chunks.extend(chunks)

    return all_chunks



@st.cache_resource(show_spinner="🔍 Preparando vectorstore local (Ollama embeddings)...")
def build_vectorstore(embed_model: str, force_recreate: bool = False, persist_path="local_vectorstore", batch_size=64):
    """
    Cria (ou continua criando) um vectorstore a partir dos PDFs processados.
    Processa em batches menores para evitar travamento em máquinas com menos CPU/RAM.
    Exibe progresso via Streamlit.
    """
    # MUDANÇA 1: O path agora é um diretório, não um arquivo .pkl
    if not os.path.isdir(persist_path):
        os.makedirs(persist_path, exist_ok=True)

    # Carrega e divide os PDFs em chunks
    docs = load_and_split_pdfs(PDFS_DIR)
    if not docs:
        st.error("Nenhum documento válido foi processado. Coloque PDFs na pasta ./pdfs")
        return None

    embeddings = OllamaEmbeddings(model=embed_model)

    # Se já existe e não for para recriar, carrega o vectorstore
    # MUDANÇA 2: Checa se o diretório e o arquivo de índice existem
    if os.path.exists(os.path.join(persist_path, "index.faiss")) and not force_recreate:
        st.info(f"🔄 Retomando vectorstore existente em `{persist_path}`...")
        # MUDANÇA 3: Usa FAISS.load_local para carregar
        vectorstore = FAISS.load_local(
            persist_path,
            embeddings,
            # Esta flag é necessária por segurança ao desserializar dados
            allow_dangerous_deserialization=True
        )
        processed_docs = len(vectorstore.index_to_docstore_id)
    else:
        if force_recreate and os.path.exists(persist_path):
            import shutil
            shutil.rmtree(persist_path) # Remove o diretório inteiro
            os.makedirs(persist_path, exist_ok=True)
            st.warning("🗑️ Vectorstore anterior removido. Recriando do zero...")
        st.info("🆕 Criando novo vectorstore...")

        # A sua lógica de inicialização é boa, vamos mantê-la
        dummy = ["init"]
        vectorstore = FAISS.from_texts(dummy, embeddings)
        vectorstore.docstore._dict = {}
        vectorstore.index.reset()
        processed_docs = 0

    total_docs = len(docs)
    st.write(f"📊 Total de chunks a processar: {total_docs}")
    st.write(f"✅ Já processados: {processed_docs}")

    remaining_docs = docs[processed_docs:]

    if not remaining_docs:
        st.success("🎉 Vectorstore já está completo e atualizado!")
        return vectorstore

    # Barra de progresso
    progress_bar = st.progress(int(processed_docs / total_docs * 100), text="Indexando documentos...")
    status_text = st.empty()

    for i in range(0, len(remaining_docs), batch_size):
        batch = remaining_docs[i:i+batch_size]

        # Adiciona batch ao vectorstore
        vectorstore.add_documents(batch)
        processed_docs += len(batch)

        # Atualiza progresso visual
        percent = int(processed_docs / total_docs * 100)
        progress_bar.progress(percent, text=f"Indexando documentos... {percent}%")
        status_text.write(f"📥 Processados {processed_docs}/{total_docs} chunks")

        # MUDANÇA 4: Salva o checkpoint usando save_local
        vectorstore.save_local(persist_path)

        time.sleep(0.2) # Evita travar a UI

    progress_bar.progress(100, text="✅ Indexação concluída")
    status_text.write(f"🎉 Vectorstore finalizado e salvo em disco no diretório `{persist_path}`!")

    return vectorstore


def retrieve_relevant_docs(vectorstore, query: str, k: int):
    """Recupera os k documentos mais relevantes e exibe um preview no Streamlit (debug)."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    relevant_docs = retriever.get_relevant_documents(query)

    st.write("🔎 Documentos recuperados (pré-visualização):")
    for doc in relevant_docs:
        filename = doc.metadata.get("filename", "desconhecido")
        preview = doc.page_content[:400].replace("\n", " ") + "..."
        st.write(f"📄 **{filename}** → {preview}")

    return relevant_docs


def get_answer_with_ollama_chat(llm, vectorstore, question: str, k: int):
    # recupera
    relevant_docs = retrieve_relevant_docs(vectorstore, question, k)

    context_text = "\n\n".join([
        f"Fonte: {doc.metadata.get('filename', 'Unknown')}\nTrecho: {doc.page_content}"
        for doc in relevant_docs
    ])

    # prompt direto (vou passar como mensagens: system + human)
    system_prompt = f"""
Você é um assistente especialista em analisar documentos PDF e responder perguntas APENAS com base no contexto fornecido.
Se a informação não estiver no contexto, diga explicitamente que não possui a informação.

Contexto dos documentos:
{context_text}
"""

    # mensagens no formato esperado pelo ChatOllama
    messages = [
        ("system", system_prompt),
        ("human", question),
    ]

    # invoke no ChatOllama (pode retornar diferentes estruturas dependendo da versão)
    response = llm.invoke(messages)

    # extrai texto de resposta de forma robusta
    answer_text = None
    try:
        # se for objeto com .content
        if hasattr(response, "content"):
            answer_text = response.content
        # se for lista (algumas versões retornam lista de mensagens)
        elif isinstance(response, (list, tuple)) and len(response) > 0 and hasattr(response[0], "content"):
            answer_text = response[0].content
        else:
            answer_text = str(response)
    except Exception:
        answer_text = str(response)

    return answer_text, relevant_docs


# ==============================
# FLUXO PRINCIPAL
# ==============================
# instrução para o usuário
st.info("Este app usa **Ollama** localmente (embeddings + chat). Verifique a sidebar para selecionar modelos. Veja instruções no final da página para instalar o Ollama e puxar modelos.")

# build / load vectorstore
vectorstore = build_vectorstore(embed_model, force_recreate=force_reindex)
if vectorstore is None:
    st.stop()

# instancia LLM de chat (Ollama)
llm = ChatOllama(model=chat_model, temperature=0.2)

# Caixa de pergunta
query = st.text_area("❓ Pergunta sobre os PDFs:", placeholder="Ex: Quais materiais podem ser usados na cimentação para poços HPHT?")
if st.button("Buscar resposta") and query.strip():
    with st.spinner("Consultando o LLM local com contexto..."):
        answer, sources = get_answer_with_ollama_chat(llm, vectorstore, query, k_results)

    st.subheader("📌 Resposta")
    st.write(answer)

    if sources:
        st.subheader("📂 Fontes (recuperadas)")
        for i, src in enumerate(sources, 1):
            with st.expander(f"{i}. {src.metadata.get('filename')}"):
                st.write(src.page_content[:1000] + "...")
