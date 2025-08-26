import os
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Carregar variáveis de ambiente
load_dotenv()

# Configurações
EMBEDDINGS_PDFS_PATH = os.getenv("EMBEDDINGS_PDFS_PATH", "/home/kiev/Documents/DECOMPilot/EmbeddingsSource")
API_KEY = os.getenv("GOOGLE_API_KEY")
VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "./chroma_db")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
JSON_OUTPUT_PATH = "./pdf_processing_info.json"


def load_pdf(pdf_path: str):
    print(f"PDF: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"\tCarregado ({len(documents)} páginas)")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"\tCHUNKS: {len(chunks)}\n")
    
    pdf_info = {
        "filename": os.path.basename(pdf_path),
        "full_path": pdf_path,
        "total_pages": len(documents),
        "total_chunks": len(chunks),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }
    
    for i, chunk in enumerate(chunks):
        chunk.metadata['source'] = pdf_path
        chunk.metadata['filename'] = os.path.basename(pdf_path)
    
    return chunks, pdf_info


def create_embeddings():
    directory = Path(EMBEDDINGS_PDFS_PATH)
    pdf_files = list(directory.glob("*.pdf"))

    if not pdf_files:
        print(f"Nenhum PDF em {EMBEDDINGS_PDFS_PATH}")
        return None

    print(f"{len(pdf_files)} PDFs")

    all_documents = []
    processing_info = {
        "processing_date": datetime.now().isoformat(),
        "config": {
            "embeddings_pdfs_path": EMBEDDINGS_PDFS_PATH,
            "vectorstore_path": VECTORSTORE_PATH,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP
        },
        "pdfs": [],
    }

    for pdf_file in pdf_files:
        try:
            documents, pdf_info = load_pdf(str(pdf_file))
            all_documents.extend(documents)
            processing_info["pdfs"].append(pdf_info)
        except Exception as e:
            print(f"Erro {pdf_file}: {e}")

    print("PDFs carregados.")

    with open(JSON_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(processing_info, f, indent=2, ensure_ascii=False)

    print(f"\nCriando Vectorstore com {len(all_documents)} documentos")

    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=API_KEY,
            model="models/embedding-001"
        )

        vectorstore = Chroma.from_documents(
            documents=all_documents,
            embedding=embeddings,
            persist_directory=VECTORSTORE_PATH
        )

        print(f"Vectorstore criado")
        return vectorstore
        
    except Exception as e:
        print(f"{e}")


if __name__ == "__main__":
    vectorstore = create_embeddings()
