# PDF Similarity Search

Sistema de busca por similaridade em documentos PDF usando LangChain, Google Gemini e ChromaDB.

## Pré-requisitos

- Docker e Docker Compose instalados
- Google API Key para Gemini

## Configuração

1. Clone o repositório:
```bash
git clone <repository-url>
cd pdf-similarity-search
```

2. Configure as variáveis de ambiente:
```bash
cp env.example .env
```

3. Edite o arquivo `.env` com suas configurações:
```env
# Google API Key para Gemini
GOOGLE_API_KEY=your_google_api_key_here

# Caminho para os PDFs (apenas para o script de criar os embeddings)
EMBEDDINGS_PDFS_PATH=/app/pdfs

# Configurações do Vectorstore (não mexer)
VECTORSTORE_PATH=./chroma_db

# Configurações de Chunks (para o script de criar os embeddings)
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Configurações de Similarity Search (quantidade inicial do slider de documentos)
K_DOCUMENTS=5
```

## Execução

1. Construa e execute o container:
```bash
docker-compose up --build
```

2. Acesse a aplicação em: http://localhost:8501

## Funcionalidades

- **Chat**: Interface para fazer perguntas sobre os documentos PDF
- **Evaluation**: Avaliação automática da qualidade das respostas
- **Similarity Search**: Busca por documentos relevantes usando embeddings

## Estrutura do Projeto

```
pdf-similarity-search/
├── main.py                 # Aplicação Streamlit
├── create_embeddings.py    # Script para criar embeddings
├── requirements.txt        # Dependências Python
├── docker-compose.yml      # Configuração Docker
├── Dockerfile             # Imagem Docker
├── .env                   # Variáveis de ambiente (criar)
├── env.example            # Exemplo de variáveis
└── chroma_db/             # Database de embeddings (criado automaticamente)
```

## Desenvolvimento Local

Para executar sem Docker:

1. Instale as dependências:
```bash
pip install -r requirements.txt
```

2. Configure o `.env` com o caminho local dos PDFs:
```env
EMBEDDINGS_PDFS_PATH=/caminho/local/para/pdfs
```

3. Execute:
```bash
streamlit run main.py
```
