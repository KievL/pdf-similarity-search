# 🤖 PDF Agent - Busca por Similaridade com LangChain

Um agente inteligente que usa LangChain e vectorstores para processar PDFs e realizar buscas por similaridade usando IA.

## ✨ Funcionalidades

- 📄 **Carregamento de PDFs**: Processa arquivos PDF automaticamente
- 🔍 **Busca por Similaridade**: Encontra conteúdo relevante usando embeddings
- 🧠 **IA Integrada**: Suporte para Gemini (Google) e OpenAI
- 🌐 **Interface Web**: Interface Streamlit amigável para uso
- 📊 **Vectorstore**: Armazena embeddings em ChromaDB para busca rápida

## 🚀 Instalação

1. **Clone o repositório**:
```bash
git clone <seu-repositorio>
cd pdf-similarity-search
```

2. **Instale as dependências**:
```bash
pip install -r requirements.txt
```

## 📖 Como Usar

### 🖥️ Interface Web

Execute o programa com a interface Streamlit:

```bash
streamlit run main.py
```

Isso abrirá uma interface web onde você pode:
- Fazer upload de um PDF
- Escolher entre Gemini ou OpenAI
- Configurar parâmetros de processamento
- Digitar sua pergunta
- Ver os resultados da busca por similaridade

### 🔧 Uso Programático

```python
from main import PDFAgent

# Inicializa o agente com Gemini (padrão)
agent = PDFAgent("sua_google_api_key", "gemini")

# Ou com OpenAI
agent = PDFAgent("sua_openai_api_key", "openai")

# Carrega um PDF
agent.load_pdf("documento.pdf")

# Cria o vectorstore
agent.create_vectorstore()

# Faz uma busca
results = agent.similarity_search("Sua pergunta aqui")

# Exibe os resultados
for doc in results:
    print(f"Página {doc.metadata.get('page')}: {doc.page_content}")
```

## 🛠️ Estrutura do Projeto

```
pdf-similarity-search/
├── main.py                    # Código principal do agente
├── requirements.txt           # Dependências do projeto
├── README.md                 # Este arquivo
└── chroma_db/                # Diretório do vectorstore (criado automaticamente)
```

## 🔧 Configuração

### 📦 Dependências

O projeto utiliza as seguintes dependências principais:

- `langchain`: Framework principal para LLMs
- `langchain-community`: Componentes comunitários do LangChain
- `langchain-openai`: Integração com OpenAI
- `langchain-google-genai`: Integração com Google Gemini
- `chromadb`: Vector database local
- `pypdf`: Processamento de PDFs
- `streamlit`: Interface web
- `openai`: Cliente da API OpenAI
- `google-generativeai`: Cliente da API Google Gemini
- `tiktoken`: Tokenização de texto

### Parâmetros Configuráveis

Na interface web, você pode ajustar:

- **Chunk Size**: Tamanho dos chunks de texto (500-2000 caracteres)
- **Chunk Overlap**: Sobreposição entre chunks (0-500 caracteres)
- **Número de Resultados**: Quantidade de resultados da busca (1-20)


## 🔍 Como Funciona

1. **Carregamento**: O PDF é carregado e dividido em chunks menores
2. **Embeddings**: Cada chunk é convertido em vetores usando Gemini ou OpenAI
3. **Armazenamento**: Os vetores são armazenados no ChromaDB
4. **Busca**: Sua pergunta é convertida em vetor e comparada com os chunks
5. **Resultados**: Os chunks mais similares são retornados


## 🚀 Funcionalidades da Interface

A interface Streamlit oferece:

- **Upload de PDF**: Arraste e solte ou clique para selecionar
- **Configuração de Modelo**: Escolha entre Gemini e OpenAI
- **Configuração de Parâmetros**: Ajuste chunk size, overlap e número de resultados
- **Busca Interativa**: Digite perguntas e veja resultados em tempo real
- **Visualização de Resultados**: Resultados organizados por página com expansores
