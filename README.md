# 🤖 PDF Agent - Busca por Similaridade com LangChain

Um agente inteligente que usa LangChain e vectorstores para processar PDFs e realizar buscas por similaridade usando IA.

## ✨ Funcionalidades

- 📄 **Carregamento de PDFs**: Processa arquivos PDF automaticamente
- 🔍 **Busca por Similaridade**: Encontra conteúdo relevante usando embeddings
- 🧠 **IA Integrada**: Suporte para Gemini (Google) e OpenAI
- 🌐 **Interface Web**: Interface Streamlit amigável para uso
- 💻 **CLI**: Modo linha de comando para automação
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

3. **Configure a API Key**:
```bash
# Para Gemini (recomendado)
export GOOGLE_API_KEY="sua_chave_gemini"

# Para OpenAI (alternativa)
export OPENAI_API_KEY="sua_chave_openai"
```

**Obtenha sua chave Gemini gratuitamente em:** https://makersuite.google.com/app/apikey

## 📖 Como Usar

### 🖥️ Interface Web (Recomendado)

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

### 💻 Linha de Comando

Para usar via linha de comando:

```bash
# Busca básica com Gemini (padrão)
python main.py documento.pdf "Qual é o tema principal?"

# Busca com Gemini explicitamente
python main.py documento.pdf "Quais são os métodos utilizados?" --model gemini

# Busca com OpenAI
python main.py documento.pdf "Quais são os métodos utilizados?" --model openai
```

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

### Variáveis de Ambiente

- `GOOGLE_API_KEY`: Sua chave da API Google Gemini (recomendado)
- `OPENAI_API_KEY`: Sua chave da API OpenAI (alternativa)

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

## 📋 Exemplos de Uso

### Exemplo 1: Análise de Documento Acadêmico
```bash
streamlit run main.py
# Faça upload de paper.pdf e pergunte: "Quais são as principais conclusões do estudo?"
```

### Exemplo 2: Busca em Manual Técnico
```bash
streamlit run main.py
# Faça upload de manual.pdf e pergunte: "Como configurar o sistema?"
```

### Exemplo 3: Análise de Relatório
```bash
streamlit run main.py
# Faça upload de relatorio.pdf e pergunte: "Quais são os resultados financeiros?"
```

## 🔍 Como Funciona

1. **Carregamento**: O PDF é carregado e dividido em chunks menores
2. **Embeddings**: Cada chunk é convertido em vetores usando Gemini ou OpenAI
3. **Armazenamento**: Os vetores são armazenados no ChromaDB
4. **Busca**: Sua pergunta é convertida em vetor e comparada com os chunks
5. **Resultados**: Os chunks mais similares são retornados

## 🐛 Solução de Problemas

### Erro: "GOOGLE_API_KEY não encontrada"
```bash
export GOOGLE_API_KEY="sua_chave_gemini"
```
**Obtenha sua chave gratuitamente em:** https://makersuite.google.com/app/apikey

### Erro: "OPENAI_API_KEY não encontrada"
```bash
export OPENAI_API_KEY="sk-sua_chave_aqui"
```

### Erro: "Arquivo PDF não encontrado"
Verifique se o caminho do arquivo está correto e se o arquivo existe.

### Erro: "Erro ao carregar o PDF"
Certifique-se de que o arquivo é um PDF válido e não está corrompido.

### Erro: "Streamlit não encontrado"
```bash
pip install streamlit
```

## 🚀 Funcionalidades da Interface

A interface Streamlit oferece:

- **Upload de PDF**: Arraste e solte ou clique para selecionar
- **Configuração de Modelo**: Escolha entre Gemini e OpenAI
- **Configuração de Parâmetros**: Ajuste chunk size, overlap e número de resultados
- **Busca Interativa**: Digite perguntas e veja resultados em tempo real
- **Visualização de Resultados**: Resultados organizados por página com expansores

## 📝 Licença

Este projeto está sob a licença MIT.

## 🤝 Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests.

## 📞 Suporte

Se você encontrar algum problema ou tiver dúvidas, abra uma issue no repositório.
