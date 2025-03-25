# 🤖 RAG-Enabled Local Chatbot

A Retrieval-Augmented Generation (RAG) chatbot using local LLMs and vector storage for document-aware conversations.

## 📸 Screenshots

![Document-Aware Chat Interface](./assets/chat-interface.png)
*The chatbot analyzing a PDF about DeFi and blockchain technology, demonstrating document-aware responses*

## 📚 Example Documents

The repository includes sample documents to test the RAG capabilities:

- [Retrieval-Augmented Generation Paper](./docs/rag-paper.pdf) - The original research paper introducing RAG (Retrieval-Augmented Generation) for enhancing language models with external knowledge

## Key Features
- 📄 Process and understand multiple document formats
- 🔍 Retrieve relevant context from documents
- 💬 Natural conversation with document awareness
- 🏃 Fast local processing with Ollama
- 🔒 Privacy-focused (all data stays local)

## 🏗️ System Architecture

```mermaid
graph TD
    subgraph UserInterface["Streamlit Interface"]
        Upload[Document Upload]
        Chat[Chat Interface]
        History[Chat History]
    end

    subgraph RAGPipeline["RAG Pipeline"]
        subgraph DocumentProcessing["Document Processing"]
            DP[Document Processor]
            TS[Text Splitter]
            OCR[Tesseract OCR]
        end

        subgraph Embeddings["Embedding Layer"]
            OE[Ollama Embeddings]
            VDB[Vector Database]
        end

        subgraph Retrieval["Context Retrieval"]
            CS[Context Search]
            CR[Context Ranking]
        end
    end

    subgraph LLMLayer["Local LLM Layer"]
        OL[Ollama - Mistral]
        PM[Prompt Management]
    end

    Upload --> DP
    DP --> |PDFs/Text| TS
    DP --> |Images| OCR
    OCR --> TS
    TS --> OE
    OE --> VDB
    Chat --> CS
    CS --> VDB
    VDB --> CR
    CR --> PM
    PM --> OL
    OL --> History
```

## 🔄 RAG Implementation Flow

```mermaid
sequenceDiagram
    participant U as User
    participant I as Interface
    participant R as RAG System
    participant L as Local LLM

    U->>I: Upload Document
    I->>R: Process Document
    activate R
    R->>R: Extract Text
    R->>R: Split into Chunks
    R->>R: Generate Embeddings
    R->>R: Store in Vector DB
    deactivate R

    U->>I: Ask Question
    I->>R: Process Query
    activate R
    R->>R: Generate Query Embedding
    R->>R: Search Similar Chunks
    R->>R: Rank Relevance
    R->>R: Build Context
    R->>L: Context + Query
    activate L
    L->>L: Generate Response
    L->>I: Return Answer
    deactivate L
    deactivate R
    I->>U: Display Response
```

## 🛠️ Technical Implementation

### Local Models
- **LLM**: Ollama (Mistral)
  - Local inference
  - No data leaves system
  - Customizable parameters

### RAG Components
1. **Document Processing**
   ```python
   # Text splitting configuration
   text_splitter = RecursiveCharacterTextSplitter(
       chunk_size=1000,
       chunk_overlap=200,
       length_function=len,
   )
   ```

2. **Embedding Generation**
   ```python
   embeddings = OllamaEmbeddings(
       model="nomic-embed-text",
       base_url="http://localhost:11434"
   )
   ```

3. **Vector Storage**
   ```python
   vectorstore = Chroma(
       persist_directory="./chroma_db",
       embedding_function=embeddings
   )
   ```

### Supported Formats
- 📄 PDF Documents
- 📝 Text Files
- 🖼️ Images (OCR-enabled)
- 📊 Markdown Files

## 🚀 Quick Start

1. **System Requirements**
```bash
# Core dependencies
brew install ollama
brew install tesseract
```

2. **Environment Setup**
```bash
# Initialize project
poetry install

# Run setup script
poetry run python setup.py

# Start Ollama
ollama serve
```

3. **Launch Application**
```bash
poetry run streamlit run app.py
```

## 🔧 Configuration

### Environment Variables
```env
OLLAMA_BASE_URL=http://localhost:11434
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### LLM Settings
```python
llm = ChatOllama(
    model="mistral",
    temperature=0.7,
    base_url="http://localhost:11434"
)
```

## 📊 Performance Considerations

1. **Memory Usage**
   - Vector DB scaling
   - Document chunk size
   - Embedding cache

2. **Processing Speed**
   - OCR optimization
   - Batch processing
   - Concurrent operations

3. **Response Quality**
   - Context window size
   - Chunk overlap
   - Relevance threshold

## 🔍 Debugging

```bash
# Check Ollama status
curl http://localhost:11434/api/version

# Verify vector store
poetry run python -c "import chromadb; print(chromadb.__version__)"

# Test OCR
poetry run python -c "import pytesseract; print(pytesseract.get_tesseract_version())"
```

## 🐛 Known Issues

1. **Image Processing**
   - OCR quality varies with image clarity
   - Large images may require preprocessing
   - PNG transparency can affect OCR

2. **Vector Storage**
   - ChromaDB requires periodic optimization
   - Large collections need index management
   - Memory usage scales with document count

## 🔒 Security

- All processing done locally
- No external API calls
- Data remains on system
- Configurable access controls

## 📚 References

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Ollama GitHub](https://github.com/ollama/ollama)
- [ChromaDB Documentation](https://docs.trychroma.com/)
