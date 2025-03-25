# 🤖 Document-Aware Chatbot

An AI-powered chatbot that processes documents and images using local LLM (Ollama) with RAG capabilities.

## 🏗️ Architecture

```mermaid
graph TD
    subgraph Frontend["Frontend (Streamlit)"]
        UI[Web Interface]
        Upload[Document Upload]
        Chat[Chat Interface]
    end

    subgraph Processing["Document Processing"]
        DP[Document Processor]
        TS[Text Splitter]
        OCR[Tesseract OCR]
    end

    subgraph Storage["Vector Storage"]
        CD[ChromaDB]
    end

    subgraph LLM["Language Models"]
        OL[Ollama - Mistral]
        OE[Ollama Embeddings]
    end

    Upload --> DP
    DP --> |PDFs/Text| TS
    DP --> |Images| OCR
    TS --> OE
    OCR --> TS
    OE --> CD
    Chat --> |Query| CD
    CD --> |Context| OL
    OL --> |Response| Chat
```

## 🔄 Document Processing Flow

```mermaid
sequenceDiagram
    participant User
    participant UI as Streamlit
    participant DP as Doc Processor
    participant DB as ChromaDB
    participant LLM as Ollama

    User->>UI: Upload Document
    UI->>DP: Process File
    alt PDF/Text
        DP->>DP: Split Content
        DP->>DB: Store Chunks
    else Image
        DP->>DP: OCR Text
        DP->>DP: Split Content
        DP->>DB: Store Chunks
    end

    User->>UI: Ask Question
    UI->>DB: Retrieve Context
    DB->>LLM: Provide Context
    LLM->>UI: Generate Response
    UI->>User: Show Answer
```

## 🛠️ Technical Stack

- **Frontend**: Streamlit (^1.24.0)
- **LLM Integration**: 
  - Ollama (local LLM)
  - Model: Mistral
- **Document Processing**:
  - Text Splitting: LangChain RecursiveCharacterTextSplitter
  - PDF Processing: PyPDF
  - Image Processing: Tesseract OCR
- **Vector Storage**: ChromaDB (^0.3.0)
- **Embeddings**: Ollama Embeddings
- **Dependencies**:
  - Python >=3.8.1
  - LangChain ^0.0.330
  - ChromaDB ^0.3.0
  - Streamlit ^1.24.0

## 📝 Features

1. **Document Processing**
   - PDF documents
   - Text files
   - Images (OCR)
   - Chunk optimization for better context

2. **Chat Interface**
   - Real-time responses
   - Document-aware context
   - History tracking
   - Clear conversation option

3. **RAG Implementation**
   - Local embeddings generation
   - Semantic search
   - Context-aware responses
   - Document source tracking

## 🚀 Getting Started

1. **Prerequisites**
```bash
# Install system dependencies
brew install tesseract  # OCR support
brew install poppler   # PDF processing
```

2. **Installation**
```bash
# Install Python dependencies
poetry install

# Run setup
poetry run python setup.py

# Start application
poetry run streamlit run app.py
```

## �� Project Structure
