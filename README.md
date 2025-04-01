# BasicChat: Your Intelligent Local AI Assistant

## Overview
BasicChat is a privacy-focused AI assistant that runs locally using Ollama. It features RAG (Retrieval Augmented Generation), multi-modal processing, and smart tools - all through a clean Streamlit interface.

![Chat Interface: Clean layout with message history and real-time response indicators](assets/chat-interface.png)

![Latest Interface: Enhanced with RAG-powered document analysis and multi-modal processing](latest-interface.png)

## 🌟 Key Features

### Core Capabilities
- Local LLM integration via Ollama
  - Configurable model selection (Mistral, LLaVA)
  - Streaming responses for real-time interaction
  - Memory-efficient processing
- Advanced context management
  - Long-term conversation memory
  - Dynamic context window optimization
  - Intelligent context pruning
- Multi-modal support
  - Text and document processing
  - Image analysis and understanding
  - Multiple document format support (PDF, TXT, MD)
- RAG-powered document analysis
  - Semantic search capabilities
  - Automatic document chunking
  - Efficient embedding generation
- Vector storage with ChromaDB
  - Fast similarity search
  - Persistent knowledge storage
  - Optimized index management
- Smart system features
  - Comprehensive error handling
  - Custom tool integration
  - Real-time system monitoring

## 🏗️ Architecture

### System Overview
```mermaid
graph TD
    %% Color definitions
    classDef primary fill:#4285f4,stroke:#2956a3,color:white
    classDef secondary fill:#34a853,stroke:#1e7e34,color:white
    classDef accent fill:#ea4335,stroke:#b92d22,color:white
    classDef storage fill:#fbbc05,stroke:#cc9a04,color:black

    A[User Interface]:::primary -->|Query/Input| B[Streamlit App]:::primary
    B -->|Document Upload| C[Document Processor]:::secondary
    B -->|Chat Query| D[Chat Engine]:::accent
    D -->|RAG Query| E[Vector Store]:::storage
    D -->|LLM Request| F[Ollama API]:::accent
    C -->|Embeddings| E
    F -->|Response| D
    D -->|Final Output| B
```
System architecture showing the flow of data through the application's core components.

### Document Processing Pipeline
```mermaid
graph LR
    %% Color definitions
    classDef input fill:#4285f4,stroke:#2956a3,color:white
    classDef process fill:#34a853,stroke:#1e7e34,color:white
    classDef storage fill:#fbbc05,stroke:#cc9a04,color:black
    classDef output fill:#ea4335,stroke:#b92d22,color:white

    A[Document Input]:::input --> B[Text Extraction]:::process
    B --> C[Chunking]:::process
    C --> D[Embedding Generation]:::process
    D --> E[Vector Storage]:::storage
    
    F[Image Input]:::input --> G[LLaVA Analysis]:::process
    G --> H[Feature Extraction]:::process
    H --> E
    
    E --> I[RAG Integration]:::output
    E --> J[Semantic Search]:::output
```
Document and image processing workflow showing how different types of inputs are processed and stored.

### Memory Management System
```mermaid
graph TD
    %% Color definitions
    classDef memory fill:#4285f4,stroke:#2956a3,color:white
    classDef process fill:#34a853,stroke:#1e7e34,color:white
    classDef storage fill:#fbbc05,stroke:#cc9a04,color:black
    classDef output fill:#ea4335,stroke:#b92d22,color:white

    A[Chat History]:::memory --> B{Memory Manager}:::process
    C[Context Window]:::memory --> B
    D[Vector Store]:::storage --> B
    
    B --> E[Short-term Memory]:::memory
    B --> F[Long-term Memory]:::memory
    
    E --> G[Active Context]:::output
    F --> H[Persistent Storage]:::storage
    
    G --> I[Response Generation]:::output
    H --> J[Knowledge Retrieval]:::output
```
Memory management architecture showing how conversation context and knowledge are maintained.

### Model Interaction Flow
```mermaid
graph TD
    %% Color definitions
    classDef model fill:#4285f4,stroke:#2956a3,color:white
    classDef process fill:#34a853,stroke:#1e7e34,color:white
    classDef data fill:#fbbc05,stroke:#cc9a04,color:black
    classDef output fill:#ea4335,stroke:#b92d22,color:white

    A[User Input]:::data --> B{Input Type}
    B -->|Text| C[Mistral]:::model
    B -->|Image| D[LLaVA]:::model
    B -->|Document| E[Text Embeddings]:::model
    
    C --> F[Response Generation]:::process
    D --> F
    E --> G[Vector Database]:::data
    
    G -->|Context| F
    F --> H[Final Output]:::output
```
Model interaction diagram showing how different AI models process various types of inputs.

## 🚀 Quick Start

### Prerequisites
1. Install [Ollama](https://ollama.ai)
2. Python 3.11+
3. Git

### Required Models
```bash
ollama pull mistral        # Core language model
ollama pull nomic-embed-text   # Embedding model
ollama pull llava         # Vision model
```

### Installation
```bash
# Clone repository
git clone https://github.com/khaosans/basic-chat-template.git
cd basic-chat-template

# Set up environment
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the app
streamlit run app.py
```

## 🔧 Troubleshooting
- Ensure Ollama is running (`ollama serve`)
- Check model downloads (`ollama list`)
- Verify port 8501 is available

## 📝 License
MIT License - See LICENSE file for details.
