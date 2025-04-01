# BasicChat: Your Intelligent Local AI Assistant

## Project Overview
BasicChat is a powerful, privacy-focused AI assistant that combines local language models with advanced features like RAG (Retrieval Augmented Generation), multi-modal processing, and intelligent tools. Built on Streamlit and powered by Ollama, it offers secure offline operation while delivering sophisticated capabilities including document analysis, image processing, and context-aware conversations - all with a modern, intuitive interface.

![Chat Interface: The original interface showcases our commitment to simplicity and usability, featuring a clean layout with intuitive message history, smart input suggestions, and real-time response indicators - demonstrating how we prioritize user experience from day one](assets/chat-interface.png)

![Latest Interface: Our evolved interface represents a quantum leap in functionality, featuring seamless integration of RAG-powered document analysis, multi-modal processing with image understanding, and intelligent tool orchestration - all while maintaining the clean, user-friendly design that our users love](latest-interface.png)



## 🌟 Key Features

### 🤖 Core Chat Functionality
- **Local LLM Integration**: Powered by Ollama for privacy-focused, offline-capable AI interactions
- **Context-Aware Responses**: Maintains conversation history for coherent dialogue
- **Multi-Modal Support**: Handles text, documents, and images seamlessly

### 📚 Document Processing
- **PDF Processing**: Extract and analyze content from PDF documents
- **Image Analysis**: Process and understand images using LLaVA model
- **RAG Implementation**: Enhances responses with relevant document context
- **Efficient Storage**: ChromaDB for vector storage and quick retrieval

### 🛠️ Smart Tools
- **Date Tool**: Intelligent date-related queries and calculations
- **Time Tool**: Timezone-aware time operations and conversions
- **Document Tool**: Smart document summarization and analysis

### 🎯 User Experience
- **Clean Interface**: Streamlit-powered UI for intuitive interactions
- **Text-to-Speech**: Audio output with playback controls
- **Real-time Processing**: Fast response times with local processing
- **File Management**: Automatic cleanup of temporary files

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

    %% Tooltips
    click A "Primary user interaction point"
    click B "Core application logic"
    click C "Handles document processing and analysis"
    click D "Manages chat interactions and responses"
    click E "ChromaDB for vector storage"
    click F "Local LLM integration"
```

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

### Tool Integration Flow
```mermaid
graph TD
    %% Color definitions
    classDef input fill:#4285f4,stroke:#2956a3,color:white
    classDef process fill:#34a853,stroke:#1e7e34,color:white
    classDef tool fill:#fbbc05,stroke:#cc9a04,color:black
    classDef output fill:#ea4335,stroke:#b92d22,color:white

    A[User Query]:::input -->|Parse| B[Tool Registry]:::process
    B -->|Match| C{Tool Selection}:::process
    C -->|Date Query| D[Date Tool]:::tool
    C -->|Time Query| E[Time Tool]:::tool
    C -->|Document Query| F[Document Tool]:::tool
    D -->|Format| G[Response Handler]:::output
    E -->|Format| G
    F -->|Format| G
    G -->|Display| H[User Interface]:::output

    %% Tooltips
    click D "Handles date-related operations"
    click E "Manages timezone conversions"
    click F "Processes document operations"
    click G "Formats tool outputs"
```

## 🚀 Getting Started

## 💻 System Requirements

### Minimum Requirements
- **CPU**: 4 cores (8 recommended for better performance)
- **RAM**: 8GB (16GB recommended)
- **Storage**: 10GB free space
- **OS**: macOS 10.15+, Ubuntu 20.04+, or Windows 10/11
- **Python**: 3.11 or higher
- **GPU**: Optional but recommended for faster processing

### Prerequisites
1. Install [Ollama](https://ollama.ai) following the official guide for your OS
2. Install Python 3.11+ from [python.org](https://python.org) or using your OS package manager
3. Install Git from [git-scm.com](https://git-scm.com)

### Required Models
Pull these models using Ollama:
```bash
# Core language model
ollama pull mistral

# Embedding model for document processing
ollama pull nomic-embed-text

# Vision model for image analysis
ollama pull llava
```

## 🚀 Installation Guide

### Using pip (Recommended)
```bash
# Clone the repository
git clone https://github.com/yourusername/basic-chat2.git
cd basic-chat2

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the setup script
python setup.py
```

### Using Poetry (Alternative)
```bash
# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Clone and setup
git clone https://github.com/yourusername/basic-chat2.git
cd basic-chat2
poetry install
```

### Build and Verify
We provide an automated build script that handles dependency installation, code quality checks, and testing:

```bash
# Make the build script executable
chmod +x build.sh

# Run the build process
./build.sh
```

The build script performs the following steps:
1. Cleans up previous build artifacts
2. Installs project dependencies
3. Runs type checking with MyPy
4. Formats code using Black and isort
5. Executes test suite with pytest

### Launch the Application
```bash
streamlit run app.py
```
The application will be available at `http://localhost:8501`

## 🔧 Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   ```
   Solution: Ensure Ollama is running with 'ollama serve'
   ```

2. **Model Download Issues**
   ```
   Solution: Check internet connection and retry with 'ollama pull [model]'
   ```

3. **Memory Issues**
   ```
   Solution: Close other applications or increase swap space
   ```

4. **Port Conflicts**
   ```
   Solution: Kill process using port 8501 or specify different port:
   streamlit run app.py --server.port [PORT]
   ```

## 🛣️ Roadmap

### Upcoming Features
- [ ] Multi-model switching capability
- [ ] Advanced memory management for longer conversations
- [ ] Custom tool creation interface
- [ ] Enhanced document processing with more formats
- [ ] Collaborative chat sessions
- [ ] API endpoint for headless operation

### Performance Improvements
- [ ] Optimized vector storage and retrieval
- [ ] Improved context window management
- [ ] Better prompt engineering for RAG

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Guidelines
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments
- Ollama team for the amazing local LLM runtime
- Streamlit team for the powerful UI framework
- All contributors and users of this project
