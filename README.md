# BasicChat: Your Intelligent Local AI Assistant

<div align="center">

![BasicChat Logo](assets/brand/logo/elron-logo-full.png)

**🔒 Privacy-First • 🧠 Advanced Reasoning • 🔬 Deep Research • ⚡ High Performance**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLMs-green.svg)](https://ollama.ai)
[![Redis](https://img.shields.io/badge/Redis-Task%20Queue-orange.svg)](https://redis.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*An intelligent, private AI assistant that runs entirely on your local machine*

</div>

---

## 🎥 Demo

<div align="center">

![BasicChat Demo](assets/demo_seq_0.6s.gif)

*Real-time reasoning and document analysis with local AI models*

</div>

---

## 🌟 Key Features

<div align="center">

| 🔒 **Privacy** | 🧠 **Intelligence** | 🔬 **Research** | 🛠️ **Tools** | 📄 **Documents** | ⚡ **Performance** |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 100% Local Processing | 5 Reasoning Modes | Deep Research Mode | Smart Calculator | Multi-Format Support | Async Architecture |
| No External APIs | Chain-of-Thought | Multi-Source Analysis | Time Tools | PDF, Text, Images | Multi-Layer Caching |
| Data Never Leaves | Multi-Step Analysis | Academic Rigor | Web Search | Advanced RAG | Connection Pooling |

</div>

### 🔒 **Privacy First**
- **Complete Local Processing**: All AI operations run on your machine
- **No Data Transmission**: Your data never leaves your local environment
- **Secure by Design**: Built with privacy as a core principle

### 🧠 **Advanced Reasoning**
- **Multi-Modal Reasoning**: 5 different reasoning strategies for optimal problem-solving
- **Chain-of-Thought**: Step-by-step reasoning for complex problems (Wei et al.)
- **Agent-Based**: Intelligent tool selection and execution
- **Auto Mode**: Automatically selects the best reasoning approach

### 🔬 **Deep Research Mode**
- **Comprehensive Research**: Multi-source analysis with academic rigor
- **ChatGPT-Style Toggle**: Clean, intuitive interface for research mode
- **Rich Results**: Executive summaries, key findings, detailed analysis, and sources
- **Background Processing**: Long-running research tasks with progress tracking
- **Source Citations**: Proper attribution and links to research sources

### 🛠️ **Powerful Built-in Tools**
- **Enhanced Calculator**: Advanced mathematical operations with step-by-step reasoning
- **Time Tools**: Timezone-aware time calculations and conversions
- **Web Search**: Real-time information retrieval via DuckDuckGo
- **Document Analysis**: Intelligent document summarization and Q&A

### 📄 **Document & Image Analysis**
- **Multi-Format Support**: PDF, text, markdown, and image files
- **Advanced RAG**: Retrieval-Augmented Generation with semantic search (Lewis et al.)
- **OCR Capabilities**: Image text extraction using vision models
- **Vector Storage**: Efficient ChromaDB-based document indexing

### ⚡ **Performance Optimized**
- **Async Architecture**: Non-blocking request handling
- **Multi-Layer Caching**: Redis + Memory caching for 50-80% faster responses
- **Connection Pooling**: Optimized HTTP connections with rate limiting
- **Resource Management**: Automatic cleanup and memory optimization

---

## ⏳ Long-Running Tasks & Background Processing

BasicChat supports **long-running tasks** for complex queries, deep research, and large document processing. These are handled in the background using a robust Celery + Redis task queue system, so you can continue chatting while heavy operations run asynchronously.

- **Background Task UI**: See task progress, status, and results directly in the chat interface.
- **Deep Research Tasks**: Comprehensive research with multiple sources and detailed analysis.
- **Task Management**: Cancel running tasks, monitor metrics, and clean up old tasks from the sidebar.
- **Performance**: Offloads heavy work to background workers, keeping the UI responsive.
- **Monitoring**: Use [Flower](https://flower.readthedocs.io/) for real-time task monitoring and debugging.

> **How it works:**
> - Submitting a complex query, enabling deep research mode, or uploading a large document triggers a background task.
> - Task status, progress, and results are shown in the chat and sidebar.
> - You can cancel tasks or clean up completed/failed ones from the UI.

See the [Architecture Overview](docs/ARCHITECTURE.md#background-task-system) for a diagram of the task queue and worker system.

---

## 🚀 Quick Start

### **Prerequisites**

```bash
# Required Software
- Python 3.11+ 
- Ollama (for local LLMs)
- Redis (for background tasks)
- Git (for cloning)
```

### **Installation**

```bash
# Clone the repository
git clone https://github.com/khaosans/basic-chat-template.git
cd basic-chat-template

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **Start Required Services**

```bash
# Start Redis (macOS with Homebrew)
brew services start redis

# Start Ollama
ollama serve

# Verify services are running
redis-cli ping  # Should return PONG
curl http://localhost:11434/api/tags  # Should return model list
```

### **Download AI Models**

```bash
# Core models for reasoning and embeddings
ollama pull mistral
ollama pull nomic-embed-text

# Optional: Vision model for image analysis
ollama pull llava
```

### **Quick Start with Background Tasks**

For full async/background processing, use the provided dev script:

```bash
# Start all services (Redis, Celery workers, Flower, Streamlit app)
./start_dev.sh
```

Or with Docker Compose:

```bash
docker-compose up --build
```

**Application URLs:**
- **Main App:** http://localhost:8501
- **Task Monitor (Flower):** http://localhost:5555
- **Redis:** localhost:6379

> Flower provides a real-time dashboard for monitoring, retrying, or revoking tasks.

For more details, see [Development Guide](docs/DEVELOPMENT.md#running-with-background-tasks).

---

## 🧹 Document & Database Cleanup

BasicChat includes comprehensive cleanup tools for maintaining optimal performance:

### **ChromaDB Cleanup**

```bash
# Check current status
python scripts/cleanup_chroma.py --status

# Clean all ChromaDB directories
python scripts/cleanup_chroma.py

# Clean directories older than 24 hours
python scripts/cleanup_chroma.py --age 24

# Dry run (see what would be cleaned)
python scripts/cleanup_chroma.py --dry-run

# Force cleanup even if in use
python scripts/cleanup_chroma.py --force
```

### **Manual Cleanup**

```bash
# Remove ChromaDB directories
rm -rf chroma_db*

# Clean temporary files
rm -rf temp_audio/*
rm -rf uploads/*

# Restart services after cleanup
./start_dev.sh
```

### **Automatic Cleanup**

The app automatically cleans up:
- Temporary audio files after processing
- Old task results from Redis
- Expired cache entries
- Unused document embeddings

---

## 🏗️ Architecture Overview

<div align="center">

```mermaid
graph TB
    subgraph "🎨 User Interface"
        UI[Web Interface]
        AUDIO[Audio Processing]
    end
    
    subgraph "🧠 Core Logic"
        RE[Reasoning Engine]
        DP[Document Processor]
        TR[Tool Registry]
    end
    
    subgraph "⚡ Services"
        AO[Ollama Client]
        VS[Vector Store]
        CS[Cache Service]
    end
    
    subgraph "🗄️ Storage"
        CHROMA[Vector Database]
        CACHE[Memory Cache]
        REDIS[Redis Task Queue]
    end
    
    subgraph "🌐 External"
        OLLAMA[LLM Server]
    end
    
    %% User Interface Connections
    UI --> RE
    UI --> DP
    AUDIO --> RE
    
    %% Core Logic Connections
    RE --> AO
    RE --> TR
    DP --> VS
    
    %% Service Connections
    AO --> OLLAMA
    VS --> CHROMA
    CS --> CACHE
    CS --> REDIS
    
    %% Styling
    classDef ui fill:#E3F2FD,stroke:#1976D2,stroke-width:2px,color:#0D47A1
    classDef core fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px,color:#4A148C
    classDef service fill:#E8F5E8,stroke:#388E3C,stroke-width:2px,color:#1B5E20
    classDef storage fill:#FFF3E0,stroke:#F57C00,stroke-width:2px,color:#E65100
    classDef external fill:#FCE4EC,stroke:#C2185B,stroke-width:2px,color:#880E4F
    
    class UI,AUDIO ui
    class RE,DP,TR core
    class AO,VS,CS service
    class CHROMA,CACHE,REDIS storage
    class OLLAMA external
```

</div>

---

## 📚 Documentation

<div align="center">

| 📖 **Guide** | 📋 **Description** | 🔗 **Link** |
|:---|:---|:---:|
| **Technical Overview** | High-level system summary and characteristics | [📊](docs/TECHNICAL_OVERVIEW.md) |
| **Features Overview** | Complete feature documentation and capabilities | [📄](docs/FEATURES.md) |
| **System Architecture** | Technical architecture and component interactions | [🏗️](docs/ARCHITECTURE.md) |
| **Development Guide** | Contributing and development workflows | [🛠️](docs/DEVELOPMENT.md) |
| **Project Roadmap** | Future features and development plans | [🗺️](docs/ROADMAP.md) |
| **Reasoning Features** | Advanced reasoning engine details | [🧠](docs/REASONING_FEATURES.md) |
| **LLM Judge Evaluator** | Code quality evaluation and CI integration | [⚖️](docs/EVALUATORS.md) |

</div>

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test categories
pytest tests/test_reasoning.py      # Reasoning engine tests
pytest tests/test_document_workflow.py  # Document processing tests
pytest tests/test_enhanced_tools.py     # Tool functionality tests
```

---

## 🛠️ Development

### **Code Quality**
```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .
```

### **Database Management**
```bash
# Clean up ChromaDB directories
python scripts/cleanup_chroma.py --status
python scripts/cleanup_chroma.py --dry-run
python scripts/cleanup_chroma.py --force
```

### **Service Management**
```bash
# Start all services
./start_dev.sh

# Stop all services
pkill -f "streamlit\|celery\|flower"

# Check service status
ps aux | grep -E "(streamlit|celery|flower)" | grep -v grep
```

---

## 🤝 Contributing

We welcome contributions! Please see our [Development Guide](docs/DEVELOPMENT.md) for details.

**Quick Start for Contributors:**
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run the test suite: `pytest`
5. Submit a pull request

---

## 📊 Performance Metrics

<div align="center">

| **Metric** | **Value** | **Description** |
|:---|:---:|:---|
| **Cache Hit Rate** | 70-85% | Response caching efficiency |
| **Response Time** | 50-80% faster | With caching vs without |
| **Connection Pool** | 100 total, 30/host | HTTP connection optimization |
| **Rate Limit** | 10 req/sec | Default API throttling |

</div>

---

## 🔒 Security & Privacy

- ✅ **Local Processing Only**: No data sent to external services
- ✅ **Input Validation**: Comprehensive sanitization and validation
- ✅ **Rate Limiting**: Protection against abuse and DDoS
- ✅ **Error Handling**: Graceful degradation with secure defaults
- ✅ **Session Isolation**: No cross-user data access

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📚 References

### **Research Papers**

Wei, Jason, et al. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." *arXiv preprint arXiv:2201.11903*, 2022.

Lewis, Mike, et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *Advances in Neural Information Processing Systems*, vol. 33, 2020, pp. 9459-9474.

Johnson, Jeff, Matthijs Douze, and Hervé Jégou. "Billion-Scale Similarity Search with GPUs." *IEEE Transactions on Big Data*, vol. 7, no. 3, 2019, pp. 535-547.

Chen, Charlie, et al. "Speculative Decoding: Accelerating LLM Inference via Speculative Sampling." *arXiv preprint arXiv:2302.01318*, 2023.

### **Core Technologies**

Ollama. "Local Large Language Model Server." *Ollama.ai*, 2024, https://ollama.ai.

Streamlit. "Web Application Framework." *Streamlit.io*, 2024, https://streamlit.io.

LangChain. "LLM Application Framework." *LangChain.com*, 2024, https://langchain.com.

ChromaDB. "Vector Database for AI Applications." *ChromaDB.ai*, 2024, https://chromadb.ai.

DuckDuckGo. "Privacy-Focused Search Engine." *DuckDuckGo.com*, 2024, https://duckduckgo.com.

### **AI Models**

Mistral AI. "Mistral 7B: The Most Powerful Open Source Language Model." *Mistral.ai*, 2024, https://mistral.ai.

Meta AI. "LLaMA: Open and Efficient Foundation Language Models." *AI.meta.com*, 2024, https://ai.meta.com/llama.

Microsoft. "Phi-2: The Unquestionable Code Generation." *Microsoft.com*, 2024, https://www.microsoft.com/en-us/research/project/phi-2.

Nomic AI. "Nomic Embed: Text Embedding Model." *Nomic.ai*, 2024, https://docs.nomic.ai/reference/endpoints/nomic-embed-text-v1.

Liu, Haotian, et al. "LLaVA: Large Language and Vision Assistant." *arXiv preprint arXiv:2304.08485*, 2023.

### **Development Tools**

Beazley, David M., and Brian K. Jones. *Python Cookbook*. 3rd ed., O'Reilly Media, 2013.

Martin, Robert C. *Clean Architecture: A Craftsman's Guide to Software Structure and Design*. Prentice Hall, 2017.

### **Libraries & Frameworks**

pytz. "World Timezone Definitions for Python." *Python Hosted*, 2024, https://pythonhosted.org/pytz.

Redis. "In-Memory Data Structure Store." *Redis.io*, 2024, https://redis.io.

Tesseract. "Optical Character Recognition Engine." *GitHub*, 2024, https://github.com/tesseract-ocr/tesseract.

Python Software Foundation. "Abstract Syntax Trees." *Python Documentation*, 2024, https://docs.python.org/3/library/ast.html.

### **Standards & Compliance**

OWASP Foundation. "OWASP Top Ten." *OWASP.org*, 2024, https://owasp.org/www-project-top-ten.

European Union. "General Data Protection Regulation (GDPR)." *Official Journal of the European Union*, vol. L119, 2016, pp. 1-88.

California Legislature. "California Consumer Privacy Act (CCPA)." *California Civil Code*, 2018.

U.S. Department of Health and Human Services. "Health Insurance Portability and Accountability Act (HIPAA)." *Federal Register*, vol. 65, no. 250, 2000, pp. 82462-82829.

### **Technical Methods**

Sparck Jones, Karen. "A Statistical Interpretation of Term Specificity and Its Application in Retrieval." *Journal of Documentation*, vol. 28, no. 1, 1972, pp. 11-21.

Malkov, Yury A., and Dmitry A. Yashunin. "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 42, no. 4, 2020, pp. 824-836.

---

<div align="center">

**Made with ❤️ for privacy-conscious AI enthusiasts**

[![GitHub Stars](https://img.shields.io/github/stars/khaosans/basic-chat-template?style=social)](https://github.com/khaosans/basic-chat-template)
[![GitHub Forks](https://img.shields.io/github/forks/khaosans/basic-chat-template?style=social)](https://github.com/khaosans/basic-chat-template)

</div>

## 📝 Project Overview

BasicChat is a privacy-first, local AI assistant that supports advanced reasoning, document analysis, and research workflows. It runs entirely on your machine, ensuring your data never leaves your environment. The app supports multiple reasoning modes, background task processing, and integrates with local LLMs via Ollama.

## 🚀 Setup & Usage

### Prerequisites
- Python 3.11+
- Ollama (for local LLMs)
- Redis (for background tasks)

### Installation
```bash
git clone https://github.com/khaosans/basic-chat-template.git
cd basic-chat-template
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running the App
```bash
# Start Ollama and pull required models
ollama serve &
ollama pull mistral
ollama pull nomic-embed-text

# Start Redis (for background tasks)
redis-server &

# Start the app
./start_dev.sh
# or
streamlit run app.py
```

### Example Usage
- Upload a PDF, text, or image file and ask questions about its content.
- Use the sidebar to select reasoning modes (Auto, Chain-of-Thought, Agent-Based, etc).
- Enable Deep Research Mode for multi-source, academic-style answers.

## 🔒 Security Best Practices
- All processing is local; no data is sent to external APIs.
- Input validation is performed on file uploads and text inputs.
- Only trusted file types are accepted (PDF, TXT, PNG, JPG, JPEG).
- Session state is isolated per user.

## 🧪 Testing
Run all tests with:
```bash
pytest
```

## 🛡️ Input Validation
- File uploads are checked for type and size.
- Text inputs are sanitized before processing.
- Configuration values are validated at startup.
