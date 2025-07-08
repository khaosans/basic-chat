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

See the [Architecture Overview](docs/DOCUMENTATION.md#background-task-system) for a diagram of the task queue and worker system.

---

## 🚀 Quick Start

### **Prerequisites**
