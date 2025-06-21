# BasicChat: Your Intelligent Local AI Assistant

**BasicChat** is an intelligent, private AI assistant that runs entirely on your local machine. It's designed for complex reasoning, document analysis, and a variety of other tasks, all while ensuring your data remains secure and private.

## 🎥 Demo

![BasicChat Demo](assets/demo_seq_0.6s.gif)

## 🌟 Key Features

- **🔒 Privacy First**: 100% local, no data ever leaves your machine.
- **🧠 Advanced Reasoning**: Features multiple reasoning modes (Chain-of-Thought, Multi-Step Analysis) to break down complex problems.
- **🛠️ Powerful Built-in Tools**: Equipped with a smart calculator, timezone-aware time tools, and integrated web search.
- **📄 Document & Image Analysis**: Process PDFs, text files, and images with advanced Retrieval-Augmented Generation (RAG) and OCR capabilities.
- **⚡ Performance Optimized**: Built with an async architecture and a multi-layer caching system for fast, reliable responses.

For a detailed list of features, see the [Features Overview](docs/FEATURES.md).

## 🚀 Getting Started

### 1. Prerequisites

- **Python**: Version 3.11 or higher.
- **Ollama**: [Install Ollama](https://ollama.ai) to serve your local AI models.
- **Git**: Required to clone the repository.

### 2. Setup

```bash
# Clone the repository
git clone https://github.com/khaosans/basic-chat-template.git
cd basic-chat-template

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: .\venv\Scripts\activate

# Install required Python packages
pip install -r requirements.txt
```

### 3. Download AI Models

BasicChat uses Ollama to run local models. Download the default models with the following commands:

```bash
# Default reasoning and embedding models
ollama pull mistral
ollama pull nomic-embed-text

# (Optional) Vision model for image analysis
ollama pull llava
```

### 4. Run the Application

```bash
# Ensure the Ollama application is running before you start.
# Then, launch the BasicChat interface:
streamlit run app.py
```

The application will now be available at `http://localhost:8501`.

## 📚 Documentation

- **[Features Overview](docs/FEATURES.md)**: A detailed look at all capabilities.
- **[System Architecture](docs/ARCHITECTURE.md)**: An overview of the technical design and data flow.
- **[Development Guide](docs/DEVELOPMENT.md)**: Information on contributing, testing, and development workflows.
- **[Roadmap](docs/ROADMAP.md)**: Our plans for future features and improvements.

## 🧪 Testing

We use `pytest` for testing. To run the test suite:

```bash
pytest
```

For more details on testing and contributions, please see our [Development Guide](docs/DEVELOPMENT.md).

## 🤝 Contributing

Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
