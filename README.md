# 🤖 Document-Aware Chat Assistant

A streamlit-based chat application that can process and understand documents, including PDFs, text files, and images.

```ascii
+----------------+     +-----------------+     +------------------+
|                |     |                 |     |                  |
|  User Upload   +---->+  Document       +---->+  Vector Store   |
|                |     |  Processing     |     |  (ChromaDB)     |
+----------------+     +-----------------+     +------------------+
                              |
                              v
+----------------+     +-----------------+     +------------------+
|                |     |                 |     |                  |
|  User Query    +---->+  RAG Pipeline   +<----+  LLM (Ollama)   |
|                |     |                 |     |                  |
+----------------+     +-----------------+     +------------------+
                              |
                              v
                      +-----------------+
                      |                 |
                      |    Response     |
                      |                 |
                      +-----------------+
```

## 🚀 Features

- 📄 Document Processing:
  - PDF files
  - Text files
  - Images (OCR support)
- 💬 Interactive Chat
- 🔍 RAG (Retrieval Augmented Generation)
- 🖼️ OCR for Images
- 🤖 Local LLM Integration (Ollama)

## 🛠️ Prerequisites

- Python 3.11+
- Poetry
- Tesseract OCR
- Ollama with Mistral model

### System Dependencies

```bash
# macOS
brew install tesseract
brew install poppler
brew install libmagic

# Ubuntu
sudo apt-get install tesseract-ocr
sudo apt-get install poppler-utils
sudo apt-get install libmagic1
```

## 📦 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd basic-chat
```

2. Install dependencies:
```bash
poetry install
```

3. Start Ollama (in a separate terminal):
```bash
ollama serve
```

4. Run the application:
```bash
./start.sh
```

## 🐛 Known Issues

1. **PNG Processing**: Some PNG files may not process correctly due to OCR limitations. Workaround:
   - Convert PNG to JPEG before uploading
   - Use high-contrast images
   - Ensure text is clearly visible

2. **Memory Usage**: Large documents may require additional memory. Configure using:
```bash
export PYTHONMEM=4G
```

## 🔧 Configuration

Environment variables (create `.env.local`):
```env
OLLAMA_BASE_URL=http://localhost:11434
CHROMADB_PATH=./chroma_db
LOG_LEVEL=INFO
```

## 📁 Project Structure
