# Installation Guide

[← Back to README](../README.md) | [Features →](FEATURES.md) | [Architecture →](ARCHITECTURE.md) | [Development →](DEVELOPMENT.md) | [Roadmap →](ROADMAP.md)

---

## Overview
This guide provides detailed setup instructions for BasicChat, including prerequisites, installation steps, configuration options, and troubleshooting. The installation process follows established best practices for Python application deployment and local AI system setup.

## Prerequisites

### System Requirements
- **Python**: 3.11 or higher
- **Ollama**: Latest version from [ollama.ai](https://ollama.ai)
- **Git**: For cloning the repository
- **Memory**: Minimum 8GB RAM (16GB+ recommended)
- **Storage**: 10GB+ free space for models

### Required Models
```bash
# Core models for basic functionality
ollama pull mistral              # Primary reasoning model
ollama pull nomic-embed-text     # Embedding model for RAG

# Optional models for enhanced capabilities
ollama pull llava               # Vision model for image analysis
ollama pull codellama           # Code generation and analysis
```

## Installation Steps

### 1. Clone Repository
```bash
git clone https://github.com/khaosans/basic-chat-template.git
cd basic-chat-template
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# OR
.\venv\Scripts\activate   # Windows
```

### 3. Install Dependencies
```bash
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install pytest pytest-asyncio pytest-cov black flake8 mypy
```

### 4. Start Ollama Service
```bash
# Start Ollama in background
ollama serve &

# Verify Ollama is running
ollama list
```

### 5. Launch Application
```bash
# Start BasicChat
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## Configuration

### Environment Variables
Create `.env.local` for custom configuration:

```bash
# Ollama Configuration
OLLAMA_API_URL=http://localhost:11434/api
OLLAMA_MODEL=mistral

# Performance Settings
ENABLE_CACHING=true
CACHE_TTL=3600
RATE_LIMIT=10
REQUEST_TIMEOUT=30

# Redis Configuration (Optional)
REDIS_URL=redis://localhost:6379
REDIS_ENABLED=false

# Logging
LOG_LEVEL=INFO
ENABLE_STRUCTURED_LOGGING=true
```

### Performance Tuning
```bash
# High-traffic scenarios
RATE_LIMIT=20
RATE_LIMIT_PERIOD=1

# Memory-constrained environments
CACHE_MAXSIZE=500
CACHE_TTL=1800

# Network optimization
REQUEST_TIMEOUT=60
CONNECT_TIMEOUT=10
```

## Advanced Setup

### Redis Installation (Optional)
```bash
# macOS
brew install redis
brew services start redis

# Ubuntu/Debian
sudo apt-get install redis-server
sudo systemctl start redis-server

# Windows
# Download from https://redis.io/download
```

### Docker Setup
```bash
# Build image
docker build -t basic-chat .

# Run container
docker run -p 8501:8501 basic-chat
```

## Troubleshooting

### Common Issues

#### Ollama Connection Issues
```bash
# Check if Ollama is running
ollama list

# Restart Ollama service
ollama serve

# Check Ollama logs
ollama logs
```

#### Model Download Issues
```bash
# Check available models
ollama list

# Pull specific model
ollama pull mistral

# Check disk space
df -h
```

#### Python Environment Issues
```bash
# Verify Python version
python --version

# Recreate virtual environment
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Port Conflicts
```bash
# Check if port 8501 is in use
lsof -i :8501

# Use different port
streamlit run app.py --server.port 8502
```

### Debug Mode
```bash
# Enable debug logging
LOG_LEVEL=DEBUG
ENABLE_STRUCTURED_LOGGING=true

# Run with verbose output
streamlit run app.py --logger.level=debug
```

## Verification

### Test Installation
```bash
# Run test suite
pytest

# Test specific components
pytest tests/test_basic.py
pytest tests/test_reasoning.py
```

### Check System Health
```bash
# Verify Ollama models
ollama list

# Check Python packages
pip list

# Test Ollama API
curl http://localhost:11434/api/tags
```

## Next Steps

- **[Features Overview](FEATURES.md)** - Learn about BasicChat's capabilities
- **[System Architecture](ARCHITECTURE.md)** - Understand the technical design
- **[Development Guide](DEVELOPMENT.md)** - Start contributing to the project
- **[Troubleshooting Guide](TROUBLESHOOTING.md)** - Resolve common issues
- **[Production Roadmap](ROADMAP.md)** - See future development plans

---

[← Back to README](../README.md) | [Features →](FEATURES.md) | [Architecture →](ARCHITECTURE.md) | [Development →](DEVELOPMENT.md) | [Roadmap →](ROADMAP.md) 