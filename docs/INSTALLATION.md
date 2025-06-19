# Installation Guide

## Prerequisites

### System Requirements
- **Python**: 3.11 or higher
- **Git**: For cloning the repository
- **Ollama**: [Install Ollama](https://ollama.ai) - Local large language model server
- **Memory**: Minimum 8GB RAM (16GB+ recommended for larger models)
- **Storage**: 10GB+ free space for models and dependencies

### Ollama Installation
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# Download from https://ollama.ai/download
```

## Quick Start

### 1. Install Required Models
```bash
# Core models for basic functionality
ollama pull mistral              # Primary reasoning model (3.8GB)
ollama pull nomic-embed-text     # Embedding model for RAG (1.2GB)

# Optional models for enhanced capabilities
ollama pull llava               # Vision model for image analysis (4.7GB)
ollama pull codellama           # Code generation and analysis (6.7GB)
ollama pull llama2              # Alternative base model (3.8GB)
```

### 2. Clone and Setup
```bash
# Clone the repository
git clone https://github.com/khaosans/basic-chat-template.git
cd basic-chat-template

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment (Optional)
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

# Vector Store Configuration
VECTORSTORE_DIR=./chroma_db
EMBEDDING_MODEL=nomic-embed-text

# LLM Parameters
TEMPERATURE=0.7
MAX_TOKENS=2048
```

### 4. Start the Application
```bash
# Start Ollama service (if not running)
ollama serve &

# Launch BasicChat
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## Advanced Configuration

### Performance Tuning
```bash
# Increase rate limits for high-traffic scenarios
RATE_LIMIT=20
RATE_LIMIT_PERIOD=1

# Adjust caching for memory-constrained environments
CACHE_MAXSIZE=500
CACHE_TTL=1800

# Optimize timeouts for your network
REQUEST_TIMEOUT=60
CONNECT_TIMEOUT=10
MAX_RETRIES=5
```

### Model Selection
```bash
# Use different models for specific tasks
OLLAMA_MODEL=llama2          # Alternative base model
OLLAMA_MODEL=codellama       # For code-related tasks
OLLAMA_MODEL=llava          # For image analysis
EMBEDDING_MODEL=nomic-embed-text  # Embedding model
```

### Redis Setup (Optional)
For production use with distributed caching:
```bash
# Install Redis
# macOS
brew install redis

# Ubuntu/Debian
sudo apt-get install redis-server

# Start Redis
redis-server

# Configure in .env.local
REDIS_URL=redis://localhost:6379
REDIS_ENABLED=true
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

# Remove and re-download a model
ollama rm mistral
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

#### Cache Issues
```bash
# Clear cache if experiencing issues
# The cache will automatically reset on restart
rm -rf chroma_db/
```

#### Performance Issues
```bash
# Check system resources
htop
free -h
df -h

# Adjust rate limits and timeouts in .env.local
# Monitor cache statistics in the UI
```

### Debug Mode
```bash
# Enable debug logging
LOG_LEVEL=DEBUG
ENABLE_STRUCTURED_LOGGING=true

# Run with verbose output
streamlit run app.py --logger.level=debug
```

### System Requirements Check
```bash
# Check Python version
python --version  # Should be 3.11+

# Check available memory
free -h  # Should have 8GB+ available

# Check disk space
df -h  # Should have 10GB+ free

# Check Ollama installation
ollama --version
```

## Production Deployment

### Docker Deployment
```bash
# Build Docker image
docker build -t basic-chat .

# Run with Docker Compose
docker-compose up -d
```

### Environment Variables for Production
```bash
# Security
LOG_LEVEL=WARNING
ENABLE_STRUCTURED_LOGGING=true

# Performance
RATE_LIMIT=50
REQUEST_TIMEOUT=60
CACHE_TTL=7200

# Redis (Production)
REDIS_URL=redis://your-redis-server:6379
REDIS_ENABLED=true

# Monitoring
ENABLE_HEALTH_CHECKS=true
```

## Support

### Getting Help
- **Documentation**: Check [README.md](../README.md) and [REASONING_FEATURES.md](../REASONING_FEATURES.md)
- **Issues**: Report bugs on [GitHub Issues](https://github.com/khaosans/basic-chat-template/issues)
- **Discussions**: Join [GitHub Discussions](https://github.com/khaosans/basic-chat-template/discussions)

### System Compatibility
- **macOS**: 10.15+ (Catalina and later)
- **Linux**: Ubuntu 18.04+, CentOS 7+, Debian 9+
- **Windows**: Windows 10+ with WSL2 recommended
- **Docker**: Docker 20.10+ with Docker Compose 2.0+ 