#!/bin/bash

# BasicChat Startup Script
# This script starts all required services for BasicChat

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    print_error "Please run this script from the BasicChat root directory"
    exit 1
fi

print_status "Starting BasicChat..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not installed"
    exit 1
fi

# Check if Ollama is running
print_status "Checking Ollama status..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    print_warning "Ollama is not running. Starting Ollama..."
    if command -v ollama &> /dev/null; then
        ollama serve &
        sleep 3
    else
        print_error "Ollama is not installed. Please install Ollama first."
        print_status "Visit: https://ollama.ai"
        exit 1
    fi
else
    print_success "Ollama is running"
fi

# Check if required models are available
print_status "Checking required models..."
REQUIRED_MODELS=("mistral" "nomic-embed-text")

for model in "${REQUIRED_MODELS[@]}"; do
    if ! ollama list | grep -q "$model"; then
        print_warning "Model $model not found. Pulling..."
        ollama pull "$model"
    else
        print_success "Model $model is available"
    fi
done

# Check if Redis is running (optional)
print_status "Checking Redis status..."
if ! redis-cli ping > /dev/null 2>&1; then
    print_warning "Redis is not running. Background tasks will be disabled."
    print_status "To enable background tasks, start Redis: brew services start redis"
else
    print_success "Redis is running"
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p data/uploads data/temp_audio logs

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Start the application
print_status "Starting BasicChat application..."
print_success "Application will be available at: http://localhost:8501"
print_success "Task monitor (if Redis is running): http://localhost:5555"

# Use the new main.py entry point
streamlit run main.py --server.port 8501 --server.address 0.0.0.0
