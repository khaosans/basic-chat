#!/bin/bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🚀 BasicChat Application Starter${NC}"
echo "====================================="

print_status() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }

MODE=${1:-"dev"}
PORT=${2:-"8501"}

print_info "Starting in $MODE mode on port $PORT"

if [ "$MODE" = "ci" ]; then
    print_info "Setting up CI environment..."
    export TESTING=true
    export CHROMA_PERSIST_DIR=./test_chroma_db
    export MOCK_EXTERNAL_SERVICES=true
    export ENABLE_BACKGROUND_TASKS=false
    export REDIS_ENABLED=false
    export CELERY_BROKER_URL=redis://localhost:6379/0
    export OLLAMA_BASE_URL=http://localhost:11434
else
    print_info "Setting up development environment..."
    export TESTING=false
    export CHROMA_PERSIST_DIR=./chroma_db
    export MOCK_EXTERNAL_SERVICES=false
    export ENABLE_BACKGROUND_TASKS=true
    export REDIS_ENABLED=true
    export CELERY_BROKER_URL=redis://localhost:6379/0
    export OLLAMA_BASE_URL=http://localhost:11434
fi

print_info "Creating directories..."
mkdir -p tests/data test_chroma_db tests/e2e/fixtures temp_audio uploads chroma_db redis_data

if [ "$MODE" = "ci" ]; then
    print_info "CI mode: Starting Streamlit in headless mode..."
    streamlit run app.py --server.port $PORT --server.headless true --server.address 0.0.0.0
else
    print_info "Development mode: Starting full application stack..."
    if ! lsof -Pi :6379 -sTCP:LISTEN -t >/dev/null 2>&1; then
        print_info "Starting Redis..."
        redis-server --port 6379 --dir ./redis_data --appendonly yes --daemonize yes --pidfile ./redis.pid
        sleep 2
    fi
    if ! lsof -Pi :11434 -sTCP:LISTEN -t >/dev/null 2>&1; then
        print_warning "Ollama not running. Please start Ollama manually:"
        print_info "  ollama serve"
        print_info "  ollama pull mistral"
        print_info "  ollama pull nomic-embed-text"
    fi
    print_info "Starting Celery workers..."
    celery -A tasks worker --loglevel=info --queues=reasoning --concurrency=2 &
    CELERY_PID=$!
    celery -A tasks worker --loglevel=info --queues=documents --concurrency=1 &
    CELERY_DOCS_PID=$!
    celery -A tasks beat --loglevel=info &
    BEAT_PID=$!
    celery -A tasks flower --port=5555 --broker=redis://localhost:6379/0 &
    FLOWER_PID=$!
    sleep 3
    print_status "All services started!"
    echo ""
    echo -e "${BLUE}📱 Application URLs:${NC}"
    echo "   Main App: http://localhost:$PORT"
    echo "   Task Monitor: http://localhost:5555"
    echo ""
    echo -e "${YELLOW}Press Ctrl+C to stop all services gracefully${NC}"
    echo ""
    trap "echo 'Stopping services...'; kill $CELERY_PID $CELERY_DOCS_PID $FLOWER_PID $BEAT_PID 2>/dev/null || true; exit" INT TERM
    streamlit run app.py --server.port $PORT --server.address 0.0.0.0
fi 