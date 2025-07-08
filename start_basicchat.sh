#!/bin/bash

# Enhanced BasicChat startup script with automatic Redis management
# Handles startup, monitoring, and graceful shutdown

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REDIS_PORT=6379
STREAMLIT_PORT=8501
FLOWER_PORT=5555
OLLAMA_PORT=11434
REDIS_DATA_DIR="./redis_data"
REDIS_PID_FILE="./redis.pid"

echo -e "${BLUE}🚀 BasicChat Enhanced Startup Script${NC}"
echo "=================================="

# Show cool ASCII animation/logo at startup
ascii_logo=(
"  ____            _      _      _____ _           _   "
" |  _ \\          | |    | |    / ____| |         | |  "
" | |_) | __ _ ___| | __ | |   | |    | |__   __ _| |_ "
" |  _ < / _\` / __| |/ / | |   | |    | '_ \\ / _\` | __|"
" | |_) | (_| \\__ \\   <  | |___| |____| | | | (_| | |_ "
" |____/ \\__,_|___/_|\\_\\ |______\\_____|_| |_|\\__,_|\\__|"
)

for line in "${ascii_logo[@]}"; do
  for ((i=0; i<${#line}; i++)); do
    echo -ne "\033[1;36m${line:$i:1}\033[0m"
    sleep 0.002
  done
  echo
  sleep 0.03
  done

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Function to check if a port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Spinner function for animated feedback
spinner() {
  local pid=$1
  local msg="$2"
  local spin='|/-\\'
  local i=0
  tput civis 2>/dev/null # Hide cursor
  while kill -0 $pid 2>/dev/null; do
    i=$(( (i+1) % 4 ))
    printf "\r\033[1;36m%s %s\033[0m" "${spin:$i:1}" "$msg"
    sleep 0.1
  done
  printf "\r\033[1;32m✔ %s\033[0m\n" "$msg"
  tput cnorm 2>/dev/null # Show cursor
}

# Enhanced wait_for_service with spinner
wait_for_service() {
  local service_name=$1
  local port=$2
  local max_attempts=30
  local attempt=1
  local spin='|/-\\'
  local i=0
  print_info "Waiting for $service_name to be ready on port $port..."
  while [ $attempt -le $max_attempts ]; do
    if check_port $port; then
      printf "\r\033[1;32m✔ %s is ready!\033[0m\n" "$service_name"
      return 0
    fi
    i=$(( (i+1) % 4 ))
    printf "\r\033[1;36m%s Waiting for %s...\033[0m" "${spin:$i:1}" "$service_name"
    sleep 0.2
    attempt=$((attempt + 1))
  done
  printf "\r\033[1;31m✖ %s failed to start within %s seconds\033[0m\n" "$service_name" "$max_attempts"
  return 1
}

# Function to start Redis
start_redis() {
    print_info "Starting Redis..."
    
    # Create Redis data directory
    mkdir -p "$REDIS_DATA_DIR"
    
    # Check if Redis is already running
    if check_port $REDIS_PORT; then
        print_status "Redis is already running on port $REDIS_PORT"
        return 0
    fi
    
    # Try to start Redis using different methods
    if command -v redis-server >/dev/null 2>&1; then
        # Start Redis server directly
        print_info "Starting Redis server..."
        redis-server --port $REDIS_PORT --dir "$REDIS_DATA_DIR" --appendonly yes --daemonize yes --pidfile "$REDIS_PID_FILE"
    elif command -v brew >/dev/null 2>&1; then
        # Use Homebrew services
        print_info "Starting Redis via Homebrew services..."
        brew services start redis
    elif command -v systemctl >/dev/null 2>&1; then
        # Use systemctl
        print_info "Starting Redis via systemctl..."
        sudo systemctl start redis
    else
        print_error "Redis not found. Please install Redis manually."
        print_info "Installation options:"
        print_info "  - macOS: brew install redis"
        print_info "  - Ubuntu: sudo apt-get install redis-server"
        print_info "  - Docker: docker run -d -p 6379:6379 redis:7-alpine"
        exit 1
    fi
    
    # Wait for Redis to be ready
    if wait_for_service "Redis" $REDIS_PORT; then
        print_status "Redis started successfully"
    else
        print_error "Failed to start Redis"
        exit 1
    fi
}

# Function to check Ollama
check_ollama() {
    print_info "Checking Ollama..."
    
    if ! check_port $OLLAMA_PORT; then
        print_warning "Ollama is not running on port $OLLAMA_PORT"
        print_info "Please start Ollama manually:"
        print_info "  ollama serve"
        print_info "Then pull required models:"
        print_info "  ollama pull mistral"
        print_info "  ollama pull nomic-embed-text"
        print_info "  ollama pull llava"
        exit 1
    fi
    
    # Test Ollama API
    if curl -s http://localhost:$OLLAMA_PORT/api/tags >/dev/null 2>&1; then
        print_status "Ollama is running and responding"
    else
        print_error "Ollama is not responding to API calls"
        exit 1
    fi
}

# Function to create necessary directories
setup_directories() {
    print_info "Setting up directories..."
    mkdir -p chroma_db temp_audio uploads
    print_status "Directories created"
}

# Function to set environment variables
setup_environment() {
    print_info "Setting up environment variables..."
    export CELERY_BROKER_URL=redis://localhost:$REDIS_PORT/0
    export CELERY_RESULT_BACKEND=redis://localhost:$REDIS_PORT/0
    export REDIS_ENABLED=true
    export REDIS_URL=redis://localhost:$REDIS_PORT
    export ENABLE_BACKGROUND_TASKS=true
    print_status "Environment variables set"
}

# Function to start Celery workers
start_celery_workers() {
    print_info "Starting Celery workers..."
    
    # Start reasoning worker
    print_info "Starting reasoning worker..."
    celery -A tasks worker --loglevel=info --queues=reasoning --concurrency=2 &
    CELERY_PID=$!
    
    # Start document worker
    print_info "Starting document worker..."
    celery -A tasks worker --loglevel=info --queues=documents --concurrency=1 &
    CELERY_DOCS_PID=$!
    
    # Start Celery beat
    print_info "Starting Celery beat..."
    celery -A tasks beat --loglevel=info &
    BEAT_PID=$!
    
    # Start Flower
    print_info "Starting Flower monitoring..."
    celery -A tasks flower --port=$FLOWER_PORT --broker=redis://localhost:$REDIS_PORT/0 &
    FLOWER_PID=$!
    
    print_status "Celery workers started"
}

# Function to cleanup and shutdown gracefully
cleanup() {
    echo ""
    print_info "🛑 Shutting down BasicChat gracefully..."
    
    # Stop Celery workers
    if [ ! -z "$CELERY_PID" ]; then
        print_info "Stopping Celery workers..."
        kill $CELERY_PID $CELERY_DOCS_PID $FLOWER_PID $BEAT_PID 2>/dev/null || true
        wait $CELERY_PID $CELERY_DOCS_PID $FLOWER_PID $BEAT_PID 2>/dev/null || true
    fi
    
    # Stop Streamlit if it's running
    if check_port $STREAMLIT_PORT; then
        print_info "Stopping Streamlit..."
        pkill -f "streamlit run app.py" 2>/dev/null || true
    fi
    
    # Stop Redis if we started it
    if [ -f "$REDIS_PID_FILE" ]; then
        print_info "Stopping Redis..."
        kill $(cat "$REDIS_PID_FILE") 2>/dev/null || true
        rm -f "$REDIS_PID_FILE"
    fi
    
    print_status "Shutdown complete"
    exit 0
}

# Set up signal handlers for graceful shutdown
trap cleanup SIGINT SIGTERM

# Main startup sequence
main() {
    # Start Redis
    start_redis
    
    # Check Ollama
    check_ollama
    
    # Setup directories and environment
    setup_directories
    setup_environment
    
    # Start Celery workers
    start_celery_workers
    
    # Wait for services to be ready
    sleep 3
    
    # Display status
    echo ""
    print_status "All services started successfully!"
    echo ""
    echo -e "${BLUE}📱 Application URLs:${NC}"
    echo "   Main App: http://localhost:$STREAMLIT_PORT"
    echo "   Task Monitor: http://localhost:$FLOWER_PORT"
    echo ""
    echo -e "${BLUE}🔧 Services:${NC}"
    echo "   Redis: localhost:$REDIS_PORT"
    echo "   Ollama: localhost:$OLLAMA_PORT"
    echo ""
    echo -e "${YELLOW}Press Ctrl+C to stop all services gracefully${NC}"
    echo ""
    
    # Start Streamlit application
    print_info "Starting Streamlit application..."
    streamlit run app.py --server.port=$STREAMLIT_PORT --server.address=0.0.0.0
}

# Run main function
main "$@" 