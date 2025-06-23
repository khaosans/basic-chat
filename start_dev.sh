#!/bin/bash

# Development startup script for BasicChat with long-running tasks

set -e

echo "🚀 Starting BasicChat with long-running tasks..."

# Check if Redis is running
if ! redis-cli ping > /dev/null 2>&1; then
    echo "⚠️  Redis is not running. Starting Redis..."
    if command -v brew > /dev/null 2>&1; then
        brew services start redis
    elif command -v systemctl > /dev/null 2>&1; then
        sudo systemctl start redis
    else
        echo "❌ Please start Redis manually and try again"
        exit 1
    fi
    sleep 2
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "⚠️  Ollama is not running. Please start Ollama manually:"
    echo "   ollama serve"
    echo "   Then pull a model: ollama pull mistral"
    exit 1
fi

# Create necessary directories
mkdir -p chroma_db temp_audio uploads

# Set environment variables
export CELERY_BROKER_URL=redis://localhost:6379/0
export CELERY_RESULT_BACKEND=redis://localhost:6379/0
export REDIS_ENABLED=true
export REDIS_URL=redis://localhost:6379
export ENABLE_BACKGROUND_TASKS=true

# Function to cleanup background processes
cleanup() {
    echo "🛑 Shutting down..."
    kill $CELERY_PID $FLOWER_PID $BEAT_PID 2>/dev/null || true
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start Celery worker for reasoning tasks
echo "🔧 Starting Celery worker (reasoning)..."
celery -A tasks worker --loglevel=info --queues=reasoning --concurrency=2 &
CELERY_PID=$!

# Start Celery worker for document tasks
echo "🔧 Starting Celery worker (documents)..."
celery -A tasks worker --loglevel=info --queues=documents --concurrency=1 &
CELERY_DOCS_PID=$!

# Start Celery beat for scheduled tasks
echo "⏰ Starting Celery beat..."
celery -A tasks beat --loglevel=info &
BEAT_PID=$!

# Start Flower for monitoring
echo "🌸 Starting Flower (task monitoring)..."
celery -A tasks flower --port=5555 --broker=redis://localhost:6379/0 &
FLOWER_PID=$!

# Wait a moment for services to start
sleep 3

echo "✅ All services started!"
echo ""
echo "📱 Application URLs:"
echo "   Main App: http://localhost:8501"
echo "   Task Monitor: http://localhost:5555"
echo ""
echo "🔧 Services:"
echo "   Redis: localhost:6379"
echo "   Ollama: localhost:11434"
echo ""
echo "Press Ctrl+C to stop all services"

# Start the main Streamlit application
echo "🌐 Starting Streamlit application..."
streamlit run app.py --server.port=8501 --server.address=0.0.0.0 