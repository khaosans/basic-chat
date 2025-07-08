#!/bin/bash

# Kill any existing backend processes
pkill -f "uvicorn.*main:app" 2>/dev/null || true
lsof -ti :8080 | xargs kill -9 2>/dev/null || true

# Start the FastAPI backend
cd "$(dirname "$0")"
echo "🚀 Starting BasicChat API backend..."
poetry run uvicorn main:app --host 0.0.0.0 --port 8080 --reload 