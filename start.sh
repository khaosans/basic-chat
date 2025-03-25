#!/bin/bash
set -e

# Ensure we're in the poetry environment
echo "🔄 Activating Poetry environment..."
poetry install

# Install watchdog
echo "📦 Installing watchdog..."
poetry add watchdog

# Only clean ChromaDB if --clean flag is provided
if [ "$1" = "--clean" ]; then
    echo "🧹 Cleaning ChromaDB data..."
    rm -rf ./chroma_db
fi

# Run setup checks
echo "🔍 Running setup checks..."
TESTING=false poetry run python setup.py

# Start the application
echo "🚀 Starting the application..."
poetry run streamlit run app.py 