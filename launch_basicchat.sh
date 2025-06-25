#!/bin/bash

# Simple BasicChat launcher
# Usage: ./launch_basicchat.sh

# Navigate to the project directory
cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Load environment variables
if [ -f "basicchat.env" ]; then
    export $(cat basicchat.env | grep -v '^#' | xargs)
fi

# Make the startup script executable and run it
chmod +x start_basicchat.sh
./start_basicchat.sh 