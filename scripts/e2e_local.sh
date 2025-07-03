#!/bin/bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }

# 1. Kill old processes
print_info "Killing old processes on ports 11434, 8501, 5555, 6379..."
lsof -i :11434 -sTCP:LISTEN | awk 'NR>1 {print $2}' | xargs kill -9 2>/dev/null || true
lsof -i :8501  -sTCP:LISTEN | awk 'NR>1 {print $2}' | xargs kill -9 2>/dev/null || true
lsof -i :5555  -sTCP:LISTEN | awk 'NR>1 {print $2}' | xargs kill -9 2>/dev/null || true
lsof -i :6379  -sTCP:LISTEN | awk 'NR>1 {print $2}' | xargs kill -9 2>/dev/null || true
print_status "Old processes killed."

# 2. Pull Ollama models
print_info "Pulling Ollama models (mistral, nomic-embed-text)..."
ollama pull mistral || true
ollama pull nomic-embed-text || true
print_status "Ollama models ready."

# 3. Start Ollama and Streamlit (background)
print_info "Starting Ollama..."
export PATH="/opt/homebrew/opt/node@20/bin:$PATH"
ollama serve &
OLLAMA_PID=$!
print_status "Ollama started (PID $OLLAMA_PID)"

print_info "Starting Streamlit app on 0.0.0.0:8501..."
./scripts/start_app.sh dev 8501 &
APP_PID=$!
print_status "Streamlit app started (PID $APP_PID)"

# 4. Wait for app to be ready
print_info "Waiting for app to be ready on http://0.0.0.0:8501..."
for i in {1..60}; do
  if curl -sSf http://0.0.0.0:8501 | grep -q "BasicChat"; then
    print_status "Streamlit is up!"
    break
  fi
  sleep 2
done

# 5. Run Playwright E2E smoke test
print_info "Running Playwright E2E smoke test..."
npx playwright test tests/e2e/specs/smoke.spec.ts --project=chromium --reporter=dot,html --output=playwright-report

# 6. Cleanup
print_info "Cleaning up background processes..."
kill $OLLAMA_PID $APP_PID 2>/dev/null || true
print_status "Done! View report with: npx playwright show-report" 