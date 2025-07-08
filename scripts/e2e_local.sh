#!/bin/bash
set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. Kill old app instances
function kill_old_instances() {
  echo -e "${YELLOW}🔪 Killing old app instances on port 8501...${NC}"
  pkill -f "uvicorn|python.*main:app|streamlit" 2>/dev/null || true
  lsof -ti :8501 | xargs kill -9 2>/dev/null || true
  sleep 2
}

# 2. Start all required services
function start_services() {
  echo -e "${YELLOW}🚀 Starting all required services...${NC}"
  # Start Ollama if not running
  if ! pgrep -f "ollama serve" >/dev/null; then
    ollama serve &
    sleep 2
  fi
  # Pull Mistral model if not present
  if ! ollama list | grep -q "mistral"; then
    ollama pull mistral
  fi
  # Start the app (Streamlit)
  ./start_basicchat.sh &
  sleep 5
}

# 3. Run health check
function run_health_check() {
  echo -e "${YELLOW}🩺 Running health check...${NC}"
  if ! poetry run python scripts/e2e_health_check.py; then
    echo -e "${RED}❌ Health check failed. Exiting.${NC}"
    exit 1
  fi
  echo -e "${GREEN}✅ All services healthy!${NC}"
}

# 4. Run Playwright E2E tests
function run_e2e_tests() {
  echo -e "${YELLOW}🧪 Running Playwright E2E tests...${NC}"
  # Use latest Node if available
  if command -v /Users/Sour/.nvm/versions/node/v22.15.0/bin/node >/dev/null; then
    NODE_BIN="/Users/Sour/.nvm/versions/node/v22.15.0/bin/node"
  elif command -v node >/dev/null && [[ $(node --version | cut -d. -f1 | tr -d v) -ge 18 ]]; then
    NODE_BIN="node"
  else
    echo -e "${RED}❌ Node.js 18+ is required. Exiting.${NC}"
    exit 1
  fi
  $NODE_BIN ./node_modules/.bin/playwright test --reporter=list
}

kill_old_instances
start_services
run_health_check
run_e2e_tests 