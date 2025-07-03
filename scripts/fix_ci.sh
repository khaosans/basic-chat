#!/bin/bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔧 BasicChat CI/CD Fix Script${NC}"
echo "=================================="

print_status() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }

print_info "Checking Poetry installation..."
if ! command -v poetry &> /dev/null; then
    print_warning "Poetry not found. Installing..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
    print_status "Poetry installed"
else
    print_status "Poetry already installed"
fi

print_info "Installing Python dependencies..."
poetry install --no-interaction

print_info "Installing Node.js dependencies..."
npm ci
npx playwright install --with-deps

print_info "Creating test directories..."
mkdir -p tests/data test_chroma_db tests/e2e/fixtures temp_audio uploads chroma_db redis_data

print_info "Generating test assets..."
python scripts/generate_test_assets.py || echo "Test assets generation failed, continuing..."

print_info "Setting up environment variables..."
export TESTING=true
export CHROMA_PERSIST_DIR=./test_chroma_db
export MOCK_EXTERNAL_SERVICES=true
export ENABLE_BACKGROUND_TASKS=true
export REDIS_ENABLED=false
export CELERY_BROKER_URL=redis://localhost:6379/0
export OLLAMA_BASE_URL=http://localhost:11434

print_info "Running unit tests..."
poetry run pytest -n auto tests/ -m "unit or fast" --ignore=tests/integration -v --tb=short --cov=app --cov=reasoning_engine --cov=document_processor --cov=utils --cov=task_manager --cov=task_ui --cov=tasks --cov-report=term-missing --cov-report=html:htmlcov

print_info "Generating final test report..."
python scripts/generate_final_report.py || true

print_status "CI/CD fix script completed successfully!"
print_info "Next steps:"
print_info "1. Run E2E tests: poetry run playwright test"
print_info "2. Start the app: poetry run streamlit run app.py"
print_info "3. Check coverage: open htmlcov/index.html" 