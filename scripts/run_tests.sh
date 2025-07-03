#!/bin/bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🧪 BasicChat Test Runner${NC}"
echo "=========================="

print_status() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }

TEST_TYPE=${1:-"all"}
PARALLEL=${2:-"auto"}

print_info "Running tests: $TEST_TYPE (parallel: $PARALLEL)"

export TESTING=true
export CHROMA_PERSIST_DIR=./test_chroma_db
export MOCK_EXTERNAL_SERVICES=true
export ENABLE_BACKGROUND_TASKS=true
export REDIS_ENABLED=false
export CELERY_BROKER_URL=redis://localhost:6379/0
export OLLAMA_BASE_URL=http://localhost:11434

mkdir -p tests/data test_chroma_db tests/e2e/fixtures
python scripts/generate_test_assets.py || echo "Test assets generation failed, continuing..."

case $TEST_TYPE in
    "unit"|"fast")
        print_info "Running unit tests..."
        poetry run pytest -n $PARALLEL tests/ -m "unit or fast" --ignore=tests/integration -v --tb=short --cov=app --cov=reasoning_engine --cov=document_processor --cov=utils --cov=task_manager --cov=task_ui --cov=tasks --cov-report=term-missing --cov-report=html:htmlcov
        ;;
    "integration")
        print_info "Running integration tests..."
        poetry run pytest -n $PARALLEL tests/ -m "integration" -v --tb=short --timeout=300
        ;;
    "e2e")
        print_info "Running E2E tests..."
        npx playwright test --reporter=html,json,junit
        ;;
    "all")
        print_info "Running all tests..."
        print_info "1. Running unit tests..."
        poetry run pytest -n $PARALLEL tests/ -m "unit or fast" --ignore=tests/integration -v --tb=short --cov=app --cov=reasoning_engine --cov=document_processor --cov=utils --cov=task_manager --cov=task_ui --cov=tasks --cov-report=term-missing --cov-report=html:htmlcov
        if [ "$CI" != "true" ]; then
            print_info "2. Running integration tests..."
            poetry run pytest -n $PARALLEL tests/ -m "integration" -v --tb=short --timeout=300
        fi
        print_info "3. Running E2E tests..."
        npx playwright test --reporter=html,json,junit
        ;;
    *)
        print_error "Unknown test type: $TEST_TYPE"
        print_info "Available options: unit, integration, e2e, all"
        exit 1
        ;;
esac

print_info "Generating final test report..."
python scripts/generate_final_report.py || true

print_status "Test run completed!"
print_info "Coverage report: open htmlcov/index.html"
print_info "E2E report: open playwright-report/index.html" 