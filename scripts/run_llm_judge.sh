#!/bin/bash

# LLM Judge Evaluation Runner
# Efficient, useful evaluation with actionable output for fixing issues

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${PURPLE}🤖 LLM JUDGE EVALUATION${NC}"
    echo "================================"
}

print_subheader() {
    echo -e "${CYAN}$1${NC}"
}

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    print_error "Please run this script from the BasicChat root directory"
    exit 1
fi

# Parse command line arguments
MODE=${1:-"quick"}
BACKEND=${2:-"ollama"}
THRESHOLD=${3:-"7.0"}

print_header
print_status "Mode: $MODE"
print_status "Backend: $BACKEND"
print_status "Threshold: $THRESHOLD"

# Set environment variables for consistent evaluation
export LLM_JUDGE_THRESHOLD=$THRESHOLD
export LLM_JUDGE_BACKEND=$BACKEND
export TESTING=true
export CHROMA_PERSIST_DIR=./test_chroma_db
export MOCK_EXTERNAL_SERVICES=true

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p tests/data test_chroma_db logs

# Check backend-specific requirements
case $BACKEND in
    "ollama")
        print_subheader "🔧 Ollama Backend Setup"
        print_status "Checking Ollama status..."
        if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            print_error "Ollama is not running. Please start Ollama first."
            print_status "Run: ollama serve"
            exit 1
        fi
        print_success "Ollama is running"
        
        # Check if mistral model is available
        if ! ollama list | grep -q "mistral"; then
            print_warning "Mistral model not found. Pulling..."
            ollama pull mistral
        fi
        print_success "Mistral model is available"
        ;;
    "openai")
        print_subheader "🔧 OpenAI Backend Setup"
        if [ -z "$OPENAI_API_KEY" ]; then
            print_error "OPENAI_API_KEY environment variable is required for OpenAI backend"
            exit 1
        fi
        print_success "OpenAI API key is configured"
        ;;
    *)
        print_error "Unknown backend: $BACKEND"
        print_status "Available backends: ollama, openai"
        exit 1
        ;;
esac

# Run the evaluation
print_subheader "🚀 Starting LLM Judge Evaluation"
print_status "Backend: $BACKEND"
print_status "Mode: $MODE"
print_status "Threshold: $THRESHOLD"

# Determine the command based on backend and mode
case $BACKEND in
    "ollama")
        if [ "$MODE" = "quick" ]; then
            CMD="poetry run python basicchat/evaluation/evaluators/check_llm_judge.py --quick"
        else
            CMD="poetry run python basicchat/evaluation/evaluators/check_llm_judge.py"
        fi
        ;;
    "openai")
        if [ "$MODE" = "quick" ]; then
            CMD="poetry run python basicchat/evaluation/evaluators/check_llm_judge_openai.py --quick"
        else
            CMD="poetry run python basicchat/evaluation/evaluators/check_llm_judge_openai.py"
        fi
        ;;
esac

print_status "Running: $CMD"
eval $CMD

# Check the exit code
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    print_success "LLM Judge evaluation completed successfully!"
    
    # Check if results file exists and generate actionable report
    if [ -f "llm_judge_results.json" ]; then
        print_status "Results saved to: llm_judge_results.json"
        
        # Generate actionable report
        print_subheader "📋 Generating Actionable Report"
        poetry run python scripts/generate_llm_judge_report.py
        
        if [ -f "llm_judge_action_items.md" ]; then
            print_success "Action items saved to: llm_judge_action_items.md"
            print_status "Review this file for specific improvements to implement"
        fi
    fi
else
    print_error "LLM Judge evaluation failed with exit code: $EXIT_CODE"
    exit $EXIT_CODE
fi

# Generate final report if available
if [ -f "scripts/generate_final_report.py" ]; then
    print_status "Generating final test report..."
    poetry run python scripts/generate_final_report.py || true
fi

print_success "LLM Judge evaluation completed!"
print_status "Check llm_judge_results.json for detailed results"
print_status "Check llm_judge_action_items.md for actionable improvements"
