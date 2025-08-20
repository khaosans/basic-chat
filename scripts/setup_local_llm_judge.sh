#!/bin/bash

# Local LLM Judge Setup Script
# This script sets up the environment for running LLM Judge locally

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
    echo -e "${PURPLE}🤖 LLM JUDGE LOCAL SETUP${NC}"
    echo "================================"
}

print_subheader() {
    echo -e "${CYAN}$1${NC}"
}

print_header

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    print_error "Please run this script from the BasicChat root directory"
    exit 1
fi

print_subheader "🔧 Environment Setup"

# Check Python and Poetry
print_status "Checking Python and Poetry..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not installed"
    exit 1
fi

if ! command -v poetry &> /dev/null; then
    print_error "Poetry is required but not installed"
    print_status "Install Poetry: https://python-poetry.org/docs/#installation"
    exit 1
fi

print_success "Python and Poetry are available"

# Install dependencies
print_status "Installing dependencies..."
poetry install
print_success "Dependencies installed"

# Check Ollama
print_subheader "🔧 Ollama Setup"
print_status "Checking Ollama installation..."

if ! command -v ollama &> /dev/null; then
    print_error "Ollama is not installed"
    print_status "Install Ollama: https://ollama.ai"
    print_status "After installation, run: ollama serve"
    exit 1
fi

print_success "Ollama is installed"

# Check if Ollama is running
print_status "Checking Ollama service..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    print_warning "Ollama is not running"
    print_status "Starting Ollama service..."
    ollama serve &
    sleep 5
fi

print_success "Ollama is running"

# Check and install required models
print_status "Checking required models..."
REQUIRED_MODELS=("mistral")

for model in "${REQUIRED_MODELS[@]}"; do
    if ! ollama list | grep -q "$model"; then
        print_warning "Model $model not found. Pulling..."
        ollama pull "$model"
    else
        print_success "Model $model is available"
    fi
done

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p tests/data test_chroma_db logs

# Test the setup
print_subheader "🧪 Testing Setup"
print_status "Running LLM Judge tests..."

if poetry run python scripts/test_llm_judge.py; then
    print_success "All tests passed!"
else
    print_error "Some tests failed. Please check the output above."
    exit 1
fi

# Run a quick evaluation
print_subheader "🚀 Quick Evaluation Test"
print_status "Running a quick LLM Judge evaluation..."

if poetry run python basicchat/evaluation/evaluators/check_llm_judge.py --quick; then
    print_success "Quick evaluation completed successfully!"
    
    # Generate action items
    if [ -f "llm_judge_results.json" ]; then
        print_status "Generating action items..."
        poetry run python scripts/generate_llm_judge_report.py
        print_success "Action items generated!"
    fi
else
    print_error "Quick evaluation failed. Please check the output above."
    exit 1
fi

print_subheader "✅ Setup Complete!"
print_success "LLM Judge is now ready for local development!"

print_status "You can now use the following commands:"
echo ""
echo "  # Quick evaluation"
echo "  make llm-judge-quick"
echo ""
echo "  # Full evaluation"
echo "  make llm-judge"
echo ""
echo "  # Custom evaluation"
echo "  ./scripts/run_llm_judge.sh quick ollama 7.0"
echo ""
echo "  # Direct evaluation"
echo "  poetry run python basicchat/evaluation/evaluators/check_llm_judge.py --quick"
echo ""
echo "  # Generate action items"
echo "  poetry run python scripts/generate_llm_judge_report.py"
echo ""

print_status "Generated files:"
if [ -f "llm_judge_results.json" ]; then
    echo "  📄 llm_judge_results.json - Detailed evaluation results"
fi
if [ -f "llm_judge_action_items.md" ]; then
    echo "  📋 llm_judge_action_items.md - Actionable improvement plan"
fi
if [ -f "llm_judge_improvement_tips.md" ]; then
    echo "  💡 llm_judge_improvement_tips.md - Specific improvement tips"
fi

print_success "Setup completed successfully!"
