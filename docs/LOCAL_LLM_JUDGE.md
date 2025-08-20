# 🤖 LLM Judge Local Development Guide

This guide explains how to set up and run the LLM Judge evaluation system locally for development and testing.

## 🚀 Quick Start

### 1. Automatic Setup (Recommended)

Run the setup script to automatically configure everything:

```bash
./scripts/setup_local_llm_judge.sh
```

This script will:
- Check and install dependencies
- Set up Ollama and required models
- Test the LLM Judge functionality
- Run a quick evaluation
- Generate action items

### 2. Manual Setup

If you prefer to set up manually, follow these steps:

#### Prerequisites

1. **Python 3.11+** and **Poetry**
   ```bash
   # Install Poetry if not already installed
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Ollama**
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Start Ollama service
   ollama serve
   
   # Pull required model
   ollama pull mistral
   ```

#### Installation

1. **Install dependencies**
   ```bash
   poetry install
   ```

2. **Create necessary directories**
   ```bash
   mkdir -p tests/data test_chroma_db logs
   ```

3. **Test the setup**
   ```bash
   poetry run python scripts/test_llm_judge.py
   ```

## 🎯 Usage

### Basic Commands

#### Quick Evaluation (Recommended for development)
```bash
# Using Makefile
make llm-judge-quick

# Using script directly
./scripts/run_llm_judge.sh quick ollama 7.0

# Using poetry directly
poetry run python basicchat/evaluation/evaluators/check_llm_judge.py --quick
```

#### Full Evaluation (Comprehensive analysis)
```bash
# Using Makefile
make llm-judge

# Using script directly
./scripts/run_llm_judge.sh full ollama 7.0

# Using poetry directly
poetry run python basicchat/evaluation/evaluators/check_llm_judge.py
```

#### OpenAI Backend (Alternative)
```bash
# Set OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Run with OpenAI
make llm-judge-openai-quick
./scripts/run_llm_judge.sh quick openai 7.0
```

### Available Makefile Commands

| Command | Description |
|---------|-------------|
| `make llm-judge-quick` | Quick evaluation with Ollama |
| `make llm-judge` | Full evaluation with Ollama |
| `make llm-judge-openai-quick` | Quick evaluation with OpenAI |
| `make llm-judge-openai` | Full evaluation with OpenAI |
| `make test-and-evaluate` | Run tests + quick LLM judge |
| `make evaluate-all` | Run all tests + full LLM judge + performance test |

## 📊 Understanding Results

### Generated Files

After running an evaluation, you'll get several files:

1. **`llm_judge_results.json`** - Raw evaluation data
2. **`llm_judge_action_items.md`** - Actionable improvement plan
3. **`llm_judge_improvement_tips.md`** - Specific improvement tips
4. **`final_test_report.md`** - Combined test and evaluation report

### Score Interpretation

- **10/10**: Exemplary - Perfect adherence to best practices
- **8-9/10**: Excellent - Minor improvements needed
- **7-8/10**: Good - Some improvements needed
- **6-7/10**: Acceptable - Notable issues but functional
- **5-6/10**: Poor - Significant problems
- **<5/10**: Critical - Major issues requiring immediate attention

### Evaluation Categories

1. **Code Quality** - Structure, naming, complexity, Python best practices
2. **Test Coverage** - Comprehensiveness, quality, effectiveness
3. **Documentation** - README quality, inline docs, project documentation
4. **Architecture** - Design patterns, modularity, scalability
5. **Security** - Potential vulnerabilities, security best practices
6. **Performance** - Code efficiency, optimization opportunities

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_JUDGE_THRESHOLD` | `7.0` | Minimum passing score |
| `LLM_JUDGE_BACKEND` | `OLLAMA` | Backend to use (OLLAMA/OPENAI) |
| `OLLAMA_API_URL` | `http://localhost:11434/api` | Ollama API URL |
| `OLLAMA_MODEL` | `mistral` | Ollama model to use |
| `OPENAI_API_KEY` | - | OpenAI API key (required for OpenAI backend) |
| `OPENAI_MODEL` | `gpt-3.5-turbo` | OpenAI model to use |

### Rules Configuration

The evaluation rules are defined in `basicchat/evaluation/evaluators/llm_judge_rules.json`. You can customize:

- Evaluation criteria and weights
- Best practices guidelines
- File patterns and exclusions
- Consistency checks
- Priority levels

## 🛠️ Troubleshooting

### Common Issues

#### 1. "No module named 'basicchat'"
```bash
# Solution: Use poetry to run commands
poetry run python basicchat/evaluation/evaluators/check_llm_judge.py --quick
```

#### 2. "Ollama is not running"
```bash
# Solution: Start Ollama service
ollama serve

# Check if it's running
curl http://localhost:11434/api/tags
```

#### 3. "Model not found"
```bash
# Solution: Pull the required model
ollama pull mistral

# List available models
ollama list
```

#### 4. "Failed to parse JSON response"
This usually means the LLM response wasn't properly formatted. Try:
- Running again (temporary issue)
- Using a different model
- Checking Ollama logs

#### 5. "Evaluation failed with exit code"
Check the detailed error message. Common causes:
- Ollama not running
- Model not available
- Network connectivity issues

### Debug Mode

Enable debug mode for more detailed output:

```bash
export LLM_JUDGE_DEBUG=1
poetry run python basicchat/evaluation/evaluators/check_llm_judge.py --quick
```

### Logs

Check Ollama logs for issues:
```bash
# View Ollama logs
ollama logs

# Check system logs
journalctl -u ollama -f
```

## 🔄 Continuous Integration

The LLM Judge is integrated into the CI pipeline and runs:

- On every push to main branch
- On pull requests from the same repository
- After unit tests pass
- With fallback to OpenAI if Ollama fails

### CI Configuration

The CI configuration is in `.github/workflows/verify.yml` and includes:

- LLM Judge evaluation job
- Automatic fallback to OpenAI
- Artifact upload for results
- Integration with final test reports

## 📈 Best Practices

### For Development

1. **Run quick evaluations frequently** during development
2. **Address critical issues immediately** (score < 6)
3. **Plan to fix high priority issues** (score 6-7)
4. **Use the action items** as a development roadmap
5. **Run full evaluations** before major releases

### For Teams

1. **Set up local development** for all team members
2. **Use consistent thresholds** across the team
3. **Review action items** in team meetings
4. **Track progress** over time
5. **Customize rules** for your project needs

### For CI/CD

1. **Set appropriate thresholds** for your project
2. **Use quick mode** for faster feedback
3. **Configure fallback** to OpenAI for reliability
4. **Upload artifacts** for review
5. **Integrate with** existing quality gates

## 🎯 Next Steps

1. **Run the setup script**: `./scripts/setup_local_llm_judge.sh`
2. **Try a quick evaluation**: `make llm-judge-quick`
3. **Review the action items**: Check `llm_judge_action_items.md`
4. **Implement improvements**: Follow the prioritized action plan
5. **Run regularly**: Integrate into your development workflow

## 📚 Additional Resources

- [LLM Judge Evaluator Documentation](EVALUATORS.md)
- [Evaluation Rules Configuration](../basicchat/evaluation/evaluators/llm_judge_rules.json)
- [GitHub Actions Workflow](../.github/workflows/verify.yml)
- [Makefile Commands](../Makefile)

---

*This guide covers local development setup. For production deployment and CI/CD integration, see the main [EVALUATORS.md](EVALUATORS.md) documentation.*
