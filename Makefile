test-fast:
	pytest -n auto -m "unit or fast"

test-all:
	pytest -n auto

test-last-failed:
	pytest --last-failed || pytest -n auto

# LLM Judge Evaluation Commands
llm-judge:
	@echo "🤖 Running LLM Judge evaluation (Ollama, full mode)..."
	@chmod +x scripts/run_llm_judge.sh
	@./scripts/run_llm_judge.sh full ollama 7.0

llm-judge-quick:
	@echo "🤖 Running LLM Judge evaluation (Ollama, quick mode)..."
	@chmod +x scripts/run_llm_judge.sh
	@./scripts/run_llm_judge.sh quick ollama 7.0

llm-judge-openai:
	@echo "🤖 Running LLM Judge evaluation (OpenAI, full mode)..."
	@chmod +x scripts/run_llm_judge.sh
	@./scripts/run_llm_judge.sh full openai 7.0

llm-judge-openai-quick:
	@echo "🤖 Running LLM Judge evaluation (OpenAI, quick mode)..."
	@chmod +x scripts/run_llm_judge.sh
	@./scripts/run_llm_judge.sh quick openai 7.0

# Performance regression test
perf-test:
	@echo "⚡ Running performance regression test..."
	@poetry run python scripts/test_performance_regression.py

# Combined test and evaluation
test-and-evaluate: test-fast llm-judge-quick
	@echo "✅ Tests and LLM Judge evaluation completed!"

# Full evaluation pipeline
evaluate-all: test-all llm-judge perf-test
	@echo "✅ Full evaluation pipeline completed!" 