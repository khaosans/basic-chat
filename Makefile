.PHONY: help install install-dev lint lint-fix test test-cov clean format check-all ci

# Default target
help:
	@echo "Available commands:"
	@echo "  install      - Install production dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  lint         - Run all static analysis tools"
	@echo "  lint-fix     - Run linting tools that can auto-fix"
	@echo "  format       - Format code with black and isort"
	@echo "  test         - Run tests"
	@echo "  test-cov     - Run tests with coverage"
	@echo "  check-all    - Run lint, test, and security checks"
	@echo "  ci           - Run CI pipeline (lint + test + security)"
	@echo "  clean        - Clean up generated files"
	@echo "  run          - Run the Streamlit app"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pre-commit install

# Linting and formatting
lint:
	@echo "Running comprehensive static analysis..."
	python scripts/lint.py --verbose

lint-fix:
	@echo "Running linting tools with auto-fix..."
	python scripts/lint.py --fix --verbose

format:
	@echo "Formatting code..."
	black .
	isort .
	autoflake --in-place --remove-all-unused-imports --remove-unused-variables .

# Individual tools
black:
	black --check --diff .

black-fix:
	black .

isort:
	isort --check-only --diff .

isort-fix:
	isort .

flake8:
	flake8 .

mypy:
	mypy .

pylint:
	pylint --rcfile=pyproject.toml .

bandit:
	bandit -r . -f json -o bandit-report.json

safety:
	safety check

# Testing
test:
	pytest

test-cov:
	pytest --cov=. --cov-report=html --cov-report=term-missing

test-fast:
	pytest -x --tb=short

# Complexity analysis
complexity:
	radon cc . -a -nc
	xenon . --max-absolute=A --max-modules=A --max-average=A

# Security checks
security:
	bandit -r . -f json -o bandit-report.json
	safety check
	semgrep --config=p/security-audit .

# Comprehensive checks
check-all: lint test security complexity
	@echo "All checks completed!"

# CI pipeline
ci: lint test security
	@echo "CI pipeline completed!"

# Development
run:
	streamlit run app.py

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type f -name "bandit-report.json" -delete
	find . -type f -name "lint-report.txt" -delete
	find . -type f -name "temp_*.mp3" -delete
	@echo "Cleanup completed!"

# Quick development workflow
dev-setup: install-dev
	@echo "Development environment setup complete!"
	@echo "Run 'make run' to start the app"
	@echo "Run 'make lint' to check code quality"
	@echo "Run 'make test' to run tests"

# Pre-commit hooks
pre-commit-install:
	pre-commit install

pre-commit-run:
	pre-commit run --all-files

# Documentation
docs:
	sphinx-build -b html docs/source docs/build/html

# Docker (if needed)
docker-build:
	docker build -t basic-chat-template .

docker-run:
	docker run -p 8501:8501 basic-chat-template

# Performance profiling
profile:
	python -m cProfile -o profile.stats app.py

profile-view:
	python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"

# Dependencies
update-deps:
	pip install --upgrade -r requirements.txt
	pip install --upgrade -r requirements-dev.txt

check-deps:
	safety check
	pip list --outdated

# Git hooks
git-hooks:
	@echo "Setting up git hooks..."
	pre-commit install
	pre-commit install --hook-type commit-msg

# Environment setup
env-setup:
	@echo "Setting up development environment..."
	python -m venv venv
	@echo "Virtual environment created. Activate it with:"
	@echo "  source venv/bin/activate  # On Unix/macOS"
	@echo "  venv\\Scripts\\activate     # On Windows"
	@echo "Then run: make install-dev" 