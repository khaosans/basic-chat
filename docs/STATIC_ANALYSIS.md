# Static Analysis Setup

This document describes the comprehensive static analysis setup for the basic-chat-template project, designed to catch issues at compile time and maintain high code quality.

## Overview

We've implemented a multi-layered static analysis approach that includes:

- **Code Formatting**: Black, isort, autoflake
- **Linting**: flake8, pylint
- **Type Checking**: mypy, pyright
- **Security**: bandit, safety, semgrep
- **Complexity Analysis**: radon, xenon
- **Testing**: pytest with coverage
- **Pre-commit Hooks**: Automated checks before commits
- **CI/CD**: GitHub Actions workflow

## Quick Start

### 1. Install Development Dependencies

```bash
# Install all development tools
pip install -r requirements-dev.txt

# Or use the Makefile
make install-dev
```

### 2. Set Up Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Or use the Makefile
make git-hooks
```

### 3. Run All Checks

```bash
# Run comprehensive static analysis
make lint

# Run with auto-fix where possible
make lint-fix

# Run all checks (lint + test + security)
make check-all
```

## Available Tools

### Code Formatting

#### Black
- **Purpose**: Uncompromising code formatter
- **Command**: `black .`
- **Check only**: `black --check --diff .`
- **Auto-fix**: `make black-fix`

#### isort
- **Purpose**: Import sorting and organization
- **Command**: `isort .`
- **Check only**: `isort --check-only --diff .`
- **Auto-fix**: `make isort-fix`

#### autoflake
- **Purpose**: Remove unused imports and variables
- **Command**: `autoflake --in-place --remove-all-unused-imports --remove-unused-variables .`
- **Auto-fix**: Included in `make format`

### Linting

#### flake8
- **Purpose**: Style guide enforcement
- **Command**: `flake8 .`
- **Configuration**: `.flake8` or `pyproject.toml`

#### pylint
- **Purpose**: Code quality and error detection
- **Command**: `pylint --rcfile=pyproject.toml .`
- **Configuration**: `pyproject.toml`

### Type Checking

#### mypy
- **Purpose**: Static type checking
- **Command**: `mypy .`
- **Configuration**: `pyproject.toml`
- **Strict mode**: Configured for maximum type safety

#### pyright
- **Purpose**: Fast type checking (alternative to mypy)
- **Command**: `pyright .`
- **Configuration**: `pyproject.toml`

### Security

#### bandit
- **Purpose**: Security linting
- **Command**: `bandit -r . -f json -o bandit-report.json`
- **Configuration**: `pyproject.toml`

#### safety
- **Purpose**: Check for known security vulnerabilities in dependencies
- **Command**: `safety check`

#### semgrep
- **Purpose**: Advanced security scanning
- **Command**: `semgrep --config=p/security-audit .`
- **Configuration**: Uses Semgrep's security rules

### Complexity Analysis

#### radon
- **Purpose**: Code complexity metrics
- **Command**: `radon cc . -a -nc`
- **Configuration**: `pyproject.toml`

#### xenon
- **Purpose**: Complexity threshold enforcement
- **Command**: `xenon . --max-absolute=A --max-modules=A --max-average=A`
- **Configuration**: `pyproject.toml`

### Testing

#### pytest
- **Purpose**: Unit and integration testing
- **Command**: `pytest`
- **With coverage**: `pytest --cov=. --cov-report=html --cov-report=term-missing`
- **Fast mode**: `pytest -x --tb=short`

## Configuration Files

### pyproject.toml
Central configuration file for most tools:
- Black formatting settings
- isort import sorting
- mypy type checking
- pylint rules
- pytest configuration
- Coverage settings

### .pre-commit-config.yaml
Pre-commit hooks configuration:
- Automatic formatting on commit
- Linting checks
- Security scans
- Test execution

### Makefile
Convenient commands for common tasks:
- `make lint`: Run all static analysis
- `make format`: Auto-format code
- `make test`: Run tests
- `make check-all`: Complete quality check

## Custom Linting Script

### scripts/lint.py
Comprehensive static analysis orchestrator:

```bash
# Run all tools
python scripts/lint.py

# Run with auto-fix
python scripts/lint.py --fix

# Run specific tools
python scripts/lint.py --tools mypy,flake8

# Verbose output
python scripts/lint.py --verbose

# Save report
python scripts/lint.py --output lint-report.txt
```

## Pre-commit Hooks

Pre-commit hooks automatically run checks before each commit:

1. **Code formatting**: Black, isort, autoflake
2. **Linting**: flake8, pylint
3. **Type checking**: mypy
4. **Security**: bandit, safety
5. **Complexity**: xenon
6. **Tests**: pytest

### Installation
```bash
pre-commit install
pre-commit install --hook-type commit-msg
```

### Manual Run
```bash
pre-commit run --all-files
```

## CI/CD Integration

### GitHub Actions
Automated checks on every push and pull request:

1. **Static Analysis**: Runs on Python 3.9, 3.10, 3.11
2. **Testing**: Full test suite with coverage
3. **Security**: Semgrep security scanning
4. **Build**: Package building and Docker image creation

### Local CI Simulation
```bash
# Run the same checks as CI
make ci
```

## Common Issues and Solutions

### Import Errors in mypy
If mypy reports import errors for third-party libraries:

```python
# Add to pyproject.toml under [tool.mypy.overrides]
[[tool.mypy.overrides]]
module = ["problematic_module.*"]
ignore_missing_imports = true
```

### False Positives in pylint
To disable specific pylint warnings:

```python
# In pyproject.toml under [tool.pylint.messages_control]
disable = [
    "C0114",  # missing-module-docstring
    "R0903",  # too-few-public-methods
]
```

### Complex Functions
If xenon reports overly complex functions:

1. Break down large functions into smaller ones
2. Extract helper methods
3. Use early returns to reduce nesting
4. Consider using the Strategy pattern

### Type Annotations
For better mypy compliance:

```python
from typing import Optional, List, Dict, Any

def process_data(data: List[Dict[str, Any]]) -> Optional[str]:
    """Process data with proper type annotations."""
    if not data:
        return None
    # ... rest of function
```

## Best Practices

### 1. Run Checks Frequently
```bash
# Before committing
make lint

# Before pushing
make check-all
```

### 2. Fix Issues Early
- Address linting issues immediately
- Don't accumulate technical debt
- Use auto-fix tools when possible

### 3. Maintain Type Safety
- Add type annotations to all functions
- Use mypy in strict mode
- Avoid `Any` types when possible

### 4. Keep Complexity Low
- Aim for cyclomatic complexity < 10
- Break down complex functions
- Use helper functions and classes

### 5. Security First
- Review bandit and safety reports
- Keep dependencies updated
- Follow security best practices

## IDE Integration

### VS Code
Install these extensions for better integration:
- Python
- Pylint
- Black Formatter
- isort
- mypy Type Checker

### PyCharm
Configure external tools:
- Black: `black $FilePath$`
- isort: `isort $FilePath$`
- mypy: `mypy $FilePath$`

## Troubleshooting

### Tool Not Found
```bash
# Install missing tool
pip install tool-name

# Or install all dev dependencies
pip install -r requirements-dev.txt
```

### Configuration Issues
- Check `pyproject.toml` for correct settings
- Verify tool-specific configuration files
- Ensure Python version compatibility

### Performance Issues
- Use `--fast` flags where available
- Run specific tools instead of all
- Use caching (pip cache, mypy cache)

### False Positives
- Configure tool-specific ignore patterns
- Use inline comments to suppress warnings
- Report issues to tool maintainers

## Contributing

When contributing to this project:

1. **Follow the linting rules**: All code must pass static analysis
2. **Add type annotations**: Use mypy-compatible types
3. **Write tests**: Maintain good test coverage
4. **Document changes**: Update relevant documentation
5. **Run checks locally**: Don't rely only on CI

## Resources

- [Black Documentation](https://black.readthedocs.io/)
- [mypy Documentation](https://mypy.readthedocs.io/)
- [pylint Documentation](https://pylint.pycqa.org/)
- [bandit Documentation](https://bandit.readthedocs.io/)
- [pre-commit Documentation](https://pre-commit.com/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions) 