# Development Guide

This guide provides instructions for developers contributing to BasicChat. It covers setting up the environment, running tests, maintaining code quality, and managing the database.

[← Back to README](../README.md)

## 🚀 Development Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/khaosans/basic-chat-template.git
    cd basic-chat-template
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\activate
    ```

3.  **Install dependencies**:
    This includes the core application requirements and development tools like `pytest` and `black`.
    ```bash
    pip install -r requirements.txt
    pip install pytest pytest-asyncio pytest-cov black flake8 mypy
    ```

## ✅ Testing

We use `pytest` for testing. The test suite covers unit, integration, and asynchronous functionality.

-   **Run all tests**:
    ```bash
    pytest
    ```

-   **Run tests with coverage report**:
    ```bash
    pytest --cov=app --cov-report=html
    ```

-   **Run specific tests**:
    ```bash
    # Test core functionality
    pytest tests/test_basic.py
    
    # Test the reasoning engine
    pytest tests/test_reasoning.py
    ```

## ✨ Code Quality

We use `black`, `flake8`, and `mypy` to ensure code quality and consistency.

-   **Format code**:
    ```bash
    black .
    ```
-   **Lint code**:
    ```bash
    flake8 .
    ```
-   **Check types**:
    ```bash
    mypy .
    ```

Please run these checks before submitting a pull request.

## 🗄️ ChromaDB Database Management

The application uses ChromaDB to store vector embeddings for documents. Over time, development and testing can create multiple database directories.

### Cleanup Script

A cleanup script is provided at `scripts/cleanup_chroma.py` to help manage these directories.

-   **Check database status**:
    See a report of all `chroma_db*` directories, including their size and age.
    ```bash
    python scripts/cleanup_chroma.py --status
    ```

-   **Perform a dry run**:
    See which directories would be deleted without actually deleting them.
    ```bash
    python scripts/cleanup_chroma.py --dry-run
    ```

-   **Clean up all directories**:
    **Warning**: This will permanently delete all ChromaDB data.
    ```bash
    python scripts/cleanup_chroma.py --force
    ```

-   **Clean up old directories**:
    Remove directories older than a specified number of hours (e.g., 24 hours).
    ```bash
    python scripts/cleanup_chroma.py --age 24
    ```

## 🤝 Contribution Guidelines

1.  **Create a branch**: Create a feature branch for your work.
    ```bash
    git checkout -b feature/my-new-feature
    ```
2.  **Develop**: Write your code and add corresponding tests.
3.  **Test and lint**: Ensure all tests pass and the code meets our quality standards.
4.  **Submit a pull request**: Push your branch to your fork and open a pull request with a clear description of your changes. 