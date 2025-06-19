# Development Guide

[← Back to README](../README.md) | [Installation ←](INSTALLATION.md) | [Features ←](FEATURES.md) | [Architecture ←](ARCHITECTURE.md) | [Roadmap →](ROADMAP.md)

---

## Overview
This guide provides comprehensive information for developers who want to contribute to BasicChat, including setup, testing, development workflows, and best practices.

## Quick Start

### Setup Development Environment
```bash
# Clone and setup
git clone https://github.com/khaosans/basic-chat-template.git
cd basic-chat-template

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-asyncio pytest-cov black flake8 mypy
```

### Run Tests
```bash
# Complete test suite
pytest

# With coverage
pytest --cov=app --cov-report=html

# Specific test categories
pytest tests/test_basic.py      # Core functionality
pytest tests/test_reasoning.py  # Reasoning engine
pytest tests/test_processing.py # Document processing
pytest tests/test_web_search.py # Web search integration
pytest tests/test_enhanced_tools.py # Enhanced tools
```

## Project Structure

```
basic-chat-template/
├── app.py                      # Main Streamlit application
├── config.py                   # Configuration management
├── reasoning_engine.py         # Advanced reasoning capabilities
├── document_processor.py       # Document processing and RAG
├── web_search.py              # Web search integration
├── ollama_api.py              # Ollama API utilities
├── utils/                      # Enhanced utilities and tools
│   ├── __init__.py
│   ├── async_ollama.py        # High-performance async Ollama client
│   ├── caching.py             # Multi-layer caching system
│   └── enhanced_tools.py      # Enhanced calculator and time tools
├── tests/                      # Comprehensive test suite
│   ├── test_basic.py          # Core functionality tests
│   ├── test_reasoning.py      # Reasoning engine tests
│   ├── test_processing.py     # Document processing tests
│   ├── test_enhanced_tools.py # Enhanced tools tests
│   ├── test_web_search.py     # Web search tests
│   └── conftest.py            # Test configuration
├── docs/                       # Documentation
├── tickets/                    # Development tickets
└── requirements.txt            # Python dependencies
```

## Key Components

### Configuration Management (`config.py`)
- Environment-based configuration with Pydantic validation
- Type-safe settings with dataclass validation
- Centralized configuration management

### Async Ollama Client (`utils/async_ollama.py`)
- Connection pooling with aiohttp
- Rate limiting and retry logic
- Streaming support and health monitoring
- Async/await patterns throughout

### Caching System (`utils/caching.py`)
- Multi-layer caching (Redis + Memory)
- Smart cache key generation with MD5
- TTL and size management with fallback

### Reasoning Engine (`reasoning_engine.py`)
- Chain-of-Thought reasoning implementation
- Multi-step analysis with RAG integration
- Agent-based tools with registry pattern
- Confidence scoring and streaming

## Development Workflow

### Code Quality
```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .

# Run all quality checks
black . && flake8 . && mypy .
```

### Testing Strategy
- **Unit Tests**: Individual component testing
- **Integration Tests**: Component interaction testing
- **Async Tests**: Performance and async functionality
- **Mock Tests**: External dependency isolation

### Test Coverage
- **46+ tests** covering all major components
- **80%+ coverage** with detailed reporting
- **Async test support** for performance components
- **Mock integration** for external dependencies

## Contributing Guidelines

### 1. Fork and Clone
```bash
git clone https://github.com/your-username/basic-chat-template.git
cd basic-chat-template
```

### 2. Create Feature Branch
```bash
git checkout -b feature/amazing-feature
```

### 3. Development Process
- Write tests for new functionality
- Implement features with type hints
- Add documentation for new features
- Ensure all tests pass

### 4. Code Review Checklist
- [ ] All tests pass
- [ ] Code follows style guidelines
- [ ] Type hints added
- [ ] Documentation updated
- [ ] No breaking changes

### 5. Submit Pull Request
- Clear description of changes
- Link to related issues
- Include test coverage
- Update documentation

## Testing Guidelines

### Writing Tests
```python
# Example test structure
import pytest
from utils.async_ollama import AsyncOllamaChat

@pytest.mark.asyncio
async def test_async_chat_query():
    """Test async chat query functionality"""
    chat = AsyncOllamaChat("mistral")
    response = await chat.query({"inputs": "Hello"})
    assert response is not None
    assert isinstance(response, str)
```

### Test Categories
- **Unit Tests**: Test individual functions/methods
- **Integration Tests**: Test component interactions
- **Performance Tests**: Test async and caching performance
- **Error Tests**: Test error handling and edge cases

### Mock External Dependencies
```python
import pytest
from unittest.mock import patch

@patch('utils.async_ollama.requests.post')
def test_ollama_api_mock(mock_post):
    """Test with mocked external API"""
    mock_post.return_value.json.return_value = {"response": "test"}
    # Test implementation
```

## Performance Development

### Async Best Practices
- Use `async/await` consistently
- Implement proper resource cleanup
- Handle connection pooling efficiently
- Use appropriate timeouts

### Caching Development
- Implement cache key strategies
- Handle cache invalidation
- Monitor cache performance
- Test fallback mechanisms

### Memory Management
- Monitor memory usage in tests
- Implement proper cleanup
- Use generators for large datasets
- Profile memory-intensive operations

## Debugging

### Debug Mode
```bash
# Enable debug logging
LOG_LEVEL=DEBUG
ENABLE_STRUCTURED_LOGGING=true

# Run with verbose output
streamlit run app.py --logger.level=debug
```

### Common Debug Scenarios
- **Async Issues**: Check event loop and coroutines
- **Cache Problems**: Verify cache keys and TTL
- **Performance Issues**: Profile async operations
- **Memory Leaks**: Monitor resource cleanup

### Debug Tools
- **pytest --pdb**: Drop into debugger on failures
- **logging.debug()**: Add debug statements
- **time.perf_counter()**: Performance timing
- **memory_profiler**: Memory usage analysis

## Documentation

### Code Documentation
- Use docstrings for all public functions
- Include type hints for all parameters
- Document async functions clearly
- Add examples for complex operations

### API Documentation
- Document all public APIs
- Include request/response examples
- Document error conditions
- Maintain up-to-date examples

### Architecture Documentation
- Keep architecture diagrams current
- Document design decisions
- Update component descriptions
- Maintain performance metrics

## Deployment

### Local Development
```bash
# Start Ollama
ollama serve &

# Run application
streamlit run app.py
```

### Production Considerations
- Environment variable configuration
- Redis setup for caching
- Health check implementation
- Monitoring and logging setup

### Docker Development
```bash
# Build image
docker build -t basic-chat .

# Run container
docker run -p 8501:8501 basic-chat
```

## Support

### Getting Help
- Check existing documentation
- Review test examples
- Search GitHub issues
- Join discussions

### Reporting Issues
- Include reproduction steps
- Provide system information
- Attach relevant logs
- Describe expected behavior

## 🔗 Related Documentation

- **[Installation Guide](INSTALLATION.md)** - Setup and configuration
- **[Features Overview](FEATURES.md)** - Detailed feature documentation
- **[System Architecture](ARCHITECTURE.md)** - Technical design and components
- **[Production Roadmap](ROADMAP.md)** - Future development plans
- **[Reasoning Features](../REASONING_FEATURES.md)** - Advanced reasoning engine details

## 📚 References

### Development Tools
- **pytest**: [https://pytest.org](https://pytest.org) - Testing framework for Python
- **pytest-asyncio**: [https://pytest-asyncio.readthedocs.io](https://pytest-asyncio.readthedocs.io) - Async support for pytest
- **Pydantic**: [https://pydantic.dev](https://pydantic.dev) - Data validation using Python type annotations
- **Black**: [https://black.readthedocs.io](https://black.readthedocs.io) - Code formatter

### Best Practices
- **SOLID Principles**: Clean Architecture by Robert C. Martin
- **Async Programming**: Python asyncio documentation
- **Testing**: Test-Driven Development by Kent Beck
- **Code Quality**: Clean Code by Robert C. Martin

---

[← Back to README](../README.md) | [Installation ←](INSTALLATION.md) | [Features ←](FEATURES.md) | [Architecture ←](ARCHITECTURE.md) | [Roadmap →](ROADMAP.md) 