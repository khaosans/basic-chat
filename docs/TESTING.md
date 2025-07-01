# Testing Guide

[← Back to README](../README.md)

## Overview

The BasicChat project includes a comprehensive test suite covering all major functionality including core chat operations, reasoning engines, document processing, enhanced tools, web search, and audio generation.

## Test Structure

### Test Files

- `tests/test_core.py` - Core OllamaChat functionality and tool registry
- `tests/test_reasoning.py` - Reasoning engine, agents, and chains
- `tests/test_documents.py` - Document processing and search
- `tests/test_tools.py` - Enhanced calculator and time tools
- `tests/test_web_search.py` - Web search functionality
- `tests/test_audio.py` - Audio generation and HTML creation

### Test Categories

- **Unit Tests** - Fast, isolated tests with mocked dependencies
- **Integration Tests** - Tests that verify component interactions
- **CI Tests** - Tests specifically designed for CI environment

## Running Tests

### Local Development

```bash
# Run all tests
python -m pytest tests/

# Run with verbose output
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_core.py

# Run specific test class
python -m pytest tests/test_core.py::TestOllamaChat

# Run with coverage (no threshold)
python -m pytest tests/ --cov=app --cov=reasoning_engine --cov=document_processor --cov=utils --cov=web_search --cov-report=term-missing
```

### CI/CD Pipeline

The GitHub Actions workflow (`/.github/workflows/verify.yml`) runs:
- Python 3.11 compatibility tests
- All unit tests with coverage reporting
- Basic functionality import tests
- Configuration validation

## Test Coverage

Current coverage: **50%**

### Coverage by Module

- `app.py`: 35% - Main application logic
- `reasoning_engine.py`: 43% - Reasoning and agent functionality
- `document_processor.py`: 56% - Document processing and search
- `utils/async_ollama.py`: 52% - Async Ollama communication
- `utils/caching.py`: 43% - Caching functionality
- `utils/enhanced_tools.py`: 90% - Calculator and time tools
- `web_search.py`: 91% - Web search functionality

### Coverage Goals

While the current test suite provides comprehensive functional testing, coverage is lower due to:
- Complex UI/UX code paths in `app.py`
- Error handling and edge cases
- Async operations and threading
- External service integrations

## Test Patterns

### Mocking Strategy

- **External Services**: All external API calls are mocked
- **File Operations**: File I/O is mocked to avoid side effects
- **Threading**: Threading operations are mocked for predictable testing
- **Async Operations**: Async functions are properly mocked with realistic responses

### Parameterized Tests

Many tests use pytest's `@pytest.mark.parametrize` for comprehensive input validation:

```python
@pytest.mark.parametrize("expression,expected", [
    ("2 + 2", 4.0),
    ("10 - 5", 5.0),
    ("3 * 4", 12.0),
])
def test_calculator(expression, expected):
    # Test implementation
```

### Fixtures

Common fixtures are defined in `tests/conftest.py`:
- Mock Ollama responses
- Sample documents
- Test configurations

## CI/CD Integration

### GitHub Actions

The CI pipeline:
1. Sets up Python 3.11 environment
2. Installs dependencies from `requirements.txt`
3. Runs all tests with coverage reporting
4. Validates core module imports
5. Tests configuration loading

### Coverage Reporting

Coverage is reported but not enforced as a failure threshold, allowing for:
- Gradual coverage improvement
- Focus on functional correctness
- Realistic development workflow

## Best Practices

### Writing Tests

1. **Test Naming**: Use descriptive names that explain the scenario
2. **Arrange-Act-Assert**: Structure tests clearly
3. **Mock External Dependencies**: Avoid real API calls
4. **Test Edge Cases**: Include error conditions and boundary values
5. **Use Parameterized Tests**: For multiple input scenarios

### Example Test Structure

```python
class TestFeature:
    def test_should_handle_normal_case(self):
        # Arrange
        input_data = "test"
        
        # Act
        result = process_data(input_data)
        
        # Assert
        assert result == expected_output
    
    def test_should_handle_error_case(self):
        # Arrange
        input_data = None
        
        # Act & Assert
        with pytest.raises(ValueError):
            process_data(input_data)
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Mock Issues**: Verify mock objects match expected interfaces
3. **Async Warnings**: Some async warnings are expected and can be ignored
4. **Coverage Gaps**: Focus on functional testing over coverage metrics

### Debugging Tests

```bash
# Run with detailed output
python -m pytest tests/ -v -s

# Run single test with debugger
python -m pytest tests/test_core.py::test_specific -s --pdb

# Check test discovery
python -m pytest --collect-only
```

## Future Improvements

1. **Increase Coverage**: Add tests for uncovered code paths
2. **Integration Tests**: Add end-to-end testing scenarios
3. **Performance Tests**: Add benchmarks for critical operations
4. **Property-Based Testing**: Use hypothesis for more thorough testing
5. **Visual Regression Tests**: For UI components when applicable 
