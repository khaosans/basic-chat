# Testing Strategy for BasicChat

## Overview

This document outlines our resilient testing strategy that separates concerns and reduces flakiness in CI/CD pipelines.

## Test Categories

### 1. Unit Tests (`@pytest.mark.unit`, `@pytest.mark.fast`)
- **Purpose**: Test individual functions and classes in isolation
- **Characteristics**: 
  - Fast execution (< 1 second per test)
  - No external dependencies
  - Fully mocked external services
  - Can run in parallel
- **Examples**: Core utilities, tool functions, data processing

### 2. Integration Tests (`@pytest.mark.integration`)
- **Purpose**: Test interactions between components
- **Characteristics**:
  - Require external services (Ollama, ChromaDB)
  - Slower execution (5-30 seconds per test)
  - Run sequentially to avoid conflicts
  - Use real or mocked external services
- **Examples**: Document processing, reasoning engine, web search

### 3. Slow Tests (`@pytest.mark.slow`)
- **Purpose**: Test LLM interactions and heavy processing
- **Characteristics**:
  - Very slow execution (30+ seconds per test)
  - Require API keys and external services
  - Run separately from other tests
  - Can be expensive (API costs)
- **Examples**: LLM Judge evaluations, full reasoning workflows

## CI/CD Pipeline Strategy

### GitHub Actions Workflow

```yaml
jobs:
  unit-tests:          # Fast, reliable, always run
    - Runs on every PR
    - Parallel execution
    - 2-3 minute execution time
    
  integration-tests:   # Slower, conditional
    - Runs on main branch only
    - Can be triggered with [run-integration] in commit message
    - Sequential execution
    - 5-10 minute execution time
    
  llm-judge:          # Expensive, conditional
    - Runs on main branch only
    - Requires API keys
    - 2-5 minute execution time
```

### Benefits

1. **Fast Feedback**: Unit tests provide immediate feedback on PRs
2. **Cost Control**: Integration and slow tests only run when needed
3. **Reliability**: Reduced flakiness from external service dependencies
4. **Scalability**: Parallel execution for fast tests

## Local Development

### Test Runner Script

Use the test runner for different scenarios:

```bash
# Run only unit tests (fastest)
python scripts/run_tests.py --mode unit --parallel

# Run integration tests (requires external services)
python scripts/run_tests.py --mode integration

# Run all tests with coverage
python scripts/run_tests.py --mode all --coverage --verbose

# Run slow tests only
python scripts/run_tests.py --mode slow --timeout 300
```

### Environment Setup

```bash
# For unit tests (no external services needed)
export MOCK_EXTERNAL_SERVICES=true

# For integration tests (requires Ollama, ChromaDB)
export MOCK_EXTERNAL_SERVICES=false
export OLLAMA_BASE_URL=http://localhost:11434
export CHROMA_PERSIST_DIR=./test_chroma_db
```

## Test Organization

### File Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── test_core.py            # Unit tests (fast)
├── test_tools.py           # Unit tests (fast)
├── test_audio.py           # Unit tests (fast)
├── test_voice.py           # Unit tests (fast)
├── test_documents.py       # Integration tests
├── test_reasoning.py       # Integration tests
├── test_web_search.py      # Integration tests
├── test_upload.py          # Integration tests
├── test_llm_judge.py       # Slow tests
└── data/                   # Test data files
```

### Marking Tests

```python
# Unit test (fast, isolated)
@pytest.mark.unit
@pytest.mark.fast
def test_calculator_function():
    assert calculate(2, 3) == 5

# Integration test (requires external services)
@pytest.mark.integration
def test_document_processing():
    # Tests with real ChromaDB, Ollama
    pass

# Slow test (LLM calls)
@pytest.mark.slow
def test_llm_judge_evaluation():
    # Tests with OpenAI API
    pass
```

## Mocking Strategy

### External Services

```python
@pytest.fixture(scope="function")
def mock_external_services():
    """Mock external services for unit tests."""
    with patch('document_processor.OllamaEmbeddings') as mock_embeddings, \
         patch('document_processor.ChatOllama') as mock_chat, \
         patch('document_processor.chromadb.PersistentClient') as mock_chroma:
        
        # Configure mocks
        mock_embeddings.return_value = Mock()
        mock_chat.return_value = Mock()
        mock_chroma.return_value = Mock()
        
        yield {
            'embeddings': mock_embeddings,
            'chat': mock_chat,
            'chroma': mock_chroma
        }
```

### File System Operations

```python
@pytest.fixture(scope="function")
def mock_file_system():
    """Mock file system operations for isolated tests."""
    with patch('builtins.open') as mock_open, \
         patch('os.path.exists') as mock_exists:
        
        mock_exists.return_value = True
        yield {
            'open': mock_open,
            'exists': mock_exists
        }
```

## Best Practices

### 1. Test Isolation
- Each test should be independent
- Use fixtures for setup/teardown
- Clean up resources after tests

### 2. Mocking
- Mock external services in unit tests
- Use real services only in integration tests
- Provide realistic mock responses

### 3. Test Data
- Use minimal, realistic test data
- Create test data programmatically when possible
- Store large test files in `tests/data/`

### 4. Error Handling
- Test both success and failure scenarios
- Verify error messages and types
- Test edge cases and boundary conditions

### 5. Performance
- Keep unit tests under 1 second
- Use timeouts for integration tests
- Monitor test execution times

## Troubleshooting

### Common Issues

1. **Flaky Tests**: Usually caused by external service dependencies
   - **Solution**: Move to integration tests, add proper mocking

2. **Slow Tests**: Tests taking too long
   - **Solution**: Add timeouts, optimize test data, use parallel execution

3. **Resource Conflicts**: Tests interfering with each other
   - **Solution**: Use isolated test environments, clean up resources

4. **Missing Dependencies**: Tests failing due to missing services
   - **Solution**: Add proper mocking, check environment variables

### Debugging

```bash
# Run tests with verbose output
python scripts/run_tests.py --mode unit --verbose

# Run specific test file
pytest tests/test_core.py -v

# Run tests with debugging
pytest tests/ -s --pdb

# Check test markers
pytest --markers
```

## Metrics and Monitoring

### Key Metrics

1. **Test Execution Time**
   - Unit tests: < 5 minutes total
   - Integration tests: < 15 minutes total
   - Slow tests: < 10 minutes total

2. **Test Reliability**
   - Unit tests: > 99% pass rate
   - Integration tests: > 95% pass rate
   - Slow tests: > 90% pass rate

3. **Coverage**
   - Target: > 80% code coverage
   - Focus on critical paths
   - Exclude generated code

### Monitoring

- Track test execution times in CI
- Monitor flaky test frequency
- Review coverage reports regularly
- Update test strategy based on metrics 