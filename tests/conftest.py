"""
Shared test fixtures and configuration for streamlined test suite
CHANGELOG:
- Consolidated all fixtures into single file
- Added comprehensive mock objects for external dependencies
- Removed redundant fixtures from individual test files
"""

import pytest
import os
import sys
import tempfile
import asyncio
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
from io import BytesIO
from PIL import Image
import shutil

# Add the project root to Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def test_config():
    """Global test configuration"""
    return {
        "temp_dir": tempfile.mkdtemp(),
        "test_data_dir": Path(__file__).parent / "data",
        "parallel_safe": True
    }

@pytest.fixture(scope="function")
def temp_dir(test_config):
    """Create a temporary directory for each test"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)

@pytest.fixture(scope="function")
def isolated_test_env():
    """Create isolated environment for tests that can't run in parallel"""
    # Save original environment
    original_env = os.environ.copy()
    
    # Set test-specific environment
    os.environ['TESTING'] = 'true'
    os.environ['PYTHONPATH'] = str(Path(__file__).parent.parent)
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)

@pytest.fixture(scope="session")
def parallel_safe():
    """Mark tests as safe for parallel execution"""
    return True

@pytest.fixture
def sample_text():
    """Sample text for testing"""
    return "This is a sample text for testing purposes."

@pytest.fixture
def sample_pdf_path(test_config):
    """Path to sample PDF file"""
    return test_config["test_data_dir"] / "test_sample.pdf"

@pytest.fixture
def sample_image_path(test_config):
    """Path to sample image file"""
    return test_config["test_data_dir"] / "test_sample.png"

@pytest.fixture
def sample_query():
    """Sample query for testing"""
    return "What is Python programming?"

@pytest.fixture
def test_model_name():
    """Test model name"""
    return "mistral"

@pytest.fixture
def mock_ollama_response():
    """Mock Ollama API response"""
    return "This is a mock response from the AI model."

@pytest.fixture
def mock_uploaded_file():
    """Mock uploaded file for testing"""
    file_mock = Mock()
    file_mock.name = "test_document.txt"
    file_mock.type = "text/plain"
    file_mock.getvalue.return_value = b"This is a test document content for testing purposes."
    return file_mock

@pytest.fixture
def mock_pdf_file():
    """Mock PDF file for testing"""
    file_mock = Mock()
    file_mock.name = "test_document.pdf"
    file_mock.type = "application/pdf"
    file_mock.getvalue.return_value = b"%PDF-1.4\nTest PDF content"
    return file_mock

@pytest.fixture
def mock_image_file():
    """Mock image file for testing"""
    # Create a simple test image
    img = Image.new('RGB', (100, 100), color='red')
    img_bytes = BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    file_mock = Mock()
    file_mock.name = "test_image.png"
    file_mock.type = "image/png"
    file_mock.getvalue.return_value = img_bytes.getvalue()
    return file_mock

@pytest.fixture
def mock_chroma_client():
    """Mock ChromaDB client"""
    client_mock = Mock()
    collection_mock = Mock()
    client_mock.get_or_create_collection.return_value = collection_mock
    client_mock.get_collection.return_value = collection_mock
    return client_mock

@pytest.fixture
def mock_llm():
    """Mock LLM for testing"""
    llm_mock = Mock()
    llm_mock.invoke.return_value.content = "Mock LLM response"
    return llm_mock

@pytest.fixture
def mock_embeddings():
    """Mock embeddings for testing"""
    embeddings_mock = Mock()
    embeddings_mock.embed_query.return_value = [0.1] * 384  # Mock embedding vector
    return embeddings_mock

# Cleanup after all tests
def pytest_sessionfinish(session, exitstatus):
    """Cleanup after test session"""
    # Clean up any remaining temporary files
    temp_dirs = [d for d in os.listdir(tempfile.gettempdir()) if d.startswith('pytest-')]
    for temp_dir in temp_dirs:
        try:
            shutil.rmtree(os.path.join(tempfile.gettempdir(), temp_dir), ignore_errors=True)
        except Exception:
            pass 