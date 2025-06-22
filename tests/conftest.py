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

# Add the project root to Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def sample_text():
    """Sample text for testing"""
    return """This is a sample text for testing document processing.
    It contains multiple lines and can be used for various test cases.
    The text includes information about artificial intelligence and machine learning."""

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
def temp_dir():
    """Temporary directory for test files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

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