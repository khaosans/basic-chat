"""
Pytest configuration and fixtures for BasicChat tests.
Provides test isolation and environment management.
"""

import pytest
import os
import tempfile
import shutil
import asyncio
from unittest.mock import Mock, patch
from pathlib import Path

# Test environment configuration
@pytest.fixture(scope="session", autouse=True)
def test_environment():
    """Setup test environment variables and cleanup."""
    # Store original environment
    original_env = os.environ.copy()
    
    # Set test-specific environment variables
    os.environ.update({
        'TESTING': 'true',
        'CHROMA_PERSIST_DIR': './test_chroma_db',
        'OLLAMA_BASE_URL': 'http://localhost:11434',
        'MOCK_EXTERNAL_SERVICES': 'true',
    })
    
    # Create test directories
    test_dirs = ['./test_chroma_db', './tests/data']
    for dir_path in test_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    yield
    
    # Cleanup
    for dir_path in test_dirs:
        try:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
        except FileNotFoundError:
            pass
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)

@pytest.fixture(scope="function")
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture(scope="function")
def mock_external_services():
    """Mock external services for unit tests."""
    with patch('basicchat.services.document_processor.OllamaEmbeddings') as mock_embeddings, \
         patch('basicchat.services.document_processor.ChatOllama') as mock_chat, \
         patch('basicchat.services.document_processor.chromadb.PersistentClient') as mock_chroma, \
         patch('basicchat.core.app.gTTS') as mock_gtts:
        
        # Configure mocks
        mock_embeddings.return_value = Mock()
        mock_chat.return_value = Mock()
        mock_chroma.return_value = Mock()
        mock_gtts.return_value = Mock()
        
        yield {
            'embeddings': mock_embeddings,
            'chat': mock_chat,
            'chroma': mock_chroma,
            'gtts': mock_gtts
        }

@pytest.fixture(scope="function")
def mock_file_system():
    """Mock file system operations for isolated tests."""
    with patch('builtins.open') as mock_open, \
         patch('os.path.exists') as mock_exists, \
         patch('os.path.getsize') as mock_getsize:
        
        # Default behaviors
        mock_exists.return_value = True
        mock_getsize.return_value = 1024
        
        yield {
            'open': mock_open,
            'exists': mock_exists,
            'getsize': mock_getsize
        }

@pytest.fixture(scope="function")
def sample_test_files():
    """Create sample test files for testing."""
    files = {}
    
    # Create sample text file
    text_content = "This is a sample test document for testing purposes."
    files['text'] = text_content.encode('utf-8')
    
    # Create sample PDF content (minimal valid PDF)
    pdf_content = b'%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n/Contents 4 0 R\n>>\nendobj\n4 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 12 Tf\n72 720 Td\n(Test PDF) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000204 00000 n \ntrailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n297\n%%EOF'
    files['pdf'] = pdf_content
    
    # Create sample image content (minimal PNG)
    png_content = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\x0cIDATx\x9cc```\x00\x00\x00\x04\x00\x01\xf6\x178U\x00\x00\x00\x00IEND\xaeB`\x82'
    files['png'] = png_content
    
    yield files

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test (fast, isolated)"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test (requires external services)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow (LLM calls, heavy processing)"
    )
    config.addinivalue_line(
        "markers", "fast: mark test as fast (mocked, no external calls)"
    )
    config.addinivalue_line(
        "markers", "isolated: mark test as needing isolation"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end test"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add default markers."""
    for item in items:
        # Skip integration tests in default runs
        if 'integration' in item.nodeid:
            item.add_marker(pytest.mark.skip(reason="Integration test - run separately"))
            continue
            
        # Add default markers based on test location and name
        if 'test_core' in item.nodeid or 'test_tools' in item.nodeid:
            item.add_marker(pytest.mark.unit)
            item.add_marker(pytest.mark.fast)
        elif 'test_audio' in item.nodeid:
            item.add_marker(pytest.mark.unit)
        elif 'test_voice' in item.nodeid:
            item.add_marker(pytest.mark.unit)
        elif 'test_llm_judge' in item.nodeid:
            item.add_marker(pytest.mark.slow)
        elif 'test_reasoning' in item.nodeid:
            item.add_marker(pytest.mark.unit)  # Changed from integration to unit
        elif 'test_documents' in item.nodeid:
            item.add_marker(pytest.mark.unit)  # Changed from integration to unit
        elif 'test_web_search' in item.nodeid:
            item.add_marker(pytest.mark.unit)  # Changed from integration to unit
        elif 'test_upload' in item.nodeid or 'test_document_processing' in item.nodeid:
            item.add_marker(pytest.mark.unit)  # Changed from integration to unit 

@pytest.fixture(scope="function")
def mock_all_external_services():
    """Comprehensive mock for all external services in integration tests."""
    with patch('basicchat.services.document_processor.OllamaEmbeddings') as mock_embeddings, \
         patch('basicchat.services.document_processor.ChatOllama') as mock_chat, \
         patch('basicchat.services.document_processor.chromadb.PersistentClient') as mock_chroma, \
         patch('basicchat.core.app.gTTS') as mock_gtts, \
         patch('basicchat.services.web_search.DDGS') as mock_ddgs, \
         patch('openai.OpenAI') as mock_openai, \
         patch('langchain_ollama.OllamaEmbeddings') as mock_langchain_embeddings, \
         patch('langchain_ollama.ChatOllama') as mock_langchain_chat:
        
        mock_embeddings.return_value = Mock()
        mock_chat.return_value = Mock()
        mock_chroma.return_value = Mock()
        mock_gtts.return_value = Mock()
        
        mock_ddgs_instance = Mock()
        mock_ddgs_instance.text.return_value = [
            {'title': 'Test Result', 'link': 'https://test.com', 'body': 'Test content'}
        ]
        mock_ddgs.return_value = mock_ddgs_instance
        
        mock_openai_instance = Mock()
        mock_openai_instance.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="Mocked AI response"))]
        )
        mock_openai.return_value = mock_openai_instance
        
        mock_langchain_embeddings.return_value = Mock()
        mock_langchain_chat.return_value = Mock()
        
        yield {
            'embeddings': mock_embeddings,
            'chat': mock_chat,
            'chroma': mock_chroma,
            'gtts': mock_gtts,
            'ddgs': mock_ddgs,
            'openai': mock_openai,
            'langchain_embeddings': mock_langchain_embeddings,
            'langchain_chat': mock_langchain_chat
        } 
