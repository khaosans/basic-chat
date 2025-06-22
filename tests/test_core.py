"""
Core application functionality tests
CHANGELOG:
- Merged test_app.py and test_basic.py into single core test file
- Removed redundant initialization tests
- Focused on critical path testing
- Added parameterized tests for better coverage
"""

import pytest
from unittest.mock import patch, Mock
from app import OllamaChat, DocumentSummaryTool, ToolRegistry
from config import config

class TestOllamaChat:
    """Test core chat functionality"""
    
    def test_should_initialize_with_default_model(self):
        """Should initialize with default model when none provided"""
        chat = OllamaChat()
        assert chat.model_name == "mistral"
        assert chat.async_chat is not None
        assert chat.system_prompt is not None
    
    def test_should_initialize_with_custom_model(self):
        """Should initialize with custom model when provided"""
        chat = OllamaChat("llama2")
        assert chat.model_name == "llama2"
    
    @pytest.mark.parametrize("payload,expected_type", [
        ({"inputs": "Hello"}, str),
        ({"inputs": ""}, (str, type(None))),
        ({"inputs": None}, (str, type(None))),
    ])
    def test_should_handle_different_payload_types(self, payload, expected_type):
        """Should handle different payload types correctly"""
        chat = OllamaChat()
        result = chat.query(payload)
        assert isinstance(result, expected_type)
    
    @patch('app.asyncio.run')
    def test_should_use_async_implementation_by_default(self, mock_asyncio_run):
        """Should use async implementation by default"""
        chat = OllamaChat()
        mock_asyncio_run.return_value = "async response"
        
        result = chat.query({"inputs": "test"})
        
        assert result == "async response"
        mock_asyncio_run.assert_called_once()
    
    def test_should_fallback_to_sync_on_async_failure(self):
        """Should fallback to sync implementation when async fails"""
        chat = OllamaChat()
        chat._use_sync_fallback = True  # Force fallback
        
        with patch('app.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.iter_content.return_value = [b'{"response": "sync response"}']
            mock_post.return_value = mock_response
            
            result = chat.query({"inputs": "test"})
            
            assert result == "sync response"
            mock_post.assert_called_once()

class TestDocumentSummaryTool:
    """Test document summary tool functionality"""
    
    def test_should_return_summary_when_documents_exist(self):
        """Should return summary when documents are processed"""
        doc_processor_mock = Mock()
        doc_processor_mock.get_processed_files.return_value = [
            {"name": "doc1.pdf", "type": "application/pdf", "size": 1000},
            {"name": "doc2.txt", "type": "text/plain", "size": 500}
        ]
        
        tool = DocumentSummaryTool(doc_processor_mock)
        result = tool.execute("summarize document")
        
        assert result.success is True
        assert "doc1.pdf" in result.content
        assert "doc2.txt" in result.content
    
    def test_should_return_error_when_no_documents(self):
        """Should return error when no documents are processed"""
        doc_processor_mock = Mock()
        doc_processor_mock.get_processed_files.return_value = []
        
        tool = DocumentSummaryTool(doc_processor_mock)
        result = tool.execute("summarize document")
        
        assert result.success is False
        assert "No documents" in result.content

class TestToolRegistry:
    """Test tool registry functionality"""
    
    def test_should_return_tool_for_matching_trigger(self):
        """Should return appropriate tool for matching trigger"""
        doc_processor_mock = Mock()
        registry = ToolRegistry(doc_processor_mock)
        
        tool = registry.get_tool("summarize the document")
        
        assert tool is not None
        assert isinstance(tool, DocumentSummaryTool)
    
    def test_should_return_none_for_no_matching_trigger(self):
        """Should return None when no tool matches trigger"""
        doc_processor_mock = Mock()
        registry = ToolRegistry(doc_processor_mock)
        
        tool = registry.get_tool("random text that doesn't match")
        
        assert tool is None

class TestConfiguration:
    """Test configuration integration"""
    
    def test_should_have_required_config_attributes(self):
        """Should have all required configuration attributes"""
        assert hasattr(config, 'ollama_url')
        assert hasattr(config, 'ollama_model')
        assert hasattr(config, 'enable_caching')
        assert config.ollama_url is not None
        assert config.ollama_model is not None 