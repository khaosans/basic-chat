#!/usr/bin/env python3
"""
Core functionality tests for BasicChat application.

These tests verify the basic functionality of the application components.
"""

import pytest
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile

# Add the parent directory to the path so we can import from app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import OllamaChat, text_to_speech, get_professional_audio_html, get_audio_file_size
from reasoning_engine import ReasoningEngine
from document_processor import DocumentProcessor
from utils.enhanced_tools import EnhancedCalculator
from config import config


@pytest.mark.unit
@pytest.mark.fast
class TestCoreFunctionality:
    """Test core application functionality"""

    def test_ollama_chat_initialization(self):
        """Test OllamaChat initialization"""
        chat = OllamaChat()
        assert chat is not None
        assert hasattr(chat, 'query')

    def test_reasoning_engine_initialization(self):
        """Test ReasoningEngine initialization"""
        engine = ReasoningEngine()
        assert engine is not None
        assert hasattr(engine, 'process_query')

    def test_document_processor_initialization(self):
        """Test DocumentProcessor initialization"""
        processor = DocumentProcessor()
        assert processor is not None
        assert hasattr(processor, 'process_file')

    def test_enhanced_calculator_initialization(self):
        """Test EnhancedCalculator initialization"""
        calc = EnhancedCalculator()
        assert calc is not None
        assert hasattr(calc, 'calculate')

    def test_config_loading(self):
        """Test configuration loading"""
        assert config is not None
        assert hasattr(config, 'ollama_model')
        assert hasattr(config, 'ollama_url')


@pytest.mark.unit
@pytest.mark.fast
class TestOllamaChat:
    """Test OllamaChat functionality"""

    @patch('utils.async_ollama.AsyncOllamaChat.query')
    def test_query_method(self, mock_async_query):
        """Test OllamaChat query method"""
        # Mock async response
        mock_async_query.return_value = "Test response from Ollama"

        chat = OllamaChat()
        result = chat.query({'inputs': 'Hello, world!'})
        
        assert result == "Test response from Ollama"
        mock_async_query.assert_called_once()

    @patch('app.requests.post')
    def test_query_with_error_handling(self, mock_post):
        """Test OllamaChat error handling"""
        # Mock error response
        mock_post.side_effect = Exception("Connection error")

        chat = OllamaChat()
        result = chat.query({'inputs': 'Hello, world!'})
        
        # Should handle error gracefully
        assert result is None or isinstance(result, str)


@pytest.mark.unit
@pytest.mark.fast
class TestReasoningEngine:
    """Test ReasoningEngine functionality"""

    def test_reasoning_modes(self):
        """Test available reasoning modes"""
        engine = ReasoningEngine()
        
        # Check that reasoning modes are available
        assert hasattr(engine, 'reasoning_modes')
        assert isinstance(engine.reasoning_modes, list)
        assert len(engine.reasoning_modes) > 0

    @patch('app.OllamaChat')
    def test_process_query(self, mock_ollama):
        """Test process_query method"""
        # Mock OllamaChat
        mock_chat = Mock()
        mock_chat.query.return_value = "Test response"
        mock_ollama.return_value = mock_chat

        engine = ReasoningEngine()
        result = engine.process_query("Test query", mode="Standard")
        assert result is not None


@pytest.mark.unit
@pytest.mark.fast
class TestDocumentProcessor:
    """Test DocumentProcessor functionality"""

    def test_processor_initialization(self):
        """Test DocumentProcessor initialization"""
        processor = DocumentProcessor()
        
        # Check that required attributes exist
        assert hasattr(processor, 'client')
        assert hasattr(processor, 'embeddings')
        assert hasattr(processor, 'text_splitter')

    def test_get_processed_files(self):
        """Test get_processed_files method"""
        processor = DocumentProcessor()
        files = processor.get_processed_files()
        
        # Should return a list
        assert isinstance(files, list)

    def test_get_available_documents(self):
        """Test get_available_documents method"""
        processor = DocumentProcessor()
        documents = processor.get_available_documents()
        
        # Should return a list
        assert isinstance(documents, list)


@pytest.mark.unit
@pytest.mark.fast
class TestEnhancedCalculator:
    """Test EnhancedCalculator functionality"""

    def test_basic_calculation(self):
        """Test basic mathematical operations"""
        calc = EnhancedCalculator()
        
        # Test basic arithmetic
        result = calc.calculate("2 + 2")
        assert result.success
        assert str(result.result) == "4" or str(result.result) == "4.0"

    def test_complex_calculation(self):
        """Test complex mathematical operations"""
        calc = EnhancedCalculator()
        
        # Test more complex expression
        result = calc.calculate("(2 + 3) * 4")
        assert result.success
        assert str(result.result) == "20" or str(result.result) == "20.0"

    def test_invalid_expression(self):
        """Test handling of invalid expressions"""
        calc = EnhancedCalculator()
        
        # Test invalid expression
        result = calc.calculate("invalid expression")
        assert not result.success
        assert result.error is not None

    def test_safe_expression_validation(self):
        """Test expression safety validation"""
        calc = EnhancedCalculator()
        
        # Test safe expression
        assert calc._is_safe_expression("2 + 2")
        
        # Test unsafe expression
        assert not calc._is_safe_expression("__import__('os')")


@pytest.mark.unit
@pytest.mark.fast
class TestConfiguration:
    """Test configuration system"""

    def test_config_attributes(self):
        """Test that config has required attributes"""
        assert hasattr(config, 'ollama_model')
        assert hasattr(config, 'ollama_url')
        assert hasattr(config, 'enable_caching')
        
        # Check that values are not None
        assert config.ollama_model is not None
        assert config.ollama_url is not None

    def test_config_types(self):
        """Test configuration value types"""
        assert isinstance(config.ollama_model, str)
        assert isinstance(config.ollama_url, str)
        assert isinstance(config.enable_caching, bool)


@pytest.mark.unit
@pytest.mark.fast
class TestAudioUtils:
    def test_text_to_speech_valid(self):
        text = "Hello, test audio!"
        audio_file = text_to_speech(text)
        assert audio_file is not None
        assert audio_file.endswith('.mp3')
        assert os.path.exists(audio_file)
        os.remove(audio_file)

    def test_text_to_speech_empty(self):
        assert text_to_speech("") is None
        assert text_to_speech(None) is None

    def test_get_professional_audio_html_valid(self):
        text = "Audio HTML test"
        audio_file = text_to_speech(text)
        html = get_professional_audio_html(audio_file)
        assert '<audio' in html
        assert 'controls' in html
        os.remove(audio_file)

    def test_get_professional_audio_html_missing(self):
        html = get_professional_audio_html("nonexistent_file.mp3")
        assert "Audio file not found" in html

    def test_get_audio_file_size(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test data")
            file_path = f.name
        try:
            size = get_audio_file_size(file_path)
            assert "B" in size
        finally:
            os.unlink(file_path)
        assert get_audio_file_size("nonexistent_file.mp3") == "Unknown size"


if __name__ == "__main__":
    pytest.main([__file__]) 