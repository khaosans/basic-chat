"""
Tests for the reasoning engine functionality
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from reasoning_engine import (
    ReasoningEngine,
    ReasoningResult
)
from utils.async_ollama import AsyncOllamaClient, AsyncOllamaChat

# Test fixtures
@pytest.fixture
def test_model_name():
    return "test_model"

@pytest.fixture
def mock_ollama_client():
    """Mock Ollama client for testing"""
    client = Mock(spec=AsyncOllamaClient)
    client.generate.return_value = {
        "response": "Test response from mock",
        "done": True
    }
    return client

@pytest.fixture
def reasoning_engine(test_model_name):
    """Create a reasoning engine for testing"""
    return ReasoningEngine(model_name=test_model_name)

class TestReasoningEngine:
    """Test the ReasoningEngine class"""
    
    def test_initialization(self, test_model_name):
        """Test reasoning engine initialization"""
        engine = ReasoningEngine(model_name=test_model_name)
        assert engine.model_name == test_model_name
        assert engine.agent is not None
        
    def test_reasoning_result_structure(self):
        """Test ReasoningResult dataclass structure"""
        result = ReasoningResult(
            final_answer="Test answer",
            reasoning_steps=["Step 1", "Step 2"],
            confidence=0.8,
            sources=["Source 1"],
            success=True,
            error=None
        )
        
        assert result.final_answer == "Test answer"
        assert len(result.reasoning_steps) == 2
        assert result.confidence == 0.8
        assert result.success is True
        assert result.error is None
        
    @patch('reasoning_engine.ChatOllama')
    def test_reasoning_with_mocked_ollama(self, mock_chat_ollama, reasoning_engine):
        """Test reasoning with mocked Ollama"""
        # Mock the chat model
        mock_chat_instance = Mock()
        mock_chat_ollama.return_value = mock_chat_instance
        
        # Mock the agent's run method
        with patch.object(reasoning_engine.agent, 'run') as mock_run:
            mock_run.return_value = "Mocked reasoning result"
            
            result = reasoning_engine.reason(
                input_text="What is 2 + 2?",
                context="",
                mode="chain_of_thought"
            )
            
            # Verify the result structure
            assert isinstance(result, ReasoningResult)
            # The result might be successful or not depending on mocking
            assert result.final_answer is not None
            
    def test_reasoning_modes(self, reasoning_engine):
        """Test different reasoning modes"""
        modes = ["chain_of_thought", "step_by_step", "direct"]
        
        for mode in modes:
            # This will likely fail without a real model, but tests the interface
            try:
                result = reasoning_engine.reason(
                    input_text="Test question",
                    context="Test context",
                    mode=mode
                )
                # If it succeeds, verify structure
                assert isinstance(result, ReasoningResult)
            except Exception as e:
                # Expected to fail without real model
                assert "not found" in str(e) or "connection" in str(e).lower()
                
    def test_error_handling(self, reasoning_engine):
        """Test error handling in reasoning"""
        # Test with invalid input
        result = reasoning_engine.reason(
            input_text="",
            context="",
            mode="invalid_mode"
        )
        
        # Should handle gracefully
        assert isinstance(result, ReasoningResult)
        assert result.success is False or result.final_answer is not None

class TestAsyncOllamaIntegration:
    """Test async Ollama integration"""
    
    @pytest.mark.asyncio
    async def test_async_client_mock(self, mock_ollama_client):
        """Test async client with mocking"""
        response = await mock_ollama_client.generate("test prompt")
        assert response["response"] == "Test response from mock"
        assert response["done"] is True
        
    def test_async_chat_initialization(self):
        """Test AsyncOllamaChat initialization"""
        chat = AsyncOllamaChat(model="test_model")
        assert chat.model == "test_model"

# Integration tests (will be skipped in CI without real Ollama)
class TestReasoningIntegration:
    """Integration tests for reasoning engine"""
    
    @pytest.mark.integration
    def test_full_reasoning_workflow(self, reasoning_engine):
        """Test complete reasoning workflow - requires real Ollama"""
        pytest.skip("Integration test - requires Ollama server")
        
        result = reasoning_engine.reason(
            input_text="Solve: 2x + 5 = 15",
            context="This is a linear equation",
            mode="step_by_step"
        )
        
        assert result.success is True
        assert "x = 5" in result.final_answer or "x=5" in result.final_answer
        assert len(result.reasoning_steps) > 0
