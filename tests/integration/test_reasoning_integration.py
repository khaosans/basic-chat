"""
Integration tests for reasoning engine functionality
Tests that require external services or complex interactions
"""

import pytest
pytest.skip("Integration tests require external dependencies", allow_module_level=True)
from unittest.mock import Mock, patch, MagicMock
from reasoning_engine import (
    ReasoningAgent, ReasoningChain, MultiStepReasoning, 
    ReasoningResult, ReasoningEngine
)

@pytest.mark.integration
class TestReasoningIntegration:
    """Integration tests for reasoning components"""
    
    @patch('reasoning_engine.ReasoningAgent')
    @patch('reasoning_engine.ReasoningChain')
    def test_should_integrate_all_reasoning_components(self, mock_chain_class, mock_agent_class):
        """Should integrate all reasoning components seamlessly"""
        mock_agent = Mock()
        mock_agent.run.return_value = ReasoningResult(
            content="Integrated result",
            reasoning_steps=[],
            thought_process="",
            final_answer="Integrated result",
            confidence=0.8,
            sources=[],
            reasoning_mode="Agent"
        )
        mock_agent_class.return_value = mock_agent
        
        mock_chain = Mock()
        mock_chain.execute_reasoning.return_value = ReasoningResult(
            content="Integrated result",
            reasoning_steps=[],
            thought_process="",
            final_answer="Integrated result",
            confidence=0.8,
            sources=[],
            reasoning_mode="Chain-of-Thought"
        )
        mock_chain_class.return_value = mock_chain
        
        engine = ReasoningEngine("test_model")
        
        # Test that all components work together
        agent_result = engine.run("Agent question", mode="Agent")
        chain_result = engine.run("Chain question", mode="Chain-of-Thought")
        
        assert agent_result.content == "Integrated result"
        assert chain_result.content == "Integrated result"

@pytest.mark.integration
class TestReasoningErrorHandling:
    """Integration tests for error handling in reasoning components"""
    
    @patch('reasoning_engine.ChatOllama')
    def test_should_handle_llm_connection_errors(self, mock_chat_ollama):
        """Should handle LLM connection errors gracefully"""
        mock_chat_ollama.side_effect = Exception("Connection failed")
        
        # Should handle initialization errors
        with pytest.raises(Exception):
            ReasoningAgent("test_model")
    
    @patch('reasoning_engine.ReasoningAgent')
    def test_should_handle_invalid_model_name(self, mock_agent_class):
        """Should handle invalid model names gracefully"""
        mock_agent = Mock()
        mock_agent.run.side_effect = Exception("Model not found")
        mock_agent_class.return_value = mock_agent
        
        engine = ReasoningEngine("invalid_model")
        
        with pytest.raises(Exception):
            engine.run("Test question", mode="Agent")

@pytest.mark.integration
class TestReasoningAgentIntegration:
    """Integration tests for ReasoningAgent"""
    
    @patch('reasoning_engine.ChatOllama')
    @patch('reasoning_engine.initialize_agent')
    def test_should_initialize_agent_with_llm(self, mock_initialize_agent, mock_chat_ollama):
        """Should initialize agent with LLM"""
        mock_llm = Mock()
        mock_chat_ollama.return_value = mock_llm
        mock_agent = Mock()
        mock_initialize_agent.return_value = mock_agent
        
        agent = ReasoningAgent("test_model")
        
        assert agent.llm is not None
        assert agent.agent is not None
    
    @patch('reasoning_engine.ChatOllama')
    @patch('reasoning_engine.initialize_agent')
    def test_should_reason_with_single_step(self, mock_initialize_agent, mock_chat_ollama):
        """Should perform single-step reasoning"""
        mock_llm = Mock()
        mock_chat_ollama.return_value = mock_llm
        # Use a dict for the response
        mock_response = {"output": "Test reasoning result"}
        class MockAgent:
            def run(self, *args, **kwargs):
                return mock_response
            def invoke(self, *args, **kwargs):
                return mock_response
        mock_initialize_agent.return_value = MockAgent()
        agent = ReasoningAgent("test_model")
        result = agent.run("What is 2+2?", "", stream_callback=None)
        assert result.content == "Test reasoning result"
        assert result.reasoning_steps != []
        assert result.confidence > 0

@pytest.mark.integration
class TestReasoningChainIntegration:
    """Integration tests for ReasoningChain"""
    
    @patch('reasoning_engine.ChatOllama')
    def test_should_execute_reasoning_chain(self, mock_chat_ollama):
        """Should execute multi-step reasoning chain"""
        mock_llm = Mock()
        # Return a string directly for .invoke()
        mock_llm.invoke.return_value = "Step result"
        mock_chat_ollama.return_value = mock_llm
        
        chain = ReasoningChain("test_model")
        result = chain.execute_reasoning("Complex question", "")
        # The code extracts 'result' from 'Step result'
        assert result.content == "result"
        assert len(result.reasoning_steps) > 0
        assert result.confidence > 0

@pytest.mark.integration
class TestMultiStepReasoningIntegration:
    """Integration tests for MultiStepReasoning"""
    
    @patch('reasoning_engine.ChatOllama')
    def test_should_perform_multi_step_reasoning(self, mock_chat_ollama):
        """Should perform multi-step reasoning with intermediate steps"""
        mock_llm = Mock()
        mock_llm.invoke.return_value.content = "Intermediate step"
        mock_chat_ollama.return_value = mock_llm
        
        reasoning = MultiStepReasoning("test_model")
        result = reasoning.step_by_step_reasoning("Complex problem", "")
        
        assert result.content == "Intermediate step"
        assert len(result.reasoning_steps) > 0
        assert result.final_answer != ""

@pytest.mark.integration
class TestReasoningEngineIntegration:
    """Integration tests for ReasoningEngine"""
    
    def test_should_initialize_reasoning_engine(self):
        """Should initialize reasoning engine with all components"""
        engine = ReasoningEngine("test_model")
        
        assert engine.model_name == "test_model"
        assert engine.agent_reasoner is None  # Lazy initialization
        assert engine.chain_reasoner is None
        assert engine.multi_step_reasoner is None
        assert engine.standard_reasoner is None
    
    @patch('reasoning_engine.ReasoningAgent')
    def test_should_reason_with_agent_mode(self, mock_agent_class):
        """Should reason using agent mode"""
        mock_agent = Mock()
        mock_agent.run.return_value = ReasoningResult(
            content="Agent result",
            reasoning_steps=[],
            thought_process="",
            final_answer="Agent result",
            confidence=0.8,
            sources=[],
            reasoning_mode="Agent"
        )
        mock_agent_class.return_value = mock_agent
        
        engine = ReasoningEngine("test_model")
        result = engine.run("Test question", mode="Agent")
        
        assert result.content == "Agent result"
        assert result.confidence > 0
    
    @patch('reasoning_engine.ReasoningChain')
    def test_should_reason_with_chain_mode(self, mock_chain_class):
        """Should reason using chain-of-thought mode"""
        mock_chain = Mock()
        mock_chain.execute_reasoning.return_value = ReasoningResult(
            content="Chain result",
            reasoning_steps=[],
            thought_process="",
            final_answer="Chain result",
            confidence=0.8,
            sources=[],
            reasoning_mode="Chain-of-Thought"
        )
        mock_chain_class.return_value = mock_chain
        
        engine = ReasoningEngine("test_model")
        result = engine.run("Test question", mode="Chain-of-Thought")
        
        assert result.content == "Chain result"
        assert result.confidence > 0 
