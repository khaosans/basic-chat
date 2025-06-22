"""
Reasoning engine functionality tests
CHANGELOG:
- Merged test_reasoning.py and test_enhanced_reasoning.py
- Removed redundant tool testing (moved to dedicated tools test)
- Focused on core reasoning logic and agent behavior
- Added parameterized tests for different reasoning modes
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from reasoning_engine import (
    ReasoningAgent, ReasoningChain, MultiStepReasoning, 
    ReasoningResult, ReasoningEngine
)

class TestReasoningResult:
    """Test reasoning result data structure"""
    
    def test_should_create_valid_reasoning_result(self):
        """Should create valid reasoning result with all fields"""
        result = ReasoningResult(
            content="Test answer",
            reasoning_steps=["Step 1", "Step 2"],
            thought_process="Test reasoning",
            final_answer="Test answer",
            confidence=0.8,
            sources=["source1"],
            reasoning_mode="chain_of_thought",
            success=True
        )
        
        assert result.content == "Test answer"
        assert len(result.reasoning_steps) == 2
        assert result.thought_process == "Test reasoning"
        assert result.final_answer == "Test answer"
        assert result.confidence == 0.8
        assert result.sources == ["source1"]
        assert result.reasoning_mode == "chain_of_thought"
        assert result.success is True
    
    def test_should_handle_missing_optional_fields(self):
        """Should handle missing optional fields gracefully"""
        result = ReasoningResult(
            content="Test answer",
            reasoning_steps=[],
            thought_process="",
            final_answer="Test answer",
            confidence=0.0,
            sources=[],
            reasoning_mode="standard"
        )
        
        assert result.content == "Test answer"
        assert result.reasoning_steps == []
        assert result.thought_process == ""
        assert result.final_answer == "Test answer"
        assert result.confidence == 0.0
        assert result.sources == []
        assert result.reasoning_mode == "standard"
        assert result.success is True  # Default value
        assert result.error is None    # Default value

class TestReasoningAgent:
    """Test reasoning agent functionality"""
    
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
        assert agent.model_name == "test_model"
        assert agent.agent is not None
    
    @patch('reasoning_engine.ChatOllama')
    @patch('reasoning_engine.initialize_agent')
    def test_should_reason_with_single_step(self, mock_initialize_agent, mock_chat_ollama):
        """Should perform single-step reasoning"""
        mock_llm = Mock()
        mock_chat_ollama.return_value = mock_llm
        mock_agent = Mock()
        mock_agent.run.return_value = {"output": "Test reasoning result"}
        mock_initialize_agent.return_value = mock_agent
        
        agent = ReasoningAgent("test_model")
        result = agent.run("What is 2+2?", "")
        
        assert result.content == "Test reasoning result"
        assert result.reasoning_steps != []
        assert result.confidence > 0
    
    @patch('reasoning_engine.ChatOllama')
    @patch('reasoning_engine.initialize_agent')
    def test_should_handle_reasoning_errors(self, mock_initialize_agent, mock_chat_ollama):
        """Should handle reasoning errors gracefully"""
        mock_llm = Mock()
        mock_chat_ollama.return_value = mock_llm
        mock_agent = Mock()
        mock_agent.run.side_effect = Exception("LLM error")
        mock_initialize_agent.return_value = mock_agent
        
        agent = ReasoningAgent("test_model")
        result = agent.run("Test question", "")
        
        assert "error" in result.content.lower()
        assert result.confidence == 0.0

class TestReasoningChain:
    """Test reasoning chain functionality"""
    
    @patch('reasoning_engine.ChatOllama')
    def test_should_execute_reasoning_chain(self, mock_chat_ollama):
        """Should execute multi-step reasoning chain"""
        mock_llm = Mock()
        mock_llm.invoke.return_value.content = "Step result"
        mock_chat_ollama.return_value = mock_llm
        
        chain = ReasoningChain("test_model")
        result = chain.execute("Complex question", steps=3)
        
        assert result.content == "Step result"
        assert len(result.steps) == 3
        assert result.confidence > 0
    
    @patch('reasoning_engine.ChatOllama')
    def test_should_handle_chain_errors(self, mock_chat_ollama):
        """Should handle chain execution errors"""
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("Chain error")
        mock_chat_ollama.return_value = mock_llm
        
        chain = ReasoningChain("test_model")
        result = chain.execute("Test question")
        
        assert "error" in result.content.lower()
        assert result.confidence == 0.0

class TestMultiStepReasoning:
    """Test multi-step reasoning functionality"""
    
    @patch('reasoning_engine.ChatOllama')
    def test_should_perform_multi_step_reasoning(self, mock_chat_ollama):
        """Should perform multi-step reasoning with intermediate steps"""
        mock_llm = Mock()
        mock_llm.invoke.return_value.content = "Intermediate step"
        mock_chat_ollama.return_value = mock_llm
        
        reasoning = MultiStepReasoning("test_model")
        result = reasoning.reason("Complex problem", max_steps=3)
        
        assert result.content == "Intermediate step"
        assert len(result.intermediate_steps) > 0
        assert result.final_answer != ""
    
    @patch('reasoning_engine.ChatOllama')
    def test_should_stop_at_max_steps(self, mock_chat_ollama):
        """Should stop reasoning at maximum steps"""
        mock_llm = Mock()
        mock_llm.invoke.return_value.content = "Step result"
        mock_chat_ollama.return_value = mock_llm
        
        reasoning = MultiStepReasoning("test_model")
        result = reasoning.reason("Test question", max_steps=2)
        
        assert len(result.intermediate_steps) <= 2

class TestReasoningEngine:
    """Test main reasoning engine functionality"""
    
    @patch('reasoning_engine.ChatOllama')
    def test_should_initialize_reasoning_engine(self, mock_chat_ollama):
        """Should initialize reasoning engine with all components"""
        mock_llm = Mock()
        mock_chat_ollama.return_value = mock_llm
        
        engine = ReasoningEngine("test_model")
        
        assert engine.agent is not None
        assert engine.chain is not None
        assert engine.multi_step is not None
        assert engine.model_name == "test_model"
    
    @patch('reasoning_engine.ChatOllama')
    def test_should_reason_with_different_modes(self, mock_chat_ollama):
        """Should reason using different reasoning modes"""
        mock_llm = Mock()
        mock_llm.invoke.return_value.content = "Reasoning result"
        mock_chat_ollama.return_value = mock_llm
        
        engine = ReasoningEngine("test_model")
        
        # Test different modes
        modes = ["agent", "chain", "multi_step", "auto"]
        for mode in modes:
            result = engine.reason("Test question", mode=mode)
            assert result.content == "Reasoning result"
            assert result.confidence > 0
    
    @patch('reasoning_engine.ChatOllama')
    def test_should_handle_invalid_reasoning_mode(self, mock_chat_ollama):
        """Should handle invalid reasoning mode gracefully"""
        mock_llm = Mock()
        mock_llm.invoke.return_value.content = "Default result"
        mock_chat_ollama.return_value = mock_llm
        
        engine = ReasoningEngine("test_model")
        result = engine.reason("Test question", mode="invalid_mode")
        
        # Should fallback to default mode
        assert result.content == "Default result"
    
    @patch('reasoning_engine.ChatOllama')
    def test_should_auto_select_best_mode(self, mock_chat_ollama):
        """Should automatically select best reasoning mode"""
        mock_llm = Mock()
        mock_llm.invoke.return_value.content = "Auto result"
        mock_chat_ollama.return_value = mock_llm
        
        engine = ReasoningEngine("test_model")
        result = engine.reason("Complex question requiring multi-step reasoning")
        
        # Auto mode should select appropriate reasoning strategy
        assert result.content == "Auto result"
        assert result.confidence > 0

class TestReasoningIntegration:
    """Test integration between reasoning components"""
    
    @patch('reasoning_engine.ChatOllama')
    def test_should_integrate_all_reasoning_components(self, mock_chat_ollama):
        """Should integrate all reasoning components seamlessly"""
        mock_llm = Mock()
        mock_llm.invoke.return_value.content = "Integrated result"
        mock_chat_ollama.return_value = mock_llm
        
        engine = ReasoningEngine("test_model")
        
        # Test that all components work together
        agent_result = engine.agent.reason("Agent question")
        chain_result = engine.chain.execute("Chain question")
        multi_result = engine.multi_step.reason("Multi-step question")
        
        assert agent_result.content == "Integrated result"
        assert chain_result.content == "Integrated result"
        assert multi_result.content == "Integrated result"
    
    @patch('reasoning_engine.ChatOllama')
    def test_should_maintain_consistency_across_modes(self, mock_chat_ollama):
        """Should maintain consistency across different reasoning modes"""
        mock_llm = Mock()
        mock_llm.invoke.return_value.content = "Consistent result"
        mock_chat_ollama.return_value = mock_llm
        
        engine = ReasoningEngine("test_model")
        
        # All modes should use the same underlying LLM
        result1 = engine.reason("Question 1", mode="agent")
        result2 = engine.reason("Question 2", mode="chain")
        result3 = engine.reason("Question 3", mode="multi_step")
        
        assert result1.content == result2.content == result3.content == "Consistent result"

class TestReasoningErrorHandling:
    """Test error handling in reasoning components"""
    
    @patch('reasoning_engine.ChatOllama')
    def test_should_handle_llm_connection_errors(self, mock_chat_ollama):
        """Should handle LLM connection errors gracefully"""
        mock_chat_ollama.side_effect = Exception("Connection failed")
        
        # Should handle initialization errors
        with pytest.raises(Exception):
            ReasoningEngine("test_model")
    
    @patch('reasoning_engine.ChatOllama')
    def test_should_handle_invalid_model_name(self, mock_chat_ollama):
        """Should handle invalid model names gracefully"""
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("Model not found")
        mock_chat_ollama.return_value = mock_llm
        
        engine = ReasoningEngine("invalid_model")
        result = engine.reason("Test question")
        
        assert "error" in result.content.lower()
        assert result.confidence == 0.0

@pytest.mark.parametrize("reasoning_class", [
    ReasoningChain,
    lambda m: MultiStepReasoning(m),
    ReasoningAgent
])
def test_should_handle_errors_gracefully(reasoning_class):
    """Should handle errors gracefully in all reasoning components"""
    if callable(reasoning_class):
        component = reasoning_class("invalid_model")
    else:
        component = reasoning_class("invalid_model")
    
    # Should not raise exception during initialization
    assert component is not None 