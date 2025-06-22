"""
Reasoning engine functionality tests
CHANGELOG:
- Merged test_reasoning.py and test_enhanced_reasoning.py
- Removed redundant tool testing (moved to dedicated tools test)
- Focused on core reasoning logic and agent behavior
- Added parameterized tests for different reasoning modes
"""

import pytest
from unittest.mock import Mock, patch
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
            thought_process="Detailed reasoning",
            final_answer="Final answer",
            confidence=0.9,
            sources=["source1"],
            reasoning_mode="chain_of_thought",
            success=True
        )
        
        assert result.content == "Test answer"
        assert len(result.reasoning_steps) == 2
        assert result.confidence == 0.9
        assert result.success is True

class TestReasoningAgent:
    """Test reasoning agent functionality"""
    
    @patch('reasoning_engine.ChatOllama')
    def test_should_initialize_with_model(self, mock_chat_ollama):
        """Should initialize agent with specified model"""
        mock_llm = Mock()
        mock_chat_ollama.return_value = mock_llm
        
        agent = ReasoningAgent("mistral")
        
        assert agent.llm == mock_llm
        assert len(agent.tools) > 0
        assert agent.agent is not None
    
    @patch('reasoning_engine.ChatOllama')
    def test_should_have_enhanced_tools(self, mock_chat_ollama):
        """Should have enhanced calculator and time tools"""
        mock_llm = Mock()
        mock_chat_ollama.return_value = mock_llm
        
        agent = ReasoningAgent("mistral")
        
        tool_names = [tool.name for tool in agent.tools]
        assert "enhanced_calculator" in tool_names
        assert "get_current_time" in tool_names
        assert "web_search" in tool_names
    
    @patch('reasoning_engine.ChatOllama')
    def test_should_handle_calculation_requests(self, mock_chat_ollama):
        """Should handle mathematical calculation requests"""
        mock_llm = Mock()
        mock_chat_ollama.return_value = mock_llm
        
        agent = ReasoningAgent("mistral")
        
        result = agent._enhanced_calculate("2 + 2")
        
        assert "✅ Calculation Result: 4" in result
        assert "📝 Expression: 2 + 2" in result
    
    @patch('reasoning_engine.ChatOllama')
    def test_should_handle_time_requests(self, mock_chat_ollama):
        """Should handle time-related requests"""
        mock_llm = Mock()
        mock_chat_ollama.return_value = mock_llm
        
        agent = ReasoningAgent("mistral")
        
        result = agent._get_current_time("UTC")
        
        assert "🕐 Current Time:" in result
        assert "🌍 Timezone: UTC" in result

class TestReasoningChain:
    """Test chain-of-thought reasoning"""
    
    @patch('reasoning_engine.ChatOllama')
    def test_should_execute_chain_of_thought(self, mock_chat_ollama):
        """Should execute chain-of-thought reasoning"""
        mock_llm = Mock()
        mock_llm.invoke.return_value.content = "Let me think step by step...\n\nFinal answer: 42"
        mock_chat_ollama.return_value = mock_llm
        
        chain = ReasoningChain("mistral")
        result = chain.execute_reasoning("What is 6 * 7?", "")
        
        assert isinstance(result, ReasoningResult)
        assert result.success is True
        assert "42" in result.content

class TestMultiStepReasoning:
    """Test multi-step reasoning"""
    
    @patch('reasoning_engine.ChatOllama')
    def test_should_break_down_complex_queries(self, mock_chat_ollama):
        """Should break down complex queries into sub-questions"""
        mock_llm = Mock()
        mock_llm.invoke.return_value.content = "Sub-question 1: What is X?\nSub-question 2: What is Y?"
        mock_chat_ollama.return_value = mock_llm
        
        multi_step = MultiStepReasoning("mistral")
        
        with patch.object(multi_step, '_answer_sub_query', return_value="Answer"):
            with patch.object(multi_step, '_synthesize_final_answer', return_value="Final synthesis"):
                result = multi_step.step_by_step_reasoning("Complex question", "")
                
                assert isinstance(result, ReasoningResult)
                assert result.success is True

class TestReasoningEngine:
    """Test main reasoning engine"""
    
    @patch('reasoning_engine.ChatOllama')
    def test_should_run_with_different_modes(self, mock_chat_ollama):
        """Should run reasoning with different modes"""
        mock_llm = Mock()
        mock_chat_ollama.return_value = mock_llm
        
        engine = ReasoningEngine("mistral")
        
        modes = ["Chain", "MultiStep", "Agent", "Auto"]
        
        for mode in modes:
            with patch.object(engine, f'{mode.lower()}_reasoning') as mock_reasoning:
                mock_reasoning.return_value = ReasoningResult(
                    content="Test response",
                    reasoning_steps=[],
                    thought_process="",
                    final_answer="Test response",
                    confidence=0.8,
                    sources=[],
                    reasoning_mode=mode,
                    success=True
                )
                
                result = engine.run("Test query", mode)
                
                assert result.success is True
                assert result.reasoning_mode == mode

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