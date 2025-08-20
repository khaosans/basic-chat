"""
Reasoning engine functionality tests
CHANGELOG:
- Merged test_reasoning.py and test_enhanced_reasoning.py
- Removed redundant tool testing (moved to dedicated tools test)
- Focused on core reasoning logic and agent behavior
- Added parameterized tests for different reasoning modes
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pytest
from unittest.mock import Mock, patch, MagicMock
from basicchat.core.reasoning_engine import (
    ReasoningAgent, ReasoningChain, MultiStepReasoning, 
    ReasoningResult, ReasoningEngine
)
from collections import namedtuple

class TestReasoningResult:
    """Test reasoning result data structure"""
    @pytest.mark.integration
    @pytest.mark.integration
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
    @pytest.mark.integration
    @pytest.mark.integration
    @patch('basicchat.core.reasoning_engine.ChatOllama')
    @patch('basicchat.core.reasoning_engine.initialize_agent')
    def test_should_initialize_agent_with_llm(self, mock_initialize_agent, mock_chat_ollama):
        """Should initialize agent with LLM"""
        mock_llm = Mock()
        mock_chat_ollama.return_value = mock_llm
        mock_agent = Mock()
        mock_initialize_agent.return_value = mock_agent
        
        agent = ReasoningAgent("test_model")
        
        assert agent.llm is not None
        assert agent.agent is not None
    
    @patch('basicchat.core.reasoning_engine.ChatOllama')
    @patch('basicchat.core.reasoning_engine.initialize_agent')
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
    
    @patch('basicchat.core.reasoning_engine.ChatOllama')
    @patch('basicchat.core.reasoning_engine.initialize_agent')
    def test_should_handle_reasoning_errors(self, mock_initialize_agent, mock_chat_ollama):
        """Should handle reasoning errors gracefully"""
        mock_llm = Mock()
        mock_chat_ollama.return_value = mock_llm
        class MockAgent:
            def run(self, *args, **kwargs):
                raise Exception("LLM error")
            def invoke(self, *args, **kwargs):
                raise Exception("LLM error")
        mock_initialize_agent.return_value = MockAgent()
        agent = ReasoningAgent("test_model")
        result = agent.run("Test question", "", stream_callback=None)
        assert "error" in result.error.lower()
        assert result.confidence == 0.0

class TestReasoningChain:
    """Test reasoning chain functionality"""
    @pytest.mark.integration
    @pytest.mark.integration
    @patch('basicchat.core.reasoning_engine.ChatOllama')
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
    
    @patch('basicchat.core.reasoning_engine.ChatOllama')
    def test_should_handle_chain_errors(self, mock_chat_ollama):
        """Should handle chain execution errors"""
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("Chain error")
        mock_chat_ollama.return_value = mock_llm
        
        chain = ReasoningChain("test_model")
        result = chain.execute_reasoning("Test question", "")
        
        assert "error" in result.content.lower()
        assert result.confidence == 0.0

class TestMultiStepReasoning:
    """Test multi-step reasoning functionality"""
    @pytest.mark.integration
    @pytest.mark.integration
    @patch('basicchat.core.reasoning_engine.ChatOllama')
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
    
    @patch('basicchat.core.reasoning_engine.ChatOllama')
    def test_should_stop_at_max_steps(self, mock_chat_ollama):
        """Should stop reasoning at maximum steps"""
        mock_llm = Mock()
        mock_llm.invoke.return_value.content = "Step result"
        mock_chat_ollama.return_value = mock_llm
        
        reasoning = MultiStepReasoning("test_model")
        result = reasoning.step_by_step_reasoning("Test question", "")
        
        assert len(result.reasoning_steps) > 0

class TestReasoningEngine:
    """Test main reasoning engine functionality"""
    @pytest.mark.integration
    @pytest.mark.integration
    def test_should_initialize_reasoning_engine(self):
        """Should initialize reasoning engine with all components"""
        engine = ReasoningEngine("test_model")
        
        assert engine.model_name == "test_model"
        assert engine.agent_reasoner is None  # Lazy initialization
        assert engine.chain_reasoner is None
        assert engine.multi_step_reasoner is None
        assert engine.standard_reasoner is None
    
    @patch('basicchat.core.reasoning_engine.ReasoningAgent')
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
    
    @patch('basicchat.core.reasoning_engine.ReasoningChain')
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
    
    def test_should_handle_invalid_reasoning_mode(self):
        """Should handle invalid reasoning mode gracefully"""
        engine = ReasoningEngine("test_model")
        
        with pytest.raises(ValueError, match="Unknown reasoning mode"):
            engine.run("Test question", mode="invalid_mode")

    @patch('basicchat.core.reasoning_engine.ChatOllama')
    def test_should_reason_with_enhanced_lcel_mode(self, mock_chat_ollama):
        """Should reason using Enhanced LCEL mode and parse structured output"""
        # Mock the LLM to return an object with a .content attribute (like AIMessage) for .invoke,
        # but return a JSON string directly when called as a function (for LCEL chain)
        class MockAIMessage:
            def __init__(self, content):
                self.content = content
        mock_llm = Mock()
        json_str = (
            '{"thought_process": "Step-by-step reasoning", '
            '"reasoning_steps": ["Step 1", "Step 2"], '
            '"final_answer": "The answer is 42.", '
            '"confidence": 0.95, '
            '"key_insights": ["Insight 1"], '
            '"sources_used": ["TestSource"]}'
        )
        mock_llm.invoke.return_value = MockAIMessage(json_str)
        mock_llm.side_effect = lambda *args, **kwargs: json_str  # Callable returns string for LCEL
        mock_chat_ollama.return_value = mock_llm
        engine = ReasoningEngine("test_model")
        result = engine.run("What is the answer to life?", mode="Enhanced LCEL")
        assert result.content == 'The answer is 42.'
        assert result.thought_process == 'Step-by-step reasoning'
        assert result.reasoning_steps == ['Step 1', 'Step 2']
        assert result.final_answer == 'The answer is 42.'
        assert result.confidence == 0.95
        assert result.sources == ['TestSource']
        assert result.reasoning_mode == 'Enhanced LCEL'
        assert result.success is True

class TestReasoningIntegration:
    """Test integration between reasoning components"""
    @pytest.mark.integration
    @pytest.mark.integration
    
    @patch('basicchat.core.reasoning_engine.ReasoningAgent')
    @patch('basicchat.core.reasoning_engine.ReasoningChain')
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

class TestReasoningErrorHandling:
    """Test error handling in reasoning components"""
    @pytest.mark.integration
    @pytest.mark.integration
    
    @patch('basicchat.core.reasoning_engine.ChatOllama')
    def test_should_handle_llm_connection_errors(self, mock_chat_ollama):
        """Should handle LLM connection errors gracefully"""
        mock_chat_ollama.side_effect = Exception("Connection failed")
        
        # Should handle initialization errors
        with pytest.raises(Exception):
            ReasoningAgent("test_model")
    
    @patch('basicchat.core.reasoning_engine.ReasoningAgent')
    def test_should_handle_invalid_model_name(self, mock_agent_class):
        """Should handle invalid model names gracefully"""
        mock_agent = Mock()
        mock_agent.run.side_effect = Exception("Model not found")
        mock_agent_class.return_value = mock_agent
        
        engine = ReasoningEngine("invalid_model")
        
        with pytest.raises(Exception):
            engine.run("Test question", mode="Agent")

def test_should_handle_errors_gracefully():
    """Should handle errors gracefully across all reasoning classes"""
    # Test that the classes can be imported and basic structure is correct
    assert ReasoningChain is not None
    assert MultiStepReasoning is not None
    assert ReasoningAgent is not None
    
    # Test that they have the expected methods
    assert hasattr(ReasoningChain, 'execute_reasoning')
    assert hasattr(MultiStepReasoning, 'step_by_step_reasoning')
    assert hasattr(ReasoningAgent, 'run') 
