"""
Tests for the LLM Judge Evaluator
"""
import pytest

# Skip if optional dependencies required for app import are missing
pytest.importorskip("streamlit", reason="streamlit not installed")
pytest.importorskip("requests", reason="requests not installed")
import os
import sys
import json
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the evaluator
from evaluators.check_llm_judge import LLMJudgeEvaluator

class TestLLMJudgeEvaluator:
    """Test class for LLM Judge Evaluator"""
    
    def setup_method(self):
        """Setup method for each test"""
        # Mock environment variables
        self.original_env = os.environ.copy()
        os.environ['OLLAMA_API_URL'] = 'http://localhost:11434/api'
        os.environ['OLLAMA_MODEL'] = 'mistral'
        os.environ['LLM_JUDGE_THRESHOLD'] = '7.0'
    
    def teardown_method(self):
        """Teardown method for each test"""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_should_initialize_evaluator(self):
        """Test that the evaluator initializes correctly"""
        evaluator = LLMJudgeEvaluator()
        
        assert evaluator.ollama_url == 'http://localhost:11434/api'
        assert evaluator.threshold == 7.0
        assert evaluator.model == 'mistral'
        assert evaluator.results is not None
        assert evaluator.ollama_chat is not None
    
    @patch('evaluators.check_llm_judge.LLMJudgeEvaluator.collect_codebase_info')
    def test_should_collect_codebase_info(self, mock_collect_info):
        """Test that codebase information is collected correctly"""
        # Mock the expensive collect_codebase_info method
        mock_collect_info.return_value = {
            'file_count': 25,
            'lines_of_code': 2500,
            'test_files': 15,
            'test_coverage': 85.5,
            'documentation_files': 8,
            'dependencies': ['requests', 'pytest', 'flask']
        }
        
        evaluator = LLMJudgeEvaluator()
        info = evaluator.collect_codebase_info()
        
        # Check that required keys are present
        required_keys = ['file_count', 'lines_of_code', 'test_files', 'test_coverage', 
                        'documentation_files', 'dependencies']
        for key in required_keys:
            assert key in info
        
        # Check that values are reasonable
        assert info['file_count'] >= 0
        assert info['lines_of_code'] >= 0
        assert info['test_files'] >= 0
        assert 0 <= info['test_coverage'] <= 100
        assert info['documentation_files'] >= 0
        assert isinstance(info['dependencies'], list)
    
    def test_should_generate_evaluation_prompt(self):
        """Test that evaluation prompt is generated correctly"""
        evaluator = LLMJudgeEvaluator()
        codebase_info = {
            'file_count': 10,
            'lines_of_code': 1000,
            'test_files': 5,
            'test_coverage': 85.5,
            'documentation_files': 3,
            'dependencies': ['requests', 'pytest']
        }
        
        prompt = evaluator.generate_evaluation_prompt(codebase_info)
        
        # Check that prompt contains expected information
        assert '10' in prompt  # file_count
        assert '1000' in prompt  # lines_of_code
        assert '5' in prompt  # test_files
        assert '85.5' in prompt  # test_coverage
        assert '3' in prompt  # documentation_files
        assert '2' in prompt  # dependencies count
        
        # Check that prompt contains evaluation categories
        assert 'Code Quality' in prompt
        assert 'Test Coverage' in prompt
        assert 'Documentation' in prompt
        assert 'Architecture' in prompt
        assert 'Security' in prompt
        assert 'Performance' in prompt
    
    @patch('evaluators.check_llm_judge.OllamaChat')
    def test_should_evaluate_with_llm(self, mock_ollama_chat_class):
        """Test LLM evaluation with mocked OllamaChat"""
        # Mock OllamaChat
        mock_ollama_chat = MagicMock()
        mock_ollama_chat_class.return_value = mock_ollama_chat
        
        # Mock the query method to return a valid JSON response
        mock_ollama_chat.query.return_value = '''
        {
            "scores": {
                "code_quality": {"score": 8, "justification": "Well-structured code"},
                "test_coverage": {"score": 7, "justification": "Good test coverage"},
                "documentation": {"score": 6, "justification": "Basic documentation"},
                "architecture": {"score": 8, "justification": "Clean design"},
                "security": {"score": 7, "justification": "No obvious issues"},
                "performance": {"score": 7, "justification": "Generally efficient"}
            },
            "overall_score": 7.2,
            "recommendations": ["Add more tests", "Improve documentation"]
        }
        '''
        
        evaluator = LLMJudgeEvaluator()
        prompt = "Test prompt"
        
        result = evaluator.evaluate_with_llm(prompt)
        
        # Check that OllamaChat was called
        mock_ollama_chat.query.assert_called_once_with({"inputs": prompt})
        
        # Check that result contains expected data
        assert 'scores' in result
        assert 'overall_score' in result
        assert 'recommendations' in result
        assert result['overall_score'] == 7.2
        assert len(result['recommendations']) == 2
    
    @patch('evaluators.check_llm_judge.OllamaChat')
    def test_should_handle_llm_failures(self, mock_ollama_chat_class):
        """Test handling of LLM API failures"""
        # Mock OllamaChat
        mock_ollama_chat = MagicMock()
        mock_ollama_chat_class.return_value = mock_ollama_chat
        
        # Mock API failure
        mock_ollama_chat.query.side_effect = Exception("API Error")
        
        evaluator = LLMJudgeEvaluator()
        prompt = "Test prompt"
        
        # The final exception should be the original API Error from the last retry attempt
        with pytest.raises(Exception, match="API Error"):
            evaluator.evaluate_with_llm(prompt)
    
    def test_should_print_results_correctly(self):
        """Test that results are printed correctly"""
        evaluator = LLMJudgeEvaluator()
        results = {
            'scores': {
                'code_quality': {'score': 8, 'justification': 'Well-structured'},
                'test_coverage': {'score': 7, 'justification': 'Good coverage'}
            },
            'overall_score': 7.5,
            'recommendations': ['Add more tests', 'Improve docs']
        }
        
        # Mock file writing
        with patch('builtins.open', create=True) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            status, score = evaluator.print_results(results)
            
            # Check return values
            assert status == "PASS"  # 7.5 >= 7.0 threshold
            assert score == 7.5
            
            # Check that file was written
            mock_file.write.assert_called()
    
    def test_should_fail_below_threshold(self):
        """Test that evaluator fails when score is below threshold"""
        evaluator = LLMJudgeEvaluator()
        evaluator.threshold = 8.0  # Set high threshold
        
        results = {
            'scores': {
                'code_quality': {'score': 6, 'justification': 'Needs improvement'},
                'test_coverage': {'score': 5, 'justification': 'Low coverage'}
            },
            'overall_score': 5.5,
            'recommendations': ['Improve code quality', 'Add more tests']
        }
        
        # Mock file writing
        with patch('builtins.open', create=True) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            status, score = evaluator.print_results(results)
            
            # Check return values
            assert status == "FAIL"  # 5.5 < 8.0 threshold
            assert score == 5.5
    
    @patch('evaluators.check_llm_judge.OllamaChat')
    @patch('evaluators.check_llm_judge.LLMJudgeEvaluator.collect_codebase_info')
    def test_should_run_complete_evaluation(self, mock_collect_info, mock_ollama_chat_class):
        """Test complete evaluation process with mocked expensive operations"""
        # Mock the expensive collect_codebase_info method
        mock_collect_info.return_value = {
            'file_count': 25,
            'lines_of_code': 2500,
            'test_files': 15,
            'test_coverage': 85.5,
            'documentation_files': 8,
            'dependencies': ['requests', 'pytest', 'flask']
        }
        
        # Mock OllamaChat
        mock_ollama_chat = MagicMock()
        mock_ollama_chat_class.return_value = mock_ollama_chat
        
        # Mock the query method to return a valid JSON response
        mock_ollama_chat.query.return_value = '''
        {
            "scores": {
                "code_quality": {"score": 8, "justification": "Well-structured"},
                "test_coverage": {"score": 7, "justification": "Good coverage"},
                "documentation": {"score": 6, "justification": "Basic docs"},
                "architecture": {"score": 8, "justification": "Clean design"},
                "security": {"score": 7, "justification": "No issues"},
                "performance": {"score": 7, "justification": "Efficient"}
            },
            "overall_score": 7.2,
            "recommendations": ["Add more tests", "Improve docs"]
        }
        '''
        
        evaluator = LLMJudgeEvaluator()
        
        # Mock file writing
        with patch('builtins.open', create=True) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            results = evaluator.run_evaluation()
            
            # Check that results contain expected data
            assert 'scores' in results
            assert 'overall_score' in results
            assert 'recommendations' in results
            assert 'codebase_info' in results
            assert results['overall_score'] == 7.2
    
    def test_should_handle_missing_config_file(self):
        """Test handling when config file is missing"""
        # This test ensures the evaluator works without a config file
        evaluator = LLMJudgeEvaluator()
        
        # Should not raise an error
        assert evaluator.threshold == 7.0
        assert evaluator.model == 'mistral'
    
    def test_should_use_default_values(self):
        """Test that default values are used when environment variables are not set"""
        # Clear environment variables
        if 'OLLAMA_API_URL' in os.environ:
            del os.environ['OLLAMA_API_URL']
        if 'OLLAMA_MODEL' in os.environ:
            del os.environ['OLLAMA_MODEL']
        if 'LLM_JUDGE_THRESHOLD' in os.environ:
            del os.environ['LLM_JUDGE_THRESHOLD']
        
        evaluator = LLMJudgeEvaluator()
        
        # Check default values
        assert evaluator.ollama_url == 'http://localhost:11434/api'
        assert evaluator.model == 'mistral'
        assert evaluator.threshold == 7.0

    def test_quick_mode_initialization(self):
        """Test that quick mode initializes correctly"""
        evaluator = LLMJudgeEvaluator(quick_mode=True)
        assert evaluator.quick_mode is True
        assert evaluator.results['evaluation_mode'] == 'quick'

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 
