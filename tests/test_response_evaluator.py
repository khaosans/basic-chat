"""
Tests for the Response Evaluator module
"""
import pytest
import tempfile
import os
from datetime import datetime
from basicchat.evaluation.response_evaluator import (
    FrugalResponseEvaluator,
    EvaluationMetric,
    EvaluationResult,
    ResponseEvaluation,
    evaluate_response_frugal,
    evaluate_response_batch_frugal
)


class TestFrugalResponseEvaluator:
    """Test class for FrugalResponseEvaluator"""
    
    def test_initialization(self):
        """Test evaluator initialization"""
        evaluator = FrugalResponseEvaluator()
        assert evaluator.model_name == "gpt-3.5-turbo"
        assert evaluator.max_tokens == 150
        assert evaluator.temperature == 0.1
    
    def test_initialization_with_custom_params(self):
        """Test evaluator initialization with custom parameters"""
        evaluator = FrugalResponseEvaluator(
            model_name="mistral:7b",
            max_tokens=200,
            temperature=0.2
        )
        assert evaluator.model_name == "mistral:7b"
        assert evaluator.max_tokens == 200
        assert evaluator.temperature == 0.2
    
    @pytest.mark.performance
    def test_fallback_evaluation_relevance(self):
        """Test fallback evaluation for relevance metric"""
        evaluator = FrugalResponseEvaluator()
        query = "What is Python?"
        response = "Python is a programming language used for web development and data science."
        
        score = evaluator._fallback_evaluation(query, response, EvaluationMetric.RELEVANCE)
        assert 0.0 <= score <= 1.0
        assert score > 0.0  # Should have some relevance
    
    @pytest.mark.performance
    def test_fallback_evaluation_completeness(self):
        """Test fallback evaluation for completeness metric"""
        evaluator = FrugalResponseEvaluator()
        query = "What is Python?"
        response = "Python is a programming language."
        
        score = evaluator._fallback_evaluation(query, response, EvaluationMetric.COMPLETENESS)
        assert 0.0 <= score <= 1.0
    
    def test_fallback_evaluation_clarity(self):
        """Test fallback evaluation for clarity metric"""
        evaluator = FrugalResponseEvaluator()
        query = "What is Python?"
        response = "Python is a programming language. It is easy to learn."
        
        score = evaluator._fallback_evaluation(query, response, EvaluationMetric.CLARITY)
        assert 0.0 <= score <= 1.0
    
    def test_fallback_evaluation_safety(self):
        """Test fallback evaluation for safety metric"""
        evaluator = FrugalResponseEvaluator()
        
        # Safe response
        safe_response = "Python is a programming language."
        safe_score = evaluator._fallback_evaluation("What is Python?", safe_response, EvaluationMetric.SAFETY)
        assert safe_score > 0.5
        
        # Unsafe response
        unsafe_response = "Here's how to hack into a system."
        unsafe_score = evaluator._fallback_evaluation("How to hack?", unsafe_response, EvaluationMetric.SAFETY)
        assert unsafe_score < 0.5
    
    def test_parse_score_valid(self):
        """Test parsing valid scores from text"""
        evaluator = FrugalResponseEvaluator()
        
        # Test various score formats
        assert evaluator._parse_score("8") == 0.8
        assert evaluator._parse_score("Score: 7") == 0.7
        assert evaluator._parse_score("The score is 9 out of 10") == 0.9
        assert evaluator._parse_score("10") == 1.0
        assert evaluator._parse_score("0") == 0.0
    
    def test_parse_score_invalid(self):
        """Test parsing invalid scores from text"""
        evaluator = FrugalResponseEvaluator()
        
        # Should return default score for invalid inputs
        assert evaluator._parse_score("no score here") == 0.7
        assert evaluator._parse_score("") == 0.7
        assert evaluator._parse_score("Score: invalid") == 0.7
    
    def test_generate_summary_and_recommendations_excellent(self):
        """Test summary generation for excellent scores"""
        evaluator = FrugalResponseEvaluator()
        query = "What is Python?"
        response = "Python is a programming language."
        
        # Create mock evaluation results with high scores
        metrics = {}
        for metric in EvaluationMetric:
            metrics[metric] = EvaluationResult(
                metric=metric,
                score=0.9,
                confidence=0.8,
                reasoning="Test",
                timestamp=datetime.now()
            )
        
        summary, recommendations = evaluator._generate_summary_and_recommendations(
            query, response, metrics, 0.9
        )
        
        assert "Excellent" in summary
        assert len(recommendations) > 0
    
    def test_generate_summary_and_recommendations_poor(self):
        """Test summary generation for poor scores"""
        evaluator = FrugalResponseEvaluator()
        query = "What is Python?"
        response = "Python is a programming language."
        
        # Create mock evaluation results with low scores
        metrics = {}
        for metric in EvaluationMetric:
            metrics[metric] = EvaluationResult(
                metric=metric,
                score=0.3,
                confidence=0.8,
                reasoning="Test",
                timestamp=datetime.now()
            )
        
        summary, recommendations = evaluator._generate_summary_and_recommendations(
            query, response, metrics, 0.3
        )
        
        assert "Poor" in summary
        assert len(recommendations) > 0
    
    def test_evaluate_response_fallback(self):
        """Test full response evaluation with fallback"""
        evaluator = FrugalResponseEvaluator(model_name="nonexistent-model")
        query = "What is Python?"
        response = "Python is a programming language used for web development and data science."
        
        result = evaluator.evaluate_response(query, response)
        
        assert isinstance(result, ResponseEvaluation)
        assert result.query == query
        assert result.response == response
        assert 0.0 <= result.overall_score <= 1.0
        assert len(result.metrics) == len(EvaluationMetric)
        assert len(result.recommendations) > 0
        assert result.summary is not None
    
    def test_evaluate_response_specific_metrics(self):
        """Test evaluation with specific metrics only"""
        evaluator = FrugalResponseEvaluator(model_name="nonexistent-model")
        query = "What is Python?"
        response = "Python is a programming language."
        
        metrics = [EvaluationMetric.RELEVANCE, EvaluationMetric.CLARITY]
        result = evaluator.evaluate_response(query, response, metrics)
        
        assert len(result.metrics) == 2
        assert EvaluationMetric.RELEVANCE in result.metrics
        assert EvaluationMetric.CLARITY in result.metrics
        assert EvaluationMetric.ACCURACY not in result.metrics
    
    def test_batch_evaluate(self):
        """Test batch evaluation"""
        evaluator = FrugalResponseEvaluator(model_name="nonexistent-model")
        evaluations = [
            ("What is Python?", "Python is a programming language."),
            ("What is JavaScript?", "JavaScript is a web programming language.")
        ]
        
        results = evaluator.batch_evaluate(evaluations)
        
        assert len(results) == 2
        assert all(isinstance(r, ResponseEvaluation) for r in results)
        assert results[0].query == "What is Python?"
        assert results[1].query == "What is JavaScript?"
    
    def test_save_and_load_evaluation(self):
        """Test saving and loading evaluation results"""
        evaluator = FrugalResponseEvaluator(model_name="nonexistent-model")
        query = "What is Python?"
        response = "Python is a programming language."
        
        # Create evaluation
        evaluation = evaluator.evaluate_response(query, response)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            evaluator.save_evaluation(evaluation, temp_file)
            
            # Load evaluation
            loaded_evaluation = evaluator.load_evaluation(temp_file)
            
            # Verify loaded data matches original
            assert loaded_evaluation.query == evaluation.query
            assert loaded_evaluation.response == evaluation.response
            assert loaded_evaluation.overall_score == evaluation.overall_score
            assert loaded_evaluation.summary == evaluation.summary
            assert loaded_evaluation.recommendations == evaluation.recommendations
            
            # Verify metrics
            for metric in evaluation.metrics:
                assert metric in loaded_evaluation.metrics
                assert loaded_evaluation.metrics[metric].score == evaluation.metrics[metric].score
                assert loaded_evaluation.metrics[metric].confidence == evaluation.metrics[metric].confidence
        
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestConvenienceFunctions:
    """Test convenience functions"""
    
    def test_evaluate_response_frugal(self):
        """Test convenience function for single evaluation"""
        query = "What is Python?"
        response = "Python is a programming language."
        
        result = evaluate_response_frugal(query, response, model="nonexistent-model")
        
        assert isinstance(result, ResponseEvaluation)
        assert result.query == query
        assert result.response == response
    
    def test_evaluate_response_batch_frugal(self):
        """Test convenience function for batch evaluation"""
        evaluations = [
            ("What is Python?", "Python is a programming language."),
            ("What is JavaScript?", "JavaScript is a web programming language.")
        ]
        
        results = evaluate_response_batch_frugal(evaluations, model="nonexistent-model")
        
        assert len(results) == 2
        assert all(isinstance(r, ResponseEvaluation) for r in results)


class TestEvaluationMetrics:
    """Test evaluation metrics enum"""
    
    def test_evaluation_metrics_values(self):
        """Test that all evaluation metrics have valid values"""
        expected_metrics = [
            "relevance", "accuracy", "completeness", 
            "clarity", "helpfulness", "safety"
        ]
        
        for metric in EvaluationMetric:
            assert metric.value in expected_metrics
    
    def test_evaluation_metrics_count(self):
        """Test that we have the expected number of metrics"""
        assert len(EvaluationMetric) == 6


class TestEvaluationResult:
    """Test EvaluationResult dataclass"""
    
    def test_evaluation_result_creation(self):
        """Test creating an evaluation result"""
        metric = EvaluationMetric.RELEVANCE
        score = 0.8
        confidence = 0.9
        reasoning = "Test reasoning"
        timestamp = datetime.now()
        
        result = EvaluationResult(
            metric=metric,
            score=score,
            confidence=confidence,
            reasoning=reasoning,
            timestamp=timestamp
        )
        
        assert result.metric == metric
        assert result.score == score
        assert result.confidence == confidence
        assert result.reasoning == reasoning
        assert result.timestamp == timestamp
    
    def test_evaluation_result_score_bounds(self):
        """Test that scores are within valid bounds"""
        metric = EvaluationMetric.RELEVANCE
        reasoning = "Test"
        timestamp = datetime.now()
        
        # Test valid scores
        for score in [0.0, 0.5, 1.0]:
            result = EvaluationResult(
                metric=metric,
                score=score,
                confidence=0.8,
                reasoning=reasoning,
                timestamp=timestamp
            )
            assert 0.0 <= result.score <= 1.0


class TestResponseEvaluation:
    """Test ResponseEvaluation dataclass"""
    
    def test_response_evaluation_creation(self):
        """Test creating a response evaluation"""
        query = "What is Python?"
        response = "Python is a programming language."
        overall_score = 0.8
        metrics = {}
        summary = "Good response"
        recommendations = ["Improve clarity"]
        timestamp = datetime.now()
        
        evaluation = ResponseEvaluation(
            query=query,
            response=response,
            overall_score=overall_score,
            metrics=metrics,
            summary=summary,
            recommendations=recommendations,
            timestamp=timestamp
        )
        
        assert evaluation.query == query
        assert evaluation.response == response
        assert evaluation.overall_score == overall_score
        assert evaluation.metrics == metrics
        assert evaluation.summary == summary
        assert evaluation.recommendations == recommendations
        assert evaluation.timestamp == timestamp
    
    def test_response_evaluation_score_bounds(self):
        """Test that overall score is within valid bounds"""
        query = "What is Python?"
        response = "Python is a programming language."
        metrics = {}
        summary = "Test"
        recommendations = []
        timestamp = datetime.now()
        
        # Test valid scores
        for score in [0.0, 0.5, 1.0]:
            evaluation = ResponseEvaluation(
                query=query,
                response=response,
                overall_score=score,
                metrics=metrics,
                summary=summary,
                recommendations=recommendations,
                timestamp=timestamp
            )
            assert 0.0 <= evaluation.overall_score <= 1.0
