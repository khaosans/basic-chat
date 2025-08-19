#!/usr/bin/env python3
"""
Integration tests for response evaluation system with prompt quality assessment.

This module tests the systematic evaluation of AI responses using the frugal
response evaluator to assess prompt effectiveness and response quality.
"""

import pytest
import sys
import os
from typing import Dict, List, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from basicchat.evaluation.response_evaluator import (
    FrugalResponseEvaluator,
    evaluate_response_frugal,
    evaluate_response_batch_frugal,
    EvaluationMetric,
    ResponseEvaluation
)


class TestResponseEvaluationIntegration:
    """Integration tests for systematic response evaluation"""
    
    @pytest.fixture
    def evaluator(self):
        """Initialize frugal response evaluator for testing"""
        return FrugalResponseEvaluator(model_name="nonexistent-model")
    
    @pytest.fixture
    def test_prompts(self) -> List[Dict]:
        """Test prompts with expected quality responses"""
        return [
            {
                "query": "What is Python?",
                "high_quality_response": "Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used in web development, data science, AI, and automation.",
                "low_quality_response": "Python is a snake.",
                "expected_high_score": 0.8,
                "expected_low_score": 0.3
            },
            {
                "query": "How do I install Python?",
                "high_quality_response": "You can install Python by downloading it from python.org, running the installer, and following the setup wizard. Make sure to check 'Add Python to PATH' during installation.",
                "low_quality_response": "Just download it.",
                "expected_high_score": 0.8,
                "expected_low_score": 0.4
            },
            {
                "query": "What are the benefits of using Python?",
                "high_quality_response": "Python offers excellent readability, extensive libraries, cross-platform compatibility, strong community support, and is great for beginners and experts alike.",
                "low_quality_response": "It's good.",
                "expected_high_score": 0.8,
                "expected_low_score": 0.3
            },
            {
                "query": "Explain machine learning",
                "high_quality_response": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to identify patterns in data.",
                "low_quality_response": "It's when computers learn.",
                "expected_high_score": 0.8,
                "expected_low_score": 0.3
            }
        ]
    
    def test_systematic_prompt_evaluation(self, evaluator, test_prompts):
        """Test systematic evaluation of prompt responses"""
        print("\n🤖 Testing Systematic Prompt Evaluation")
        print("=" * 60)
        
        total_tests = 0
        passed_tests = 0
        evaluation_results = []
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n--- Test Case {i}: {prompt['query']} ---")
            
            # Evaluate high-quality response
            high_eval = evaluator.evaluate_response(
                prompt['query'], 
                prompt['high_quality_response']
            )
            
            # Evaluate low-quality response
            low_eval = evaluator.evaluate_response(
                prompt['query'], 
                prompt['low_quality_response']
            )
            
            # Store results for analysis
            evaluation_results.append({
                'query': prompt['query'],
                'high_score': high_eval.overall_score,
                'low_score': low_eval.overall_score,
                'high_summary': high_eval.summary,
                'low_summary': low_eval.summary
            })
            
            # Validate score ordering
            total_tests += 2
            if high_eval.overall_score > low_eval.overall_score:
                passed_tests += 2
                print(f"✅ PASS: High-quality response ({high_eval.overall_score:.2f}) > Low-quality ({low_eval.overall_score:.2f})")
            else:
                print(f"❌ FAIL: Score ordering incorrect - High: {high_eval.overall_score:.2f}, Low: {low_eval.overall_score:.2f}")
            
            # Display detailed metrics for high-quality response
            print(f"📊 High-quality metrics:")
            for metric, result in high_eval.metrics.items():
                print(f"  • {metric.value.capitalize()}: {result.score:.2f}")
        
        # Summary
        print(f"\n" + "=" * 60)
        print(f"🎯 Evaluation Results: {passed_tests}/{total_tests} tests passed")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        # Assert overall success
        assert passed_tests == total_tests, f"Only {passed_tests}/{total_tests} tests passed"
        
        return evaluation_results
    
    def test_batch_evaluation_performance(self, evaluator, test_prompts):
        """Test batch evaluation performance and consistency"""
        print("\n🔄 Testing Batch Evaluation Performance")
        print("=" * 60)
        
        # Prepare batch data
        batch_data = []
        for prompt in test_prompts:
            batch_data.append((prompt['query'], prompt['high_quality_response']))
            batch_data.append((prompt['query'], prompt['low_quality_response']))
        
        # Run batch evaluation
        batch_results = evaluator.batch_evaluate(batch_data)
        
        print(f"✅ Evaluated {len(batch_results)} responses in batch")
        
        # Verify batch results match individual results
        for i, result in enumerate(batch_results):
            query, response = batch_data[i]
            individual_result = evaluator.evaluate_response(query, response)
            
            # Scores should be similar (allowing for small variations)
            score_diff = abs(result.overall_score - individual_result.overall_score)
            assert score_diff < 0.1, f"Batch vs individual score difference too large: {score_diff}"
        
        print("✅ Batch evaluation consistency verified")
    
    def test_metric_specific_evaluation(self, evaluator, test_prompts):
        """Test evaluation with specific metrics only"""
        print("\n🎯 Testing Metric-Specific Evaluation")
        print("=" * 60)
        
        # Test with only relevance and clarity metrics
        specific_metrics = [EvaluationMetric.RELEVANCE, EvaluationMetric.CLARITY]
        
        for prompt in test_prompts:
            evaluation = evaluator.evaluate_response(
                prompt['query'],
                prompt['high_quality_response'],
                metrics=specific_metrics
            )
            
            # Verify only specified metrics are present
            assert len(evaluation.metrics) == 2
            assert EvaluationMetric.RELEVANCE in evaluation.metrics
            assert EvaluationMetric.CLARITY in evaluation.metrics
            assert EvaluationMetric.ACCURACY not in evaluation.metrics
            
            print(f"✅ {prompt['query']}: Relevance={evaluation.metrics[EvaluationMetric.RELEVANCE].score:.2f}, Clarity={evaluation.metrics[EvaluationMetric.CLARITY].score:.2f}")
    
    def test_evaluation_thresholds(self, evaluator, test_prompts):
        """Test evaluation quality thresholds"""
        print("\n📊 Testing Evaluation Quality Thresholds")
        print("=" * 60)
        
        high_quality_threshold = 0.7
        low_quality_threshold = 0.5
        
        for prompt in test_prompts:
            high_eval = evaluator.evaluate_response(
                prompt['query'], 
                prompt['high_quality_response']
            )
            low_eval = evaluator.evaluate_response(
                prompt['query'], 
                prompt['low_quality_response']
            )
            
            # High-quality responses should meet threshold
            assert high_eval.overall_score >= high_quality_threshold, \
                f"High-quality response scored {high_eval.overall_score:.2f} < {high_quality_threshold}"
            
            # Low-quality responses should be below threshold
            assert low_eval.overall_score < low_quality_threshold, \
                f"Low-quality response scored {low_eval.overall_score:.2f} >= {low_quality_threshold}"
            
            print(f"✅ {prompt['query']}: High={high_eval.overall_score:.2f}, Low={low_eval.overall_score:.2f}")
    
    def test_evaluation_recommendations(self, evaluator, test_prompts):
        """Test that evaluations provide actionable recommendations"""
        print("\n💡 Testing Evaluation Recommendations")
        print("=" * 60)
        
        for prompt in test_prompts:
            # Test high-quality response
            high_eval = evaluator.evaluate_response(
                prompt['query'], 
                prompt['high_quality_response']
            )
            
            # Test low-quality response
            low_eval = evaluator.evaluate_response(
                prompt['query'], 
                prompt['low_quality_response']
            )
            
            # Both should have recommendations
            assert len(high_eval.recommendations) > 0, "High-quality response missing recommendations"
            assert len(low_eval.recommendations) > 0, "Low-quality response missing recommendations"
            
            # Low-quality responses should have more recommendations
            assert len(low_eval.recommendations) >= len(high_eval.recommendations), \
                "Low-quality response should have more recommendations"
            
            print(f"✅ {prompt['query']}: High={len(high_eval.recommendations)} recs, Low={len(low_eval.recommendations)} recs")


def test_convenience_functions():
    """Test convenience functions for response evaluation"""
    print("\n🚀 Testing Convenience Functions")
    print("=" * 60)
    
    query = "What is Python?"
    high_response = "Python is a high-level, interpreted programming language."
    low_response = "Python is a snake."
    
    # Test single evaluation
    high_eval = evaluate_response_frugal(query, high_response, model="nonexistent-model")
    low_eval = evaluate_response_frugal(query, low_response, model="nonexistent-model")
    
    assert isinstance(high_eval, ResponseEvaluation)
    assert isinstance(low_eval, ResponseEvaluation)
    assert high_eval.overall_score > low_eval.overall_score
    
    # Test batch evaluation
    batch_data = [(query, high_response), (query, low_response)]
    batch_results = evaluate_response_batch_frugal(batch_data, model="nonexistent-model")
    
    assert len(batch_results) == 2
    assert all(isinstance(r, ResponseEvaluation) for r in batch_results)
    
    print("✅ Convenience functions working correctly")


if __name__ == "__main__":
    # Run the integration tests
    pytest.main([__file__, "-v", "-s"])
