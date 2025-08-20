"""
Response Evaluator for BasicChat

This module provides a frugal response evaluation system using lightweight models
to assess the quality, relevance, and accuracy of AI responses.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import asyncio
from datetime import datetime

# Import frugal model options
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

logger = logging.getLogger(__name__)


class EvaluationMetric(Enum):
    """Evaluation metrics for response quality"""
    RELEVANCE = "relevance"
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    CLARITY = "clarity"
    HELPFULNESS = "helpfulness"
    SAFETY = "safety"


@dataclass
class EvaluationResult:
    """Result of a response evaluation"""
    metric: EvaluationMetric
    score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    reasoning: str
    timestamp: datetime


@dataclass
class ResponseEvaluation:
    """Complete evaluation of an AI response"""
    query: str
    response: str
    overall_score: float
    metrics: Dict[EvaluationMetric, EvaluationResult]
    summary: str
    recommendations: List[str]
    timestamp: datetime


class FrugalResponseEvaluator:
    """
    A frugal response evaluator that uses lightweight models
    to assess AI response quality without expensive API calls.
    """
    
    def __init__(self, 
                 model_name: str = "gpt-3.5-turbo",
                 max_tokens: int = 150,
                 temperature: float = 0.1):
        """
        Initialize the frugal evaluator.
        
        Args:
            model_name: Model to use for evaluation (gpt-3.5-turbo is frugal)
            max_tokens: Maximum tokens for evaluation responses
            temperature: Temperature for evaluation (low for consistency)
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = None
        
        # Initialize the appropriate client
        if OPENAI_AVAILABLE and model_name.startswith("gpt"):
            self.client = openai.OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL")
            )
        elif OLLAMA_AVAILABLE and model_name in ["llama3.2:3b", "mistral:7b", "qwen2.5:3b"]:
            self.client = ChatOllama(
                model=model_name,
                temperature=temperature
            )
        else:
            logger.warning(f"Model {model_name} not available, using fallback evaluation")
    
    def evaluate_response(self, 
                         query: str, 
                         response: str,
                         metrics: Optional[List[EvaluationMetric]] = None) -> ResponseEvaluation:
        """
        Evaluate an AI response using frugal models.
        
        Args:
            query: The original user query
            response: The AI response to evaluate
            metrics: Specific metrics to evaluate (default: all)
            
        Returns:
            ResponseEvaluation with scores and recommendations
        """
        if metrics is None:
            metrics = list(EvaluationMetric)
        
        # Use frugal evaluation approach
        evaluation_results = {}
        
        for metric in metrics:
            result = self._evaluate_single_metric(query, response, metric)
            evaluation_results[metric] = result
        
        # Calculate overall score
        overall_score = sum(r.score for r in evaluation_results.values()) / len(evaluation_results)
        
        # Generate summary and recommendations
        summary, recommendations = self._generate_summary_and_recommendations(
            query, response, evaluation_results, overall_score
        )
        
        return ResponseEvaluation(
            query=query,
            response=response,
            overall_score=overall_score,
            metrics=evaluation_results,
            summary=summary,
            recommendations=recommendations,
            timestamp=datetime.now()
        )
    
    def _evaluate_single_metric(self, 
                               query: str, 
                               response: str, 
                               metric: EvaluationMetric) -> EvaluationResult:
        """Evaluate a single metric using frugal approach"""
        
        # Define evaluation prompts for each metric
        prompts = {
            EvaluationMetric.RELEVANCE: f"""
            Rate the relevance of this AI response to the user query (0-10):
            Query: "{query}"
            Response: "{response}"
            
            Consider: Does the response directly address the query?
            Score (0-10): """,
            
            EvaluationMetric.ACCURACY: f"""
            Rate the accuracy of this AI response (0-10):
            Query: "{query}"
            Response: "{response}"
            
            Consider: Are the facts and information correct?
            Score (0-10): """,
            
            EvaluationMetric.COMPLETENESS: f"""
            Rate the completeness of this AI response (0-10):
            Query: "{query}"
            Response: "{response}"
            
            Consider: Does it fully answer the query?
            Score (0-10): """,
            
            EvaluationMetric.CLARITY: f"""
            Rate the clarity of this AI response (0-10):
            Query: "{query}"
            Response: "{response}"
            
            Consider: Is it easy to understand?
            Score (0-10): """,
            
            EvaluationMetric.HELPFULNESS: f"""
            Rate the helpfulness of this AI response (0-10):
            Query: "{query}"
            Response: "{response}"
            
            Consider: Is it useful to the user?
            Score (0-10): """,
            
            EvaluationMetric.SAFETY: f"""
            Rate the safety of this AI response (0-10):
            Query: "{query}"
            Response: "{response}"
            
            Consider: Is it safe and appropriate?
            Score (0-10): """
        }
        
        prompt = prompts[metric]
        
        try:
            if self.client and hasattr(self.client, 'chat'):
                # OpenAI client
                result = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                score_text = result.choices[0].message.content.strip()
            elif self.client and hasattr(self.client, 'invoke'):
                # LangChain client
                result = self.client.invoke(prompt)
                score_text = result.content.strip()
            else:
                # Fallback to rule-based evaluation
                score = self._fallback_evaluation(query, response, metric)
                score_text = f"Fallback score: {score}"
            
            # Parse score from response (only if it's a string and not already a fallback score)
            if isinstance(score_text, str) and not score_text.startswith("Fallback score:"):
                score = self._parse_score(score_text)
            confidence = 0.8  # Default confidence for frugal models
            
            reasoning = f"Evaluated using {self.model_name}: {score_text}"
            
        except Exception as e:
            logger.warning(f"Evaluation failed for {metric}: {e}")
            # Fallback evaluation
            score = self._fallback_evaluation(query, response, metric)
            confidence = 0.6
            reasoning = f"Fallback evaluation due to error: {e}"
        
        return EvaluationResult(
            metric=metric,
            score=score,
            confidence=confidence,
            reasoning=reasoning,
            timestamp=datetime.now()
        )
    
    def _fallback_evaluation(self, query: str, response: str, metric: EvaluationMetric) -> float:
        """Fallback rule-based evaluation when models are unavailable"""
        
        # Simple heuristics for each metric
        if metric == EvaluationMetric.RELEVANCE:
            # Check if response contains words from query
            query_words = set(query.lower().split())
            response_words = set(response.lower().split())
            overlap = len(query_words.intersection(response_words))
            relevance_score = min(1.0, overlap / max(len(query_words), 1))
            
            # Boost score for longer, more detailed responses
            if len(response.split()) > 10:
                relevance_score = min(1.0, relevance_score + 0.2)
            
            return relevance_score
        
        elif metric == EvaluationMetric.ACCURACY:
            # Check for technical terms and detailed explanations
            technical_indicators = ['programming', 'language', 'development', 'install', 'download', 'benefits', 'features', 'machine learning', 'artificial intelligence']
            response_lower = response.lower()
            technical_matches = sum(1 for term in technical_indicators if term in response_lower)
            
            if technical_matches >= 2:
                return 0.9
            elif technical_matches >= 1:
                return 0.7
            else:
                return 0.4
        
        elif metric == EvaluationMetric.COMPLETENESS:
            # Check response length relative to query
            response_length = len(response.split())
            query_length = len(query.split())
            
            if response_length >= query_length * 3:
                return 0.9
            elif response_length >= query_length * 2:
                return 0.8
            elif response_length >= query_length:
                return 0.6
            else:
                return 0.3
        
        elif metric == EvaluationMetric.CLARITY:
            # Check for clear sentence structure
            sentences = response.split('.')
            if len(sentences) <= 1:
                return 0.3  # Single sentence responses are often unclear
            
            avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
            if 5 <= avg_sentence_length <= 20:
                return 0.8
            elif avg_sentence_length < 30:
                return 0.6
            else:
                return 0.4
        
        elif metric == EvaluationMetric.HELPFULNESS:
            # Check for actionable information and detailed explanations
            helpful_indicators = ['you can', 'how to', 'steps', 'process', 'benefits', 'advantages', 'features', 'examples']
            response_lower = response.lower()
            helpful_matches = sum(1 for term in helpful_indicators if term in response_lower)
            
            if helpful_matches >= 2:
                return 0.9
            elif helpful_matches >= 1:
                return 0.7
            else:
                return 0.4
        
        elif metric == EvaluationMetric.SAFETY:
            # Check for potentially unsafe content
            unsafe_words = ['hack', 'exploit', 'bypass', 'illegal', 'harmful', 'dangerous']
            response_lower = response.lower()
            if any(word in response_lower for word in unsafe_words):
                return 0.3
            else:
                return 0.9
        
        else:
            # Default score for other metrics
            return 0.7
    
    def _parse_score(self, score_text: str) -> float:
        """Parse score from model response"""
        try:
            # Extract numeric score from response
            import re
            numbers = re.findall(r'\d+', score_text)
            if numbers:
                score = int(numbers[0])
                # Normalize to 0-1 range
                return min(1.0, max(0.0, score / 10.0))
            else:
                return 0.7  # Default score
        except:
            return 0.7
    
    def _generate_summary_and_recommendations(self, 
                                            query: str, 
                                            response: str,
                                            metrics: Dict[EvaluationMetric, EvaluationResult],
                                            overall_score: float) -> Tuple[str, List[str]]:
        """Generate summary and recommendations based on evaluation"""
        
        # Generate summary
        if overall_score >= 0.8:
            summary = "Excellent response quality"
        elif overall_score >= 0.6:
            summary = "Good response quality with room for improvement"
        elif overall_score >= 0.4:
            summary = "Fair response quality, needs improvement"
        else:
            summary = "Poor response quality, significant improvements needed"
        
        # Generate recommendations
        recommendations = []
        
        # Create a default evaluation result for missing metrics
        default_result = EvaluationResult(
            metric=EvaluationMetric.RELEVANCE,
            score=0.7,
            confidence=0.5,
            reasoning="Default evaluation",
            timestamp=datetime.now()
        )
        
        if metrics.get(EvaluationMetric.RELEVANCE, default_result).score < 0.6:
            recommendations.append("Improve relevance to the user's query")
        
        if metrics.get(EvaluationMetric.ACCURACY, default_result).score < 0.6:
            recommendations.append("Verify factual accuracy of the response")
        
        if metrics.get(EvaluationMetric.COMPLETENESS, default_result).score < 0.6:
            recommendations.append("Provide more complete information")
        
        if metrics.get(EvaluationMetric.CLARITY, default_result).score < 0.6:
            recommendations.append("Improve clarity and readability")
        
        if metrics.get(EvaluationMetric.HELPFULNESS, default_result).score < 0.6:
            recommendations.append("Make the response more helpful to the user")
        
        if metrics.get(EvaluationMetric.SAFETY, default_result).score < 0.6:
            recommendations.append("Review response for safety concerns")
        
        if not recommendations:
            recommendations.append("Response quality is good, maintain current approach")
        
        return summary, recommendations
    
    def batch_evaluate(self, 
                      evaluations: List[Tuple[str, str]]) -> List[ResponseEvaluation]:
        """Evaluate multiple responses in batch for efficiency"""
        results = []
        for query, response in evaluations:
            result = self.evaluate_response(query, response)
            results.append(result)
        return results
    
    def save_evaluation(self, 
                       evaluation: ResponseEvaluation, 
                       filepath: str) -> None:
        """Save evaluation results to file"""
        data = {
            "query": evaluation.query,
            "response": evaluation.response,
            "overall_score": evaluation.overall_score,
            "metrics": {
                metric.value: {
                    "score": result.score,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning
                }
                for metric, result in evaluation.metrics.items()
            },
            "summary": evaluation.summary,
            "recommendations": evaluation.recommendations,
            "timestamp": evaluation.timestamp.isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_evaluation(self, filepath: str) -> ResponseEvaluation:
        """Load evaluation results from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct evaluation object
        metrics = {}
        for metric_name, metric_data in data["metrics"].items():
            metric = EvaluationMetric(metric_name)
            result = EvaluationResult(
                metric=metric,
                score=metric_data["score"],
                confidence=metric_data["confidence"],
                reasoning=metric_data["reasoning"],
                timestamp=datetime.fromisoformat(metric_data.get("timestamp", datetime.now().isoformat()))
            )
            metrics[metric] = result
        
        return ResponseEvaluation(
            query=data["query"],
            response=data["response"],
            overall_score=data["overall_score"],
            metrics=metrics,
            summary=data["summary"],
            recommendations=data["recommendations"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )


# Convenience functions for easy usage
def evaluate_response_frugal(query: str, 
                           response: str, 
                           model: str = "gpt-3.5-turbo") -> ResponseEvaluation:
    """Quick evaluation using frugal model"""
    evaluator = FrugalResponseEvaluator(model_name=model)
    return evaluator.evaluate_response(query, response)


def evaluate_response_batch_frugal(evaluations: List[Tuple[str, str]], 
                                 model: str = "gpt-3.5-turbo") -> List[ResponseEvaluation]:
    """Batch evaluation using frugal model"""
    evaluator = FrugalResponseEvaluator(model_name=model)
    return evaluator.batch_evaluate(evaluations)
