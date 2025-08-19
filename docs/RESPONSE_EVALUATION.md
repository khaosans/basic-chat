# Response Evaluation System

## Overview

The BasicChat Response Evaluation System provides a frugal, cost-effective way to assess AI response quality using lightweight models. This system helps ensure that AI responses meet quality standards while minimizing costs.

## Features

### 🎯 **Frugal Model Support**
- **OpenAI Models**: `gpt-3.5-turbo` (recommended for cost-effectiveness)
- **Ollama Models**: `llama3.2:3b`, `mistral:7b`, `qwen2.5:3b`
- **Fallback System**: Rule-based evaluation when models are unavailable

### 📊 **Comprehensive Metrics**
- **Relevance**: Does the response address the query?
- **Accuracy**: Are the facts and information correct?
- **Completeness**: Does it fully answer the query?
- **Clarity**: Is it easy to understand?
- **Helpfulness**: Is it useful to the user?
- **Safety**: Is it safe and appropriate?

### ⚡ **Performance Features**
- **Batch Processing**: Evaluate multiple responses efficiently
- **JSON Export/Import**: Save and load evaluation results
- **Configurable Parameters**: Customize model, tokens, temperature
- **Actionable Recommendations**: Get specific improvement suggestions

## Quick Start

### Basic Usage

```python
from response_evaluator import evaluate_response_frugal

# Evaluate a single response
query = "What is Python?"
response = "Python is a programming language used for web development and data science."

evaluation = evaluate_response_frugal(query, response)
print(f"Overall Score: {evaluation.overall_score:.2f}")
print(f"Summary: {evaluation.summary}")
```

### Advanced Usage

```python
from response_evaluator import FrugalResponseEvaluator, EvaluationMetric

# Initialize evaluator with custom settings
evaluator = FrugalResponseEvaluator(
    model_name="gpt-3.5-turbo",
    max_tokens=150,
    temperature=0.1
)

# Evaluate with specific metrics
metrics = [EvaluationMetric.RELEVANCE, EvaluationMetric.CLARITY]
evaluation = evaluator.evaluate_response(query, response, metrics)

# Get detailed results
for metric, result in evaluation.metrics.items():
    print(f"{metric.value}: {result.score:.2f} (confidence: {result.confidence:.2f})")
```

### Batch Evaluation

```python
from response_evaluator import evaluate_response_batch_frugal

# Prepare batch data
evaluations = [
    ("What is Python?", "Python is a programming language."),
    ("How to install Python?", "Download from python.org"),
    ("Python benefits?", "Readable, extensive libraries, cross-platform")
]

# Evaluate all responses
results = evaluate_response_batch_frugal(evaluations)

# Process results
for i, result in enumerate(results):
    print(f"Response {i+1}: {result.overall_score:.2f} - {result.summary}")
```

## API Reference

### FrugalResponseEvaluator

#### Constructor

```python
FrugalResponseEvaluator(
    model_name: str = "gpt-3.5-turbo",
    max_tokens: int = 150,
    temperature: float = 0.1
)
```

**Parameters:**
- `model_name`: Model to use for evaluation
- `max_tokens`: Maximum tokens for evaluation responses
- `temperature`: Temperature for evaluation (low for consistency)

#### Methods

##### `evaluate_response(query, response, metrics=None)`

Evaluates a single AI response.

**Parameters:**
- `query`: The original user query
- `response`: The AI response to evaluate
- `metrics`: List of specific metrics to evaluate (default: all)

**Returns:** `ResponseEvaluation` object

##### `batch_evaluate(evaluations)`

Evaluates multiple responses in batch.

**Parameters:**
- `evaluations`: List of (query, response) tuples

**Returns:** List of `ResponseEvaluation` objects

##### `save_evaluation(evaluation, filepath)`

Saves evaluation results to JSON file.

##### `load_evaluation(filepath)`

Loads evaluation results from JSON file.

### Convenience Functions

#### `evaluate_response_frugal(query, response, model="gpt-3.5-turbo")`

Quick evaluation using frugal model.

#### `evaluate_response_batch_frugal(evaluations, model="gpt-3.5-turbo")`

Quick batch evaluation using frugal model.

## Data Structures

### ResponseEvaluation

```python
@dataclass
class ResponseEvaluation:
    query: str
    response: str
    overall_score: float  # 0.0 to 1.0
    metrics: Dict[EvaluationMetric, EvaluationResult]
    summary: str
    recommendations: List[str]
    timestamp: datetime
```

### EvaluationResult

```python
@dataclass
class EvaluationResult:
    metric: EvaluationMetric
    score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    reasoning: str
    timestamp: datetime
```

### EvaluationMetric

```python
class EvaluationMetric(Enum):
    RELEVANCE = "relevance"
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    CLARITY = "clarity"
    HELPFULNESS = "helpfulness"
    SAFETY = "safety"
```

## Configuration

### Environment Variables

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1  # Optional

# Model Selection
EVALUATION_MODEL=gpt-3.5-turbo  # Default model
```

### Model Recommendations

| Use Case | Recommended Model | Cost | Performance |
|----------|------------------|------|-------------|
| Production | `gpt-3.5-turbo` | Low | High |
| Development | `llama3.2:3b` | Free | Medium |
| Testing | `mistral:7b` | Free | High |
| Offline | `qwen2.5:3b` | Free | Medium |

## Integration Examples

### Streamlit Integration

```python
import streamlit as st
from response_evaluator import evaluate_response_frugal

def evaluate_chat_response(query, response):
    """Evaluate chat response in Streamlit app"""
    evaluation = evaluate_response_frugal(query, response)
    
    # Display results
    st.metric("Overall Score", f"{evaluation.overall_score:.2f}")
    st.write(f"**Summary:** {evaluation.summary}")
    
    # Show recommendations
    if evaluation.recommendations:
        st.write("**Recommendations:**")
        for rec in evaluation.recommendations:
            st.write(f"• {rec}")
    
    return evaluation
```

### API Integration

```python
from flask import Flask, request, jsonify
from response_evaluator import evaluate_response_frugal

app = Flask(__name__)

@app.route('/evaluate', methods=['POST'])
def evaluate_response():
    data = request.json
    query = data.get('query')
    response = data.get('response')
    
    evaluation = evaluate_response_frugal(query, response)
    
    return jsonify({
        'overall_score': evaluation.overall_score,
        'summary': evaluation.summary,
        'recommendations': evaluation.recommendations,
        'metrics': {
            metric.value: {
                'score': result.score,
                'confidence': result.confidence
            }
            for metric, result in evaluation.metrics.items()
        }
    })
```

### Testing Integration

```python
import pytest
from response_evaluator import FrugalResponseEvaluator

class TestResponseQuality:
    def test_response_relevance(self):
        evaluator = FrugalResponseEvaluator(model_name="nonexistent-model")
        query = "What is Python?"
        response = "Python is a programming language."
        
        evaluation = evaluator.evaluate_response(query, response)
        
        # Assert minimum quality standards
        assert evaluation.overall_score >= 0.6
        assert evaluation.metrics[EvaluationMetric.RELEVANCE].score >= 0.7
```

## Best Practices

### 1. **Model Selection**
- Use `gpt-3.5-turbo` for production (cost-effective)
- Use local models for development/testing
- Always have fallback evaluation enabled

### 2. **Batch Processing**
- Group evaluations for efficiency
- Use batch processing for large datasets
- Cache results when possible

### 3. **Error Handling**
```python
try:
    evaluation = evaluate_response_frugal(query, response)
except Exception as e:
    # Fallback to rule-based evaluation
    evaluator = FrugalResponseEvaluator(model_name="nonexistent-model")
    evaluation = evaluator.evaluate_response(query, response)
```

### 4. **Performance Optimization**
- Set appropriate `max_tokens` (100-150 for evaluations)
- Use low temperature (0.1) for consistency
- Cache evaluation results for repeated queries

### 5. **Quality Thresholds**
```python
def is_response_acceptable(evaluation, threshold=0.7):
    """Check if response meets quality standards"""
    return (
        evaluation.overall_score >= threshold and
        evaluation.metrics[EvaluationMetric.SAFETY].score >= 0.8
    )
```

## Troubleshooting

### Common Issues

1. **Model Not Available**
   - Check model name spelling
   - Verify API keys for OpenAI models
   - Ensure Ollama is running for local models

2. **Low Evaluation Scores**
   - Review response content
   - Check for safety concerns
   - Verify response relevance to query

3. **Slow Performance**
   - Reduce `max_tokens`
   - Use batch processing
   - Consider local models for development

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

evaluator = FrugalResponseEvaluator()
evaluation = evaluator.evaluate_response(query, response)
```

## Examples

See `examples/response_evaluation_example.py` for comprehensive usage examples.

## Contributing

To add new evaluation metrics or models:

1. Add new metric to `EvaluationMetric` enum
2. Implement evaluation logic in `_evaluate_single_metric`
3. Add fallback logic in `_fallback_evaluation`
4. Update tests in `tests/test_response_evaluator.py`

## License

This response evaluation system is part of BasicChat and follows the same license terms.
