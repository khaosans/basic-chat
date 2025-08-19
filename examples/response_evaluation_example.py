#!/usr/bin/env python3
"""
Response Evaluation Example

This example demonstrates how to use the frugal response evaluator
to assess AI response quality using lightweight models.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from response_evaluator import (
    FrugalResponseEvaluator,
    evaluate_response_frugal,
    evaluate_response_batch_frugal,
    EvaluationMetric
)


def main():
    """Main example function"""
    print("🤖 BasicChat Response Evaluator Example")
    print("=" * 50)
    
    # Example queries and responses
    examples = [
        {
            "query": "What is Python?",
            "response": "Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used in web development, data science, AI, and automation."
        },
        {
            "query": "How do I install Python?",
            "response": "You can download Python from python.org and run the installer."
        },
        {
            "query": "What are the benefits of using Python?",
            "response": "Python offers excellent readability, extensive libraries, cross-platform compatibility, and strong community support."
        }
    ]
    
    # Initialize evaluator with frugal model
    print("\n📊 Initializing frugal response evaluator...")
    evaluator = FrugalResponseEvaluator(
        model_name="gpt-3.5-turbo",  # Frugal model choice
        max_tokens=100,  # Keep responses short
        temperature=0.1  # Low temperature for consistency
    )
    
    # Evaluate each example
    print("\n🔍 Evaluating response quality...")
    for i, example in enumerate(examples, 1):
        print(f"\n--- Example {i} ---")
        print(f"Query: {example['query']}")
        print(f"Response: {example['response']}")
        
        # Evaluate the response
        evaluation = evaluator.evaluate_response(
            example['query'], 
            example['response']
        )
        
        # Display results
        print(f"\n📈 Overall Score: {evaluation.overall_score:.2f}/1.0")
        print(f"📝 Summary: {evaluation.summary}")
        
        print("\n📊 Detailed Metrics:")
        for metric, result in evaluation.metrics.items():
            print(f"  • {metric.value.capitalize()}: {result.score:.2f} (confidence: {result.confidence:.2f})")
        
        print("\n💡 Recommendations:")
        for rec in evaluation.recommendations:
            print(f"  • {rec}")
    
    # Demonstrate batch evaluation
    print("\n" + "=" * 50)
    print("🔄 Batch Evaluation Example")
    print("=" * 50)
    
    # Prepare batch data
    batch_data = [
        (example['query'], example['response']) 
        for example in examples
    ]
    
    # Use convenience function for batch evaluation
    batch_results = evaluate_response_batch_frugal(
        batch_data, 
        model="gpt-3.5-turbo"
    )
    
    print(f"\n✅ Evaluated {len(batch_results)} responses in batch")
    
    # Show batch summary
    print("\n📊 Batch Summary:")
    for i, result in enumerate(batch_results, 1):
        print(f"  Response {i}: {result.overall_score:.2f}/1.0 - {result.summary}")
    
    # Demonstrate specific metric evaluation
    print("\n" + "=" * 50)
    print("🎯 Specific Metric Evaluation")
    print("=" * 50)
    
    # Evaluate only relevance and clarity
    specific_metrics = [EvaluationMetric.RELEVANCE, EvaluationMetric.CLARITY]
    
    for i, example in enumerate(examples, 1):
        print(f"\n--- Example {i} (Relevance & Clarity Only) ---")
        
        evaluation = evaluator.evaluate_response(
            example['query'], 
            example['response'],
            metrics=specific_metrics
        )
        
        print(f"Query: {example['query']}")
        print(f"Relevance: {evaluation.metrics[EvaluationMetric.RELEVANCE].score:.2f}")
        print(f"Clarity: {evaluation.metrics[EvaluationMetric.CLARITY].score:.2f}")
    
    # Demonstrate saving and loading
    print("\n" + "=" * 50)
    print("💾 Save/Load Example")
    print("=" * 50)
    
    # Evaluate and save
    example_evaluation = evaluator.evaluate_response(
        examples[0]['query'], 
        examples[0]['response']
    )
    
    # Save to file
    save_path = "example_evaluation.json"
    evaluator.save_evaluation(example_evaluation, save_path)
    print(f"✅ Saved evaluation to {save_path}")
    
    # Load from file
    loaded_evaluation = evaluator.load_evaluation(save_path)
    print(f"✅ Loaded evaluation from {save_path}")
    print(f"📊 Loaded score: {loaded_evaluation.overall_score:.2f}")
    
    # Clean up
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"🧹 Cleaned up {save_path}")
    
    print("\n" + "=" * 50)
    print("🎉 Response Evaluation Example Complete!")
    print("=" * 50)
    
    print("\n💡 Key Benefits of Frugal Evaluation:")
    print("  • Uses lightweight models (gpt-3.5-turbo, llama3.2:3b)")
    print("  • Fallback to rule-based evaluation when models unavailable")
    print("  • Batch processing for efficiency")
    print("  • Comprehensive metrics: relevance, accuracy, completeness, clarity, helpfulness, safety")
    print("  • Actionable recommendations for improvement")
    print("  • JSON export/import for analysis")


if __name__ == "__main__":
    main()
