#!/usr/bin/env python3
"""
Test script for LLM Judge evaluation
This script tests the LLM judge functionality and ensures it's working correctly.
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def test_llm_judge_import():
    """Test that the LLM judge can be imported"""
    try:
        from basicchat.evaluation.evaluators.check_llm_judge import LLMJudgeEvaluator
        print("✅ LLM Judge import successful")
        return True
    except ImportError as e:
        print(f"❌ LLM Judge import failed: {e}")
        return False

def test_rules_loading():
    """Test that evaluation rules can be loaded"""
    try:
        rules_file = Path("basicchat/evaluation/evaluators/llm_judge_rules.json")
        if rules_file.exists():
            with open(rules_file, 'r') as f:
                rules = json.load(f)
            print(f"✅ Rules loaded successfully (version: {rules.get('version', 'unknown')})")
            return True
        else:
            print("⚠️ Rules file not found, using defaults")
            return True
    except Exception as e:
        print(f"❌ Rules loading failed: {e}")
        return False

def test_ollama_connection():
    """Test Ollama connection"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama connection successful")
            return True
        else:
            print(f"❌ Ollama connection failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        return False

def test_quick_evaluation():
    """Test a quick evaluation"""
    try:
        from basicchat.evaluation.evaluators.check_llm_judge import LLMJudgeEvaluator
        
        print("🧪 Running quick evaluation test...")
        evaluator = LLMJudgeEvaluator(quick_mode=True)
        
        # Test codebase info collection
        info = evaluator.collect_codebase_info()
        print(f"✅ Codebase info collected: {info['file_count']} files, {info['lines_of_code']} lines")
        
        # Test prompt generation
        prompt = evaluator.generate_evaluation_prompt(info)
        print(f"✅ Prompt generated: {len(prompt)} characters")
        
        print("✅ Quick evaluation test completed")
        return True
    except Exception as e:
        print(f"❌ Quick evaluation test failed: {e}")
        return False

def test_report_generation():
    """Test report generation script"""
    try:
        # Create a mock results file for testing
        mock_results = {
            "scores": {
                "code_quality": {"score": 7, "justification": "Good structure with room for improvement"},
                "test_coverage": {"score": 6, "justification": "Basic testing present"},
                "documentation": {"score": 5, "justification": "Minimal documentation"},
                "architecture": {"score": 8, "justification": "Well-designed architecture"},
                "security": {"score": 7, "justification": "Basic security practices"},
                "performance": {"score": 6, "justification": "Acceptable performance"}
            },
            "overall_score": 6.5,
            "recommendations": ["Add more tests", "Improve documentation"]
        }
        
        with open('llm_judge_results.json', 'w') as f:
            json.dump(mock_results, f)
        
        # Test report generation
        result = subprocess.run([sys.executable, 'scripts/generate_llm_judge_report.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Report generation successful")
            
            # Check if files were created
            if os.path.exists('llm_judge_action_items.md'):
                print("✅ Action items file created")
            if os.path.exists('llm_judge_improvement_tips.md'):
                print("✅ Improvement tips file created")
            
            return True
        else:
            print(f"❌ Report generation failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Report generation test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Testing LLM Judge Evaluation System")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_llm_judge_import),
        ("Rules Loading", test_rules_loading),
        ("Ollama Connection", test_ollama_connection),
        ("Quick Evaluation", test_quick_evaluation),
        ("Report Generation", test_report_generation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! LLM Judge is ready to use.")
        print("\n🚀 You can now run:")
        print("  make llm-judge-quick    # Quick evaluation")
        print("  make llm-judge          # Full evaluation")
        print("  ./scripts/run_llm_judge.sh quick ollama 7.0  # Custom evaluation")
        return 0
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
