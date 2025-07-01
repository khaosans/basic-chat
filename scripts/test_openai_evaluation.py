#!/usr/bin/env python3
"""
Test script for OpenAI LLM Judge evaluation

This script tests the OpenAI evaluator with the cheapest model to ensure it works correctly.
"""

import os
import sys
import subprocess
import pytest
pytest.skip("OpenAI evaluator tests require API access", allow_module_level=True)
from pathlib import Path

def test_openai_evaluator():
    """Test the OpenAI evaluator with quick mode"""
    print("🧪 Testing OpenAI LLM Judge Evaluator...")
    
    # Check if OpenAI API key is available
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("⚠️  No OPENAI_API_KEY found in environment")
        print("   Set it with: export OPENAI_API_KEY='your-key-here'")
        return False
    
    # Set environment variables for testing
    os.environ['OPENAI_MODEL'] = 'gpt-3.5-turbo'
    os.environ['LLM_JUDGE_THRESHOLD'] = '7.0'
    
    try:
        # Run the evaluator in quick mode
        print("🤖 Running OpenAI evaluator in quick mode...")
        result = subprocess.run([
            sys.executable, 'evaluators/check_llm_judge_openai.py', '--quick'
        ], capture_output=True, text=True, timeout=120)
        
        print("📋 STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("❌ STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ OpenAI evaluator test PASSED")
            
            # Check if results file was created
            if os.path.exists('llm_judge_results.json'):
                print("📄 Results file created successfully")
                return True
            else:
                print("❌ Results file not found")
                return False
        else:
            print(f"❌ OpenAI evaluator test FAILED (exit code: {result.returncode})")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out after 120 seconds")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting OpenAI LLM Judge Test...")
    print("=" * 50)
    
    success = test_openai_evaluator()
    
    print("=" * 50)
    if success:
        print("🎉 All tests PASSED!")
        sys.exit(0)
    else:
        print("💥 Some tests FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    main() 
