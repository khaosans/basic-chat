#!/usr/bin/env python3
"""
Test script for quick LLM Judge evaluation mode

This script tests the quick evaluation mode to ensure it works correctly
and provides faster results for CI/CD pipelines.
"""

import sys
import os
import subprocess
import pytest
pytest.skip("Quick evaluation script tests require Ollama", allow_module_level=True)
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_quick_evaluation():
    """Test the quick evaluation mode"""
    print("🧪 Testing Quick LLM Judge Evaluation Mode")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('evaluators/check_llm_judge.py'):
        print("❌ Error: evaluators/check_llm_judge.py not found")
        print("   Please run this script from the project root directory")
        return False
    
    # Test the quick mode argument parsing
    try:
        result = subprocess.run([
            sys.executable, 'evaluators/check_llm_judge.py', '--help'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Help command works correctly")
            if '--quick' in result.stdout:
                print("✅ Quick mode argument is available")
            else:
                print("❌ Quick mode argument not found in help")
                return False
        else:
            print(f"❌ Help command failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Help command timed out")
        return False
    except Exception as e:
        print(f"❌ Help command failed with exception: {e}")
        return False
    
    # Test quick mode without Ollama (should fail gracefully)
    print("\n🔍 Testing quick mode without Ollama (expected to fail)...")
    try:
        result = subprocess.run([
            sys.executable, 'evaluators/check_llm_judge.py', '--quick'
        ], capture_output=True, text=True, timeout=60)
        
        # Should fail because Ollama is not running, but should show quick mode
        if 'QUICK MODE' in result.stdout or 'quick mode' in result.stdout.lower():
            print("✅ Quick mode is being used")
        else:
            print("❌ Quick mode not detected in output")
            print(f"Output: {result.stdout}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Quick mode test timed out")
        return False
    except Exception as e:
        print(f"❌ Quick mode test failed with exception: {e}")
        return False
    
    print("\n✅ Quick evaluation mode test completed successfully")
    print("📝 Note: Full evaluation requires Ollama to be running")
    return True

def main():
    """Main entry point"""
    print("🚀 LLM Judge Quick Mode Test")
    print("=" * 30)
    
    success = test_quick_evaluation()
    
    if success:
        print("\n🎉 All tests passed!")
        print("\n💡 To run full evaluation with Ollama:")
        print("   1. Start Ollama: ollama serve")
        print("   2. Pull model: ollama pull mistral")
        print("   3. Run: python evaluators/check_llm_judge.py --quick")
        return 0
    else:
        print("\n❌ Tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
