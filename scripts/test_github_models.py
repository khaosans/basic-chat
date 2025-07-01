#!/usr/bin/env python3
"""
Test script for GitHub Models LLM Judge Evaluator

This script tests the GitHub Models integration using the provided token.
"""

import os
import sys
import subprocess
import pytest
pytest.skip("GitHub Models tests require external dependencies", allow_module_level=True)
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_github_models_setup():
    """Test the GitHub Models setup and basic functionality"""
    print("🧪 Testing GitHub Models LLM Judge Setup")
    print("=" * 50)
    
    # Check if token is set
    token = os.getenv('GITHUB_TOKEN')
    if not token:
        print("❌ GITHUB_TOKEN not set")
        print("💡 Set it with: export GITHUB_TOKEN='your-token-here'")
        return False
    
    print(f"✅ GITHUB_TOKEN is set (length: {len(token)})")
    
    # Test Azure AI Inference SDK import
    try:
        from azure.ai.inference import ChatCompletionsClient
        from azure.core.credentials import AzureKeyCredential
        from azure.ai.inference.models import SystemMessage, UserMessage
        print("✅ Azure AI Inference SDK imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Azure AI Inference SDK: {e}")
        print("💡 Install with: pip install azure-ai-inference")
        return False
    
    # Test basic API call
    try:
        print("🔄 Testing basic API call...")
        
        endpoint = "https://models.github.ai/inference"
        model = "microsoft/phi-3.5-mini"  # Use a low-tier model for testing
        
        client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(token),
        )
        
        response = client.complete(
            messages=[
                SystemMessage("You are a helpful assistant."),
                UserMessage("Say 'Hello from GitHub Models!' and nothing else."),
            ],
            temperature=0.1,
            max_tokens=50,
            model=model
        )
        
        content = response.choices[0].message.content.strip()
        print(f"✅ API call successful: {content}")
        
    except Exception as e:
        print(f"❌ API call failed: {e}")
        return False
    
    return True

def test_evaluator_import():
    """Test importing the GitHub Models evaluator"""
    print("\n📦 Testing Evaluator Import")
    print("-" * 30)
    
    try:
        from evaluators.check_llm_judge_github import GitHubModelsEvaluator
        print("✅ GitHubModelsEvaluator imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import GitHubModelsEvaluator: {e}")
        return False

def test_quick_evaluation():
    """Test a quick evaluation"""
    print("\n⚡ Testing Quick Evaluation")
    print("-" * 30)
    
    try:
        # Run the evaluator in quick mode
        result = subprocess.run([
            sys.executable, 
            'evaluators/check_llm_judge_github.py',
            '--quick',
            '--model', 'microsoft/phi-3.5-mini'  # Use a low-tier model
        ], capture_output=True, text=True, timeout=120)
        
        print(f"Exit code: {result.returncode}")
        print(f"Stdout: {result.stdout}")
        if result.stderr:
            print(f"Stderr: {result.stderr}")
        
        if result.returncode == 0:
            print("✅ Quick evaluation completed successfully")
            return True
        else:
            print("❌ Quick evaluation failed")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Evaluation timed out")
        return False
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return False

def test_model_selection():
    """Test different model options"""
    print("\n🤖 Testing Model Selection")
    print("-" * 30)
    
    models_to_test = [
        "microsoft/phi-3.5-mini",  # Low tier, fast
        "microsoft/phi-3.5",       # Low tier, good quality
        "deepseek/deepseek-coder-6.7b-instruct"   # High tier, excellent quality
    ]
    
    for model in models_to_test:
        print(f"🔄 Testing model: {model}")
        try:
            result = subprocess.run([
                sys.executable, 
                'evaluators/check_llm_judge_github.py',
                '--quick',
                '--model', model
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print(f"✅ {model} - Success")
            else:
                print(f"❌ {model} - Failed")
                print(f"   Error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"❌ {model} - Timeout")
        except Exception as e:
            print(f"❌ {model} - Error: {e}")

def main():
    """Main test function"""
    print("🚀 GitHub Models LLM Judge Test Suite")
    print("=" * 50)
    
    # Test setup
    if not test_github_models_setup():
        print("\n❌ Setup test failed. Please check your configuration.")
        return 1
    
    # Test evaluator import
    if not test_evaluator_import():
        print("\n❌ Import test failed. Please check the evaluator code.")
        return 1
    
    # Test quick evaluation
    if not test_quick_evaluation():
        print("\n❌ Quick evaluation test failed.")
        return 1
    
    # Test model selection
    test_model_selection()
    
    print("\n✅ All tests completed!")
    print("\n💡 Next steps:")
    print("   1. Update your GitHub Actions workflow to use GitHub Models")
    print("   2. Set GITHUB_TOKEN as a repository secret")
    print("   3. Configure the model and threshold as needed")
    
    return 0

if __name__ == '__main__':
    # Use environment variable instead of hardcoded token
    if not os.getenv('GITHUB_TOKEN'):
        print("❌ GITHUB_TOKEN environment variable not set")
        print("💡 Set it with: export GITHUB_TOKEN='your-token-here'")
        sys.exit(1)
    
    sys.exit(main()) 
