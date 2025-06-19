#!/usr/bin/env python3
"""
Test script for reasoning engine components
"""

import os
import sys
from reasoning_engine import ReasoningChain, ReasoningAgent, MultiStepReasoning, ReasoningDocumentProcessor

def print_banner(msg, passed=True):
    banner = "\n" + ("✅ PASS" if passed else "❌ FAIL") + f" | {msg}"
    print("\n" + ("=" * 60))
    print(banner)
    print("=" * 60 + "\n")

def test_reasoning_chain():
    """Test chain-of-thought reasoning"""
    print("🧠 Testing Chain-of-Thought Reasoning...")
    try:
        chain = ReasoningChain()
        result = chain.execute_reasoning("What is 2 + 2?")
        print(f"Result: {result.content[:100]}...")
        print(f"   Confidence: {result.confidence}")
        print(f"   Success: {result.success}")
        print_banner("Chain-of-Thought Reasoning", result.success)
        return result.success
    except Exception as e:
        print(f"❌ Chain-of-Thought Error: {e}")
        print_banner("Chain-of-Thought Reasoning", False)
        return False

def test_reasoning_agent():
    """Test agent-based reasoning"""
    print("\n🤖 Testing Agent-Based Reasoning...")
    try:
        agent = ReasoningAgent()
        result = agent.run("What is the current time?")
        print(f"Result: {result.content[:100]}...")
        print(f"   Confidence: {result.confidence}")
        print(f"   Success: {result.success}")
        print_banner("Agent-Based Reasoning", result.success)
        return result.success
    except Exception as e:
        print(f"❌ Agent Error: {e}")
        print_banner("Agent-Based Reasoning", False)
        return False

def test_multi_step_reasoning():
    """Test multi-step reasoning"""
    print("\n📝 Testing Multi-Step Reasoning...")
    try:
        # Create a mock document processor
        class MockDocProcessor:
            def get_relevant_context(self, query, k=3):
                return "Mock context for testing"
        
        multi_step = MultiStepReasoning(MockDocProcessor())
        result = multi_step.step_by_step_reasoning("Explain how photosynthesis works")
        print(f"Result: {result.content[:100]}...")
        print(f"   Confidence: {result.confidence}")
        print(f"   Success: {result.success}")
        print_banner("Multi-Step Reasoning", result.success)
        return result.success
    except Exception as e:
        print(f"❌ Multi-Step Error: {e}")
        print_banner("Multi-Step Reasoning", False)
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Reasoning Engine Components\n")
    
    tests = [
        test_reasoning_chain,
        test_reasoning_agent,
        test_multi_step_reasoning
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + ("#" * 60))
    if passed == total:
        print(f"🎉 ALL TESTS PASSED: {passed}/{total} | Reasoning engine is working correctly.")
    else:
        print(f"⚠️  SOME TESTS FAILED: {passed}/{total} | Please check the errors above.")
    print("#" * 60 + "\n")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main()) 