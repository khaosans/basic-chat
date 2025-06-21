#!/usr/bin/env python3
"""
Test script to verify app functionality
"""

import sys
import os
import tempfile
from io import BytesIO

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_document_processor():
    """Test the DocumentProcessor functionality"""
    print("🧪 Testing DocumentProcessor...")
    
    try:
        from app import DocumentProcessor
        
        # Create a test document processor
        doc_processor = DocumentProcessor()
        
        # Test initialization
        assert hasattr(doc_processor, 'get_uploaded_documents')
        assert hasattr(doc_processor, 'process_file')
        assert hasattr(doc_processor, 'get_relevant_context')
        print("✅ DocumentProcessor initialization successful")
        
        # Test document list
        documents = doc_processor.get_uploaded_documents()
        assert isinstance(documents, list)
        print("✅ Document list retrieval successful")
        
        # Test context search
        context = doc_processor.get_relevant_context("test", k=1)
        assert isinstance(context, str)
        print("✅ Context search successful")
        
        print("✅ All DocumentProcessor tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ DocumentProcessor test failed: {e}")
        return False

def test_tool_registry():
    """Test the ToolRegistry functionality"""
    print("🧪 Testing ToolRegistry...")
    
    try:
        from app import ToolRegistry, DocumentProcessor
        
        # Create a document processor
        doc_processor = DocumentProcessor()
        
        # Create tool registry
        tool_registry = ToolRegistry(doc_processor)
        
        # Test tool registration
        assert hasattr(tool_registry, 'tools')
        assert len(tool_registry.tools) > 0
        print("✅ Tool registry initialization successful")
        
        # Test tool detection
        tool = tool_registry.get_tool("what is the current time")
        assert tool is not None
        print("✅ Tool detection successful")
        
        print("✅ All ToolRegistry tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ ToolRegistry test failed: {e}")
        return False

def test_ollama_chat():
    """Test the OllamaChat functionality"""
    print("🧪 Testing OllamaChat...")
    
    try:
        from app import OllamaChat
        
        # Create chat instance
        chat = OllamaChat("mistral")
        
        # Test initialization
        assert hasattr(chat, 'model_name')
        assert hasattr(chat, 'query')
        print("✅ OllamaChat initialization successful")
        
        print("✅ All OllamaChat tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ OllamaChat test failed: {e}")
        return False

def test_reasoning_engine():
    """Test the reasoning engine imports"""
    print("🧪 Testing Reasoning Engine...")
    
    try:
        from reasoning_engine import ReasoningAgent, ReasoningChain, MultiStepReasoning, ReasoningResult
        
        print("✅ Reasoning engine imports successful")
        
        # Test ReasoningResult
        result = ReasoningResult(
            content="Test content",
            reasoning_steps=["Step 1", "Step 2"],
            confidence=0.8,
            sources=["test"]
        )
        assert result.content == "Test content"
        print("✅ ReasoningResult creation successful")
        
        print("✅ All reasoning engine tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Reasoning engine test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting app functionality tests...\n")
    
    tests = [
        test_document_processor,
        test_tool_registry,
        test_ollama_chat,
        test_reasoning_engine
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The app should work correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 