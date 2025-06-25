#!/usr/bin/env python3
"""
Simple test for BasicChat Deep Research functionality
"""

import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_web_search():
    """Test web search functionality"""
    print("🌐 Testing Web Search Functionality")
    print("=" * 40)
    
    try:
        from web_search import WebSearch
        
        web_search = WebSearch()
        query = "quantum computing 2024"
        
        print(f"🔍 Searching for: {query}")
        results = web_search.search(query, max_results=2)
        
        print(f"✅ Found {len(results)} results")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result.title}")
            print(f"   URL: {result.link}")
            print(f"   Snippet: {result.snippet[:80]}...")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Web search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_task_manager():
    """Test task manager functionality"""
    print("📋 Testing Task Manager")
    print("=" * 30)
    
    try:
        from task_manager import TaskManager
        
        task_manager = TaskManager()
        print("✅ Task manager initialized successfully")
        
        # Test submitting a simple task
        task_id = task_manager.submit_task(
            "deep_research",
            query="test query",
            research_depth="quick"
        )
        
        print(f"✅ Task submitted with ID: {task_id}")
        
        # Check task status
        task_status = task_manager.get_task_status(task_id)
        if task_status:
            print(f"✅ Task status retrieved: {task_status.status}")
        else:
            print("❌ Could not retrieve task status")
        
        return True
        
    except Exception as e:
        print(f"❌ Task manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ollama_connection():
    """Test Ollama connection"""
    print("🤖 Testing Ollama Connection")
    print("=" * 30)
    
    try:
        import requests
        
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print(f"✅ Ollama is running with {len(models.get('models', []))} models")
            for model in models.get('models', [])[:3]:
                print(f"   - {model.get('name', 'Unknown')}")
            return True
        else:
            print(f"❌ Ollama returned status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 BasicChat Quick Test Suite")
    print("=" * 40)
    print()
    
    # Test Ollama connection
    ollama_success = test_ollama_connection()
    print()
    
    # Test web search
    web_search_success = test_web_search()
    print()
    
    # Test task manager
    task_manager_success = test_task_manager()
    print()
    
    print("=" * 40)
    print("📊 Test Results Summary:")
    print(f"   Ollama Connection: {'✅ PASS' if ollama_success else '❌ FAIL'}")
    print(f"   Web Search: {'✅ PASS' if web_search_success else '❌ FAIL'}")
    print(f"   Task Manager: {'✅ PASS' if task_manager_success else '❌ FAIL'}")
    
    if ollama_success and web_search_success and task_manager_success:
        print("\n🎉 All basic tests passed! Deep research should work.")
        print("\n🔗 To test the full deep research functionality:")
        print("   1. Open http://localhost:8501 in your browser")
        print("   2. Enable 'Deep Research Mode' toggle")
        print("   3. Ask a complex question like:")
        print("      'What are the latest developments in quantum computing?'")
        print("   4. Monitor progress at http://localhost:5555")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
    
    print("\n🔗 Application URLs:")
    print("   Main App: http://localhost:8501")
    print("   Task Monitor: http://localhost:5555")

if __name__ == "__main__":
    main() 