#!/usr/bin/env python3
"""
Simple test for BasicChat Deep Research functionality
"""

import sys
import os
from pathlib import Path
import pytest
import requests

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

log_path = "logs/test_deep_research_simple.log"
os.makedirs(os.path.dirname(log_path), exist_ok=True)

def log(msg):
    print(msg)
    sys.stdout.flush()
    with open(log_path, "a") as f:
        f.write(msg + "\n")

def test_web_search():
    """Test web search functionality"""
    log("🌐 Testing Web Search Functionality")
    log("=" * 40)
    
    try:
        from web_search import WebSearch
        
        web_search = WebSearch()
        query = "quantum computing 2024"
        
        log(f"🔍 Searching for: {query}")
        results = web_search.search(query, max_results=2)
        
        log(f"✅ Found {len(results)} results")
        for i, result in enumerate(results, 1):
            log(f"{i}. {result.title}")
            log(f"   URL: {result.link}")
            log(f"   Snippet: {result.snippet[:80]}...")
        assert len(results) > 0, "No results returned from web search"
    except Exception as e:
        log(f"❌ Web search test failed: {e}")
        import traceback
        traceback.print_exc()
        assert False, f"Web search test failed: {e}"

def test_task_manager():
    """Test task manager functionality"""
    log("📋 Testing Task Manager")
    log("=" * 30)
    
    try:
        from task_manager import TaskManager
        
        task_manager = TaskManager()
        log("✅ Task manager initialized successfully")
        
        # Test submitting a simple task
        task_id = task_manager.submit_task(
            "deep_research",
            query="test query",
            research_depth="quick"
        )
        
        log(f"✅ Task submitted with ID: {task_id}")
        
        # Check task status
        task_status = task_manager.get_task_status(task_id)
        if task_status:
            log(f"✅ Task status retrieved: {task_status.status}")
        else:
            log("❌ Could not retrieve task status")
        assert task_status is not None, "Could not retrieve task status"
    except Exception as e:
        log(f"❌ Task manager test failed: {e}")
        import traceback
        traceback.print_exc()
        assert False, f"Task manager test failed: {e}"

def openai_available():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return False
    try:
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 5
            },
            timeout=5
        )
        return r.status_code == 200
    except Exception:
        return False

def huggingface_available():
    api_key = os.environ.get("HUGGINGFACE_API_KEY")
    if not api_key:
        return False
    try:
        r = requests.post(
            "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"inputs": "Hello!"},
            timeout=10
        )
        return r.status_code == 200
    except Exception:
        return False

@pytest.mark.integration
# AI/CI: This test is skipped if neither OpenAI nor Hugging Face API is available.
def test_llm_connectivity():
    if openai_available():
        log("🤖 Testing OpenAI API Connection")
        log("=" * 30)
        api_key = os.environ["OPENAI_API_KEY"]
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 5
            },
            timeout=10
        )
        assert r.status_code == 200, f"OpenAI API returned status code: {r.status_code}"
        log(f"✅ OpenAI API is reachable and responded with status {r.status_code}")
        data = r.json()
        log(f"   Model: {data.get('model', 'Unknown')}")
        log(f"   Choices: {len(data.get('choices', []))}")
    elif huggingface_available():
        log("🤖 Testing Hugging Face Inference API (Mistral-7B)")
        log("=" * 30)
        api_key = os.environ["HUGGINGFACE_API_KEY"]
        r = requests.post(
            "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"inputs": "Hello!"},
            timeout=20
        )
        assert r.status_code == 200, f"Hugging Face API returned status code: {r.status_code}"
        log(f"✅ Hugging Face API is reachable and responded with status {r.status_code}")
        data = r.json()
        log(f"   Output: {data}")
    else:
        log("[AI/CI] Skipping test_llm_connectivity: No OpenAI or Hugging Face API key available")
        pytest.skip("No OpenAI or Hugging Face API key available")

def main():
    log("")
    # Test OpenAI connection
    openai_success = test_llm_connectivity()
    log("")
    # Test web search
    web_search_success = test_web_search()
    log("")
    # Test task manager
    task_manager_success = test_task_manager()
    log("")
    log("=" * 40)
    log("📊 Test Results Summary:")
    log(f"   OpenAI Connection: {'✅ PASS' if openai_success else '❌ FAIL'}")
    log(f"   Web Search: {'✅ PASS' if web_search_success else '❌ FAIL'}")
    log(f"   Task Manager: {'✅ PASS' if task_manager_success else '❌ FAIL'}")
    if openai_success and web_search_success and task_manager_success:
        log("\n🎉 All basic tests passed! Deep research should work.")
        log("\n🔗 To test the full deep research functionality:")
        log("   1. Open http://localhost:8501 in your browser")
        log("   2. Enable 'Deep Research Mode' toggle")
        log("   3. Ask a complex question like:")
        log("      'What are the latest developments in quantum computing?'")
        log("   4. Monitor progress at http://localhost:5555")
    else:
        log("\n⚠️  Some tests failed. Check the output above for details.")
    log("\n🔗 Application URLs:")
    log("   Main App: http://localhost:8501")
    log("   Task Monitor: http://localhost:5555")

if __name__ == "__main__":
    main() 
