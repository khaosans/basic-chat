#!/usr/bin/env python3
"""
Test script for BasicChat Deep Research functionality
"""

import requests
import json
import time
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from task_manager import TaskManager

def test_deep_research():
    """Test the deep research functionality"""
    print("🔬 Testing BasicChat Deep Research Functionality")
    print("=" * 50)
    
    # Initialize task manager
    task_manager = TaskManager()
    
    # Test research query
    research_query = "What are the latest developments in quantum computing and their implications for cryptography?"
    
    print(f"📝 Research Query: {research_query}")
    print()
    
    # Submit deep research task
    print("🚀 Submitting deep research task...")
    task_id = task_manager.submit_task(
        "deep_research",
        query=research_query,
        research_depth="comprehensive"
    )
    
    print(f"✅ Task submitted with ID: {task_id}")
    print()
    
    # Monitor task progress
    print("📊 Monitoring task progress...")
    print("-" * 30)
    
    max_wait_time = 300  # 5 minutes
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        task_status = task_manager.get_task_status(task_id)
        
        if task_status:
            print(f"⏱️  Time elapsed: {time.time() - start_time:.1f}s")
            print(f"📈 Status: {task_status.status}")
            
            if task_status.metadata:
                progress = task_status.metadata.get('progress', 0)
                status_msg = task_status.metadata.get('status', 'Unknown')
                print(f"📊 Progress: {progress:.1%}")
                print(f"💬 Status: {status_msg}")
            
            if task_status.status == "completed":
                print("✅ Task completed successfully!")
                print()
                print("📋 Research Results:")
                print("=" * 30)
                
                if task_status.result:
                    result = task_status.result
                    
                    # Display executive summary
                    if result.get('executive_summary'):
                        print("📋 Executive Summary:")
                        print(result['executive_summary'])
                        print()
                    
                    # Display key findings
                    if result.get('key_findings'):
                        print("🎯 Key Findings:")
                        print(result['key_findings'])
                        print()
                    
                    # Display sources
                    if result.get('sources'):
                        print("📚 Sources:")
                        for i, source in enumerate(result['sources'][:3], 1):
                            print(f"{i}. {source.get('title', 'No title')}")
                            if source.get('url'):
                                print(f"   URL: {source['url']}")
                        print()
                    
                    # Display metadata
                    print("📊 Research Metadata:")
                    print(f"   Sources analyzed: {result.get('sources_analyzed', 'Unknown')}")
                    print(f"   Search terms used: {len(result.get('search_terms_used', []))}")
                    print(f"   Confidence: {result.get('confidence', 'Unknown')}")
                    print(f"   Execution time: {result.get('execution_time', 'Unknown')}s")
                
                return True
                
            elif task_status.status == "failed":
                print("❌ Task failed!")
                if task_status.error:
                    print(f"Error: {task_status.error}")
                return False
        
        print()
        time.sleep(10)  # Wait 10 seconds before next check
    
    print("⏰ Timeout reached. Task may still be running.")
    return False

def test_web_search():
    """Test web search functionality"""
    print("🌐 Testing Web Search Functionality")
    print("=" * 40)
    
    try:
        from web_search import WebSearch
        
        web_search = WebSearch()
        query = "quantum computing 2024"
        
        print(f"🔍 Searching for: {query}")
        results = web_search.search(query, max_results=3)
        
        print(f"✅ Found {len(results)} results")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result.title}")
            print(f"   URL: {result.link}")
            print(f"   Snippet: {result.snippet[:100]}...")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Web search test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 BasicChat Deep Research Test Suite")
    print("=" * 50)
    print()
    
    # Test web search first
    web_search_success = test_web_search()
    print()
    
    if web_search_success:
        # Test deep research
        deep_research_success = test_deep_research()
        
        print("=" * 50)
        print("📊 Test Results Summary:")
        print(f"   Web Search: {'✅ PASS' if web_search_success else '❌ FAIL'}")
        print(f"   Deep Research: {'✅ PASS' if deep_research_success else '❌ FAIL'}")
        
        if web_search_success and deep_research_success:
            print("\n🎉 All tests passed! Deep research functionality is working correctly.")
        else:
            print("\n⚠️  Some tests failed. Check the output above for details.")
    else:
        print("❌ Web search test failed. Cannot proceed with deep research test.")
    
    print("\n🔗 You can also test the web interface:")
    print("   Main App: http://localhost:8501")
    print("   Task Monitor: http://localhost:5555")

if __name__ == "__main__":
    main() 