#!/usr/bin/env python3
"""
Test script for BasicChat Deep Research functionality
"""

import requests
import json
import time
import sys
import os
from pathlib import Path
import pytest

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from task_manager import TaskManager

log_path = "logs/test_deep_research_full.log"
os.makedirs(os.path.dirname(log_path), exist_ok=True)

def log(msg):
    print(msg)
    sys.stdout.flush()
    with open(log_path, "a") as f:
        f.write(msg + "\n")

def celery_worker_available():
    # Try to connect to the default Celery worker ping endpoint (if exposed)
    # Or check if TaskManager.celery_available is True and a worker responds
    try:
        tm = TaskManager()
        if not getattr(tm, 'celery_available', False):
            return False
        # Try to submit a trivial task and see if it completes quickly
        tid = tm.submit_task("health_check")
        start = time.time()
        while time.time() - start < 10:
            status = tm.get_task_status(tid)
            if status and status.status == "completed":
                return True
            elif status and status.status == "failed":
                return False
            time.sleep(1)
        return False
    except Exception as e:
        log(f"[red][AI/CI] Celery worker check failed: {e}[/red]")
        return False

@pytest.mark.integration
@pytest.mark.long  # Mark as long-running for special handling
def test_deep_research():
    log("[yellow][Test Notice] Deep research test may take several minutes. Please be patient while the system completes a comprehensive research task. Long-running tasks are expected in this phase.[/yellow]")
    if not celery_worker_available():
        log("[red][Test Skipped] Celery worker not running or not available. Deep research test skipped. Please ensure all services are up (docker compose up) and try again.[/red]")
        pytest.skip("Celery worker not running or not available")
    log("[cyan]🔬 Testing BasicChat Deep Research Functionality[/cyan]")
    log("[cyan]=" * 50 + "[/cyan]")
    task_manager = TaskManager()
    research_query = "What are the latest developments in quantum computing and their implications for cryptography?"
    log(f"[blue]📝 Research Query: {research_query}[/blue]")
    log("")
    log("[green]🚀 Submitting deep research task...[/green]")
    task_id = task_manager.submit_task(
        "deep_research",
        query=research_query,
        research_depth="comprehensive"
    )
    log(f"[green]✅ Task submitted with ID: {task_id}[/green]")
    log("")
    log("[magenta]📊 Monitoring task progress...[/magenta]")
    log("[magenta]-" * 30 + "[/magenta]")
    max_wait_time = 300  # 5 minutes
    warn_wait_time = 30  # Warn if stuck for >30s
    start_time = time.time()
    last_status = None
    last_status_time = start_time
    while True:
        elapsed = time.time() - start_time
        status = task_manager.get_task_status(task_id)
        if status:
            log(f"[white]⏱️  Time elapsed: {elapsed:.1f}s[/white]")
            log(f"[white]📈 Status: {status.status}[/white]")
            if status.metadata:
                progress = status.metadata.get('progress', 0)
                status_msg = status.metadata.get('status', 'Unknown')
                log(f"[white]📊 Progress: {progress:.1%}[/white]")
                log(f"[white]💬 Status: {status_msg}[/white]")
            if status.status == "completed":
                log("[green]✅ Task completed:[/green]")
                log(f"[green]{status.result}[/green]")
                assert status.result is not None
                break
            elif status.status == "failed":
                log("[red]❌ Task failed![/red]")
                if status.error:
                    log(f"[red]Error: {status.error}[/red]")
                pytest.fail("Deep research task failed")
                break
            # Detect if stuck in same state
            if last_status == status.status:
                if time.time() - last_status_time > warn_wait_time:
                    log(f"[yellow]⚠️  Task stuck in '{status.status}' for over {warn_wait_time}s![/yellow]")
                    pytest.fail(f"Task stuck in '{status.status}' for over {warn_wait_time}s")
                    break
            else:
                last_status = status.status
                last_status_time = time.time()
        else:
            log("[red]❓ No status returned for task![/red]")
        if elapsed > max_wait_time:
            log("[red]⏰ Timeout reached. Task may still be running.[/red]")
            pytest.fail("Deep research task timed out")
            break
        time.sleep(5)
    log("[cyan]=" * 50 + "[/cyan]")
    log("[green]🎉 Deep research test finished! If this took a while, that's normal for comprehensive research tasks.[/green]")

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
