"""
Task UI components for BasicChat long-running tasks
"""

import streamlit as st
import time
from typing import Optional, Dict, Any
from task_manager import TaskManager, TaskStatus

def display_task_status(task_id: str, task_manager: TaskManager, context: str = "default"):
    """
    Display task status with controls.
    
    Args:
        task_id: The task ID
        task_manager: The task manager instance
        context: Context to make keys unique (e.g., "message_loop", "new_task")
    """
    task_status = task_manager.get_task_status(task_id)
    if not task_status:
        st.warning("Task not found")
        return
    
    # Create unique keys based on context
    cancel_key = f"cancel_{task_id}_{context}"
    refresh_key = f"refresh_{task_id}_{context}"
    
    # Display status with emoji
    status_emoji = {
        "pending": "⏳",
        "running": "🔄", 
        "completed": "✅",
        "failed": "❌",
        "cancelled": "🚫"
    }.get(task_status.status, "❓")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"{status_emoji} **{task_status.status.title()}**")
        
        # Show progress for running tasks
        if task_status.status == "running":
            if hasattr(task_status, 'progress') and task_status.progress:
                st.progress(task_status.progress)
            else:
                st.progress(0.5)  # Indeterminate progress
        
        # Show status messages
        if task_status.status == "pending":
            st.info("⏳ Task is queued and waiting to start")
        elif task_status.status == "running":
            # Show more detailed status for running tasks
            status_msg = task_status.metadata.get('status', 'Running')
            st.info(f"🔄 Task is currently running... ({status_msg})")
            
            # Show progress percentage
            if hasattr(task_status, 'progress') and task_status.progress:
                progress_pct = int(task_status.progress * 100)
                st.caption(f"Progress: {progress_pct}%")
            
            # Show last update time
            if hasattr(task_status, 'updated_at') and task_status.updated_at:
                from datetime import datetime
                last_update = datetime.fromtimestamp(task_status.updated_at).strftime('%H:%M:%S')
                st.caption(f"Last update: {last_update}")
        elif task_status.status == "completed":
            st.success("✅ Task completed successfully!")
            
            # Automatically display results for completed tasks
            if task_status.result:
                st.markdown("### 📋 Results")
                display_task_result(task_status)
        elif task_status.status == "failed":
            st.error(f"❌ Task failed: {task_status.error}")
            
            # Show error details if available
            if hasattr(task_status, 'traceback') and task_status.traceback:
                with st.expander("🔍 Error Details", expanded=False):
                    st.code(task_status.traceback)
        elif task_status.status == "cancelled":
            st.warning("🚫 Task was cancelled")
    
    with col2:
        # Cancel button for running tasks
        if task_status.status in ["pending", "running"]:
            if st.button("❌ Cancel", key=cancel_key):
                if task_manager.cancel_task(task_id):
                    st.success("Task cancelled successfully!")
                    st.rerun()
                else:
                    st.error("Failed to cancel task")
    
    with col3:
        # Refresh button
        if st.button("🔄", key=refresh_key, help="Refresh task status"):
            st.rerun()

def create_task_message(task_id: str, task_type: str, **kwargs) -> Dict[str, Any]:
    """Create a special message for long-running tasks"""
    return {
        "role": "assistant",
        "content": f"🚀 **Long-running task started**\n\n**Type:** {task_type}\n**Task ID:** `{task_id}`\n\nThis task is running in the background. You can continue chatting while it processes.",
        "task_id": task_id,
        "is_task": True,
        "metadata": kwargs
    }

def display_task_result(task_status: TaskStatus):
    """Display task result in a formatted way"""
    if not task_status.result:
        st.info("No result available")
        return
    
    result = task_status.result
    
    # Display based on task type
    task_type = task_status.metadata.get('task_type', 'unknown')
    
    if task_type == "reasoning":
        display_reasoning_result(result)
    elif task_type == "deep_research":
        display_deep_research_result(result)
    elif task_type in ["document_analysis", "document_processing"]:
        display_document_result(result)
    else:
        # Generic result display
        st.json(result)

def display_reasoning_result(result: Dict[str, Any]):
    """Display reasoning task result"""
    if not result:
        st.info("No reasoning result available")
        return
    
    # Display final answer
    if 'final_answer' in result:
        st.markdown("### 📝 Final Answer")
        st.markdown(result['final_answer'])
    
    # Display thought process
    if 'thought_process' in result and result['thought_process']:
        with st.expander("💭 Thought Process", expanded=False):
            st.markdown(result['thought_process'])
    
    # Display reasoning steps
    if 'reasoning_steps' in result and result['reasoning_steps']:
        with st.expander("🔍 Reasoning Steps", expanded=False):
            for i, step in enumerate(result['reasoning_steps'], 1):
                st.markdown(f"**Step {i}:** {step}")
    
    # Display metadata
    col1, col2, col3 = st.columns(3)
    with col1:
        if 'confidence' in result:
            confidence = result['confidence']
            if confidence >= 0.8:
                st.metric("Confidence", f"{confidence:.1%}", delta="High")
            elif confidence >= 0.6:
                st.metric("Confidence", f"{confidence:.1%}", delta="Medium")
            else:
                st.metric("Confidence", f"{confidence:.1%}", delta="Low")
    
    with col2:
        if 'execution_time' in result:
            st.metric("Execution Time", f"{result['execution_time']:.2f}s")
    
    with col3:
        if 'reasoning_mode' in result:
            st.metric("Mode", result['reasoning_mode'])

def display_deep_research_result(result: Dict[str, Any]):
    """Display deep research task result with rich formatting"""
    if not result:
        st.info("No research result available")
        return
    
    # Research query
    st.markdown(f"### 🔍 Research Query")
    st.info(result.get('research_query', 'No query specified'))
    
    # Executive Summary
    if result.get('executive_summary'):
        st.markdown("### 📋 Executive Summary")
        st.markdown(result['executive_summary'])
    
    # Key Findings
    if result.get('key_findings'):
        st.markdown("### 🎯 Key Findings")
        st.markdown(result['key_findings'])
    
    # Detailed Analysis
    if result.get('detailed_analysis'):
        with st.expander("📊 Detailed Analysis", expanded=True):
            st.markdown(result['detailed_analysis'])
    
    # Sources
    if result.get('sources'):
        with st.expander("📚 Sources & Citations", expanded=False):
            st.markdown("### Sources Used:")
            for i, source in enumerate(result['sources'][:5], 1):
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.write(f"**{i}.**")
                with col2:
                    if source.get('title'):
                        st.markdown(f"**{source['title']}**")
                    if source.get('url'):
                        st.markdown(f"🔗 [{source['url']}]({source['url']})")
                    if source.get('snippet'):
                        st.caption(source['snippet'])
                st.markdown("---")
    
    # Recommendations
    if result.get('recommendations'):
        with st.expander("💡 Recommendations", expanded=False):
            st.markdown(result['recommendations'])
    
    # Further Research
    if result.get('further_research'):
        with st.expander("🔬 Areas for Further Research", expanded=False):
            st.markdown(result['further_research'])
    
    # Research Metadata
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if 'sources_analyzed' in result:
            st.metric("Sources", result['sources_analyzed'])
    with col2:
        if 'confidence' in result:
            confidence = result['confidence']
            if confidence >= 0.8:
                st.metric("Confidence", f"{confidence:.1%}", delta="High")
            elif confidence >= 0.6:
                st.metric("Confidence", f"{confidence:.1%}", delta="Medium")
            else:
                st.metric("Confidence", f"{confidence:.1%}", delta="Low")
    with col3:
        if 'execution_time' in result:
            st.metric("Research Time", f"{result['execution_time']:.1f}s")
    with col4:
        if 'search_terms_used' in result:
            st.metric("Search Terms", len(result['search_terms_used']))

def display_document_result(result: Dict[str, Any]):
    """Display document processing result"""
    if not result:
        st.info("No document result available")
        return
    
    # Display file information
    if 'file_path' in result:
        st.markdown(f"**📄 File:** {result['file_path']}")
    
    if 'file_type' in result:
        st.markdown(f"**📋 Type:** {result['file_type']}")
    
    # Display processing status
    if 'processing_complete' in result and result['processing_complete']:
        st.success("✅ Document processing completed")
    elif 'analysis_complete' in result and result['analysis_complete']:
        st.success("✅ Document analysis completed")
    
    # Display processed files
    if 'processed_files' in result and result['processed_files']:
        st.markdown("### 📚 Processed Files")
        for file_data in result['processed_files']:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"• {file_data.get('name', 'Unknown')}")
            with col2:
                st.write(f"({file_data.get('type', 'unknown')})")
    
    # Display processing details
    if 'vectorized' in result and result['vectorized']:
        st.info("🔢 Document vectorized for semantic search")
    
    if 'indexed' in result and result['indexed']:
        st.info("📇 Document indexed for fast retrieval")
    
    if 'searchable' in result and result['searchable']:
        st.info("🔍 Document ready for semantic search")

def display_task_metrics(task_manager: TaskManager):
    """Display task metrics in the sidebar"""
    metrics = task_manager.get_task_metrics()
    
    st.sidebar.header("📊 Task Metrics")
    
    # Status counts
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Active", metrics['status_counts']['running'] + metrics['status_counts']['pending'])
        st.metric("Completed", metrics['status_counts']['completed'])
    with col2:
        st.metric("Failed", metrics['status_counts']['failed'])
        st.metric("Cancelled", metrics['status_counts']['cancelled'])
    
    # Additional metrics
    st.sidebar.markdown("---")
    st.sidebar.metric("Total Tasks", metrics['total_tasks'])
    
    if metrics['avg_completion_time'] > 0:
        st.sidebar.metric("Avg Time", f"{metrics['avg_completion_time']:.1f}s")
    
    # Celery status
    if metrics['celery_available']:
        st.sidebar.success("🟢 Celery Available")
    else:
        st.sidebar.warning("🟡 Celery Unavailable (using fallback)")

def display_active_tasks(task_manager: TaskManager):
    """Display active tasks in sidebar"""
    active_tasks = task_manager.get_active_tasks()
    
    if not active_tasks:
        st.sidebar.info("No active tasks")
        return
    
    st.sidebar.subheader("🔄 Active Tasks")
    
    for task in active_tasks:
        # Create a more informative expander title
        status_emoji = {
            "pending": "⏳",
            "running": "🔄", 
            "completed": "✅",
            "failed": "❌",
            "cancelled": "🚫"
        }.get(task.status, "❓")
        
        task_type = task.metadata.get('task_type', 'Task')
        short_id = task.task_id[:8]
        
        # Show progress in title for running tasks
        if task.status == "running" and hasattr(task, 'progress') and task.progress:
            progress_pct = int(task.progress * 100)
            title = f"{status_emoji} {task_type} ({short_id}...) - {progress_pct}%"
        else:
            title = f"{status_emoji} {task_type} ({short_id}...)"
        
        with st.sidebar.expander(title, expanded=False):
            display_task_status(task.task_id, task_manager, "sidebar")

def is_long_running_query(query: str, reasoning_mode: str) -> bool:
    """Determine if a query should be processed as a long-running task"""
    # Complex queries that might take longer
    complex_keywords = [
        "analyze", "comprehensive", "detailed", "research", "investigate",
        "compare", "evaluate", "assess", "examine", "study", "explain",
        "break down", "step by step", "thorough", "in-depth"
    ]
    
    # Long queries
    if len(query.split()) > 20:
        return True
    
    # Complex reasoning modes
    if reasoning_mode in ["Multi-Step", "Agent-Based"]:
        return True
    
    # Contains complex keywords
    if any(keyword in query.lower() for keyword in complex_keywords):
        return True
    
    # Queries that explicitly request long processing
    if any(phrase in query.lower() for phrase in ["take your time", "detailed analysis", "comprehensive answer"]):
        return True
    
    return False

def should_use_background_task(query: str, reasoning_mode: str, config) -> bool:
    """Determine if background task processing should be used"""
    # Check if background tasks are enabled
    if not config.enable_background_tasks:
        return False
    
    # Check if it's a long-running query
    return is_long_running_query(query, reasoning_mode)

def create_deep_research_message(task_id: str, query: str) -> Dict[str, Any]:
    """Create a special message for deep research tasks"""
    return {
        "role": "assistant",
        "content": f"🔬 **Deep Research Started**\n\n**Query:** {query}\n**Task ID:** `{task_id}`\n\nI'm conducting comprehensive research on your query. This includes:\n• Multiple source searches\n• Content analysis and synthesis\n• Detailed findings and recommendations\n• Source citations\n\nThis may take a few minutes. You can continue chatting while I research!",
        "task_id": task_id,
        "is_task": True,
        "is_deep_research": True,
        "metadata": {"query": query}
    } 
