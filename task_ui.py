"""
Task UI components for BasicChat long-running tasks
"""

import streamlit as st
import time
from typing import Optional, Dict, Any
from task_manager import TaskManager, TaskStatus

def display_task_status(task_id: str, task_manager: TaskManager):
    """Display task status with progress bar and controls"""
    task_status = task_manager.get_task_status(task_id)
    
    if not task_status:
        st.error("Task not found")
        return
    
    # Create a container for the task
    with st.container():
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            # Task title and status
            status_emoji = {
                "pending": "⏳",
                "running": "🔄", 
                "completed": "✅",
                "failed": "❌",
                "cancelled": "🚫"
            }.get(task_status.status, "❓")
            
            task_type = task_status.metadata.get('task_type', 'Task').title()
            st.markdown(f"**{status_emoji} {task_type}**")
            
            # Progress bar
            if task_status.status in ["pending", "running"]:
                progress_bar = st.progress(task_status.progress)
                
                # Estimated time remaining
                if task_status.progress > 0:
                    elapsed = time.time() - task_status.created_at
                    estimated_total = elapsed / task_status.progress
                    remaining = estimated_total - elapsed
                    st.caption(f"⏱️ Estimated {remaining:.0f}s remaining")
            
            # Status message
            if task_status.status == "running":
                status_msg = task_status.metadata.get('status', 'Processing...')
                st.info(f"🔄 {status_msg}")
            elif task_status.status == "completed":
                st.success("✅ Task completed successfully!")
            elif task_status.status == "failed":
                st.error(f"❌ Task failed: {task_status.error}")
            elif task_status.status == "cancelled":
                st.warning("🚫 Task was cancelled")
        
        with col2:
            # Cancel button for running tasks
            if task_status.status in ["pending", "running"]:
                if st.button("❌ Cancel", key=f"cancel_{task_id}"):
                    if task_manager.cancel_task(task_id):
                        st.success("Task cancelled successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to cancel task")
        
        with col3:
            # Refresh button
            if st.button("🔄", key=f"refresh_{task_id}", help="Refresh task status"):
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
    """Display active tasks in the sidebar"""
    active_tasks = task_manager.get_active_tasks()
    
    if active_tasks:
        st.sidebar.header("🔄 Active Tasks")
        
        for task in active_tasks:
            with st.sidebar.expander(f"{task.metadata.get('task_type', 'Task')} ({task.task_id[:8]}...)", expanded=False):
                display_task_status(task.task_id, task_manager)
    else:
        st.sidebar.header("🔄 Active Tasks")
        st.sidebar.info("No active tasks")

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