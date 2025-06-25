#!/usr/bin/env python3
"""
Task management tests for BasicChat application.

These tests verify the task management functionality.
"""

import pytest
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import time

# Add the parent directory to the path so we can import from app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from task_manager import TaskManager, TaskStatus
from task_ui import (
    display_task_status, 
    create_task_message, 
    display_task_result,
    display_task_metrics,
    display_active_tasks,
    should_use_background_task,
    create_deep_research_message,
    is_long_running_query
)
from config import config


@pytest.mark.unit
@pytest.mark.fast
class TestTaskManager:
    """Test TaskManager functionality"""

    def test_task_manager_initialization(self):
        """Test TaskManager initialization"""
        manager = TaskManager()
        assert manager is not None
        assert hasattr(manager, 'active_tasks')
        assert hasattr(manager, 'completed_tasks')

    def test_submit_task(self):
        """Test task submission"""
        manager = TaskManager()
        task_id = manager.submit_task("test_task", param1="value1")
        
        assert task_id is not None
        assert task_id in manager.active_tasks
        task_status = manager.active_tasks[task_id]
        assert task_status.status in ["pending", "running", "failed"]
        assert task_status.metadata["task_type"] == "test_task"
        assert task_status.metadata["param1"] == "value1"

    def test_get_task_status(self):
        """Test getting task status"""
        manager = TaskManager()
        task_id = manager.submit_task("test_task")
        
        task_status = manager.get_task_status(task_id)
        assert task_status is not None
        assert task_status.task_id == task_id
        assert task_status.status in ["pending", "running", "failed"]

    def test_get_nonexistent_task(self):
        """Test getting non-existent task"""
        manager = TaskManager()
        task_status = manager.get_task_status("nonexistent_id")
        assert task_status is None

    def test_cancel_task(self):
        """Test cancelling a task"""
        # Patch Celery to simulate fallback mode
        with patch('task_manager.Celery', side_effect=Exception("Celery not available")):
            manager = TaskManager()
            task_id = manager.submit_task("reasoning", query="test query", mode="Standard")
            # Give it a moment to start
            time.sleep(0.1)
            result = manager.cancel_task(task_id)
            assert result is True
            task_status = manager.get_task_status(task_id)
            assert task_status.status == "cancelled"

    def test_get_active_tasks(self):
        """Test getting active tasks"""
        manager = TaskManager()
        task_id1 = manager.submit_task("task1")
        task_id2 = manager.submit_task("task2")
        
        active_tasks = manager.get_active_tasks()
        assert len(active_tasks) >= 2
        task_ids = [task.task_id for task in active_tasks]
        assert task_id1 in task_ids
        assert task_id2 in task_ids


@pytest.mark.unit
@pytest.mark.fast
class TestTaskUI:
    """Test TaskUI functionality"""

    def test_create_task_message(self):
        """Test task message creation"""
        message = create_task_message("test-123", "test_type")
        assert message is not None
        assert message["role"] == "assistant"
        assert "test_type" in message["content"]
        assert "test-123" in message["content"]
        assert message["is_task"] is True

    def test_should_use_background_task(self):
        """Test background task decision logic"""
        # Test simple queries (should not use background)
        assert not should_use_background_task("Hello", "Standard", config)
        assert not should_use_background_task("What is 2+2?", "Standard", config)
        
        # Test complex queries (should use background)
        assert should_use_background_task("Analyze this document in detail", "Standard", config)
        assert should_use_background_task("Research the latest AI developments", "Standard", config)

    def test_is_long_running_query(self):
        """Test long running query detection"""
        # Test simple queries
        assert not is_long_running_query("Hello", "Standard")
        assert not is_long_running_query("What is 2+2?", "Standard")
        
        # Test complex queries
        assert is_long_running_query("Analyze this document in detail", "Standard")
        assert is_long_running_query("Research the latest AI developments", "Standard")
        
        # Test complex reasoning modes
        assert is_long_running_query("Simple question", "Multi-Step")
        assert is_long_running_query("Simple question", "Agent-Based")
        
        # Test long queries
        long_query = "This is a very long query that contains many words and should be considered as a long running task because it has more than twenty words in it"
        assert is_long_running_query(long_query, "Standard")

    def test_create_deep_research_message(self):
        """Test deep research message creation"""
        message = create_deep_research_message("test-123", "Research topic")
        assert message is not None
        assert message["role"] == "assistant"
        assert "Research topic" in message["content"]
        assert "deep research" in message["content"].lower()
        assert message["is_task"] is True


@pytest.mark.unit
@pytest.mark.fast
class TestTaskStatus:
    """Test TaskStatus dataclass"""

    def test_task_status_creation(self):
        """Test TaskStatus creation"""
        task_status = TaskStatus(
            task_id="test-123",
            status="pending",
            progress=0.0
        )
        
        assert task_status.task_id == "test-123"
        assert task_status.status == "pending"
        assert task_status.progress == 0.0
        assert task_status.result is None
        assert task_status.error is None

    def test_task_status_to_dict(self):
        """Test TaskStatus to_dict conversion"""
        task_status = TaskStatus(
            task_id="test-123",
            status="completed",
            progress=1.0,
            result={"test": "data"},
            metadata={"type": "test"}
        )
        
        data = task_status.to_dict()
        assert data["task_id"] == "test-123"
        assert data["status"] == "completed"
        assert data["progress"] == 1.0
        assert data["result"] == {"test": "data"}
        assert data["metadata"]["type"] == "test"

    def test_task_status_from_dict(self):
        """Test TaskStatus from_dict creation"""
        data = {
            "task_id": "test-456",
            "status": "running",
            "progress": 0.5,
            "result": None,
            "error": None,
            "created_at": 1234567890.0,
            "updated_at": 1234567890.0,
            "metadata": {"type": "test"},
            "celery_task_id": None
        }
        
        task_status = TaskStatus.from_dict(data)
        assert task_status.task_id == "test-456"
        assert task_status.status == "running"
        assert task_status.progress == 0.5
        assert task_status.metadata["type"] == "test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 