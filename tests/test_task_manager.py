"""
Unit tests for task management system
"""

import pytest
import time
import tempfile
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from task_manager import TaskManager, TaskStatus
from task_ui import (
    is_long_running_query,
    should_use_background_task,
    create_task_message,
    display_task_result
)
from config import config


class TestTaskStatus:
    """Test TaskStatus dataclass"""
    
    def test_task_status_creation(self):
        """Test creating a TaskStatus instance"""
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
        assert task_status.created_at > 0
        assert task_status.updated_at > 0
        assert isinstance(task_status.metadata, dict)
    
    def test_task_status_to_dict(self):
        """Test converting TaskStatus to dictionary"""
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
        """Test creating TaskStatus from dictionary"""
        data = {
            "task_id": "test-456",
            "status": "running",
            "progress": 0.5,
            "result": None,
            "error": None,
            "created_at": time.time(),
            "updated_at": time.time(),
            "metadata": {"type": "test"},
            "celery_task_id": None
        }
        
        task_status = TaskStatus.from_dict(data)
        
        assert task_status.task_id == "test-456"
        assert task_status.status == "running"
        assert task_status.progress == 0.5
        assert task_status.metadata["type"] == "test"


class TestTaskManager:
    """Test TaskManager class"""
    
    @patch('task_manager.Celery')
    def test_task_manager_initialization_with_celery(self, mock_celery):
        """Test TaskManager initialization with Celery available"""
        mock_celery_instance = Mock()
        mock_celery.return_value = mock_celery_instance
        
        task_manager = TaskManager()
        
        assert task_manager.celery_available is True
        assert task_manager.active_tasks == {}
        assert task_manager.completed_tasks == {}
        assert task_manager.max_completed_tasks == 100
    
    @patch('task_manager.Celery', side_effect=Exception("Celery not available"))
    def test_task_manager_initialization_without_celery(self, mock_celery):
        """Test TaskManager initialization without Celery"""
        task_manager = TaskManager()
        
        assert task_manager.celery_available is False
        assert task_manager.active_tasks == {}
        assert task_manager.completed_tasks == {}
    
    def test_submit_task_basic(self):
        """Test basic task submission"""
        with patch('task_manager.Celery', side_effect=Exception("Celery not available")):
            task_manager = TaskManager()
            
            task_id = task_manager.submit_task("test_task", param1="value1")
            
            assert task_id in task_manager.active_tasks
            task_status = task_manager.active_tasks[task_id]
            assert task_status.status == "running"
            assert task_status.metadata["task_type"] == "test_task"
            assert task_status.metadata["param1"] == "value1"
    
    def test_submit_task_with_celery(self):
        """Test task submission with Celery available"""
        mock_celery_task = Mock()
        mock_celery_task.id = "celery-task-123"
        
        with patch('task_manager.Celery') as mock_celery:
            mock_celery_instance = Mock()
            mock_celery_instance.send_task.return_value = mock_celery_task
            mock_celery.return_value = mock_celery_instance
            
            task_manager = TaskManager()
            
            task_id = task_manager.submit_task("reasoning", query="test query", mode="Standard")
            
            assert task_id in task_manager.active_tasks
            task_status = task_manager.active_tasks[task_id]
            assert task_status.status == "running"
            assert task_status.celery_task_id == "celery-task-123"
    
    def test_get_task_status(self):
        """Test getting task status"""
        with patch('task_manager.Celery', side_effect=Exception("Celery not available")):
            task_manager = TaskManager()
            
            # Submit a task
            task_id = task_manager.submit_task("test_task")
            
            # Get task status
            task_status = task_manager.get_task_status(task_id)
            
            assert task_status is not None
            assert task_status.task_id == task_id
            assert task_status.status == "running"
    
    def test_get_task_status_not_found(self):
        """Test getting task status for non-existent task"""
        with patch('task_manager.Celery', side_effect=Exception("Celery not available")):
            task_manager = TaskManager()
            
            task_status = task_manager.get_task_status("non-existent")
            
            assert task_status is None
    
    def test_cancel_task(self):
        """Test cancelling a task"""
        with patch('task_manager.Celery', side_effect=Exception("Celery not available")):
            task_manager = TaskManager()
            
            # Submit a task
            task_id = task_manager.submit_task("test_task")
            
            # Cancel the task
            result = task_manager.cancel_task(task_id)
            
            assert result is True
            task_status = task_manager.get_task_status(task_id)
            assert task_status.status == "cancelled"
    
    def test_cancel_task_not_found(self):
        """Test cancelling a non-existent task"""
        with patch('task_manager.Celery', side_effect=Exception("Celery not available")):
            task_manager = TaskManager()
            
            result = task_manager.cancel_task("non-existent")
            
            assert result is False
    
    def test_get_active_tasks(self):
        """Test getting active tasks"""
        with patch('task_manager.Celery', side_effect=Exception("Celery not available")):
            task_manager = TaskManager()
            
            # Submit multiple tasks
            task_id1 = task_manager.submit_task("task1")
            task_id2 = task_manager.submit_task("task2")
            
            active_tasks = task_manager.get_active_tasks()
            
            assert len(active_tasks) == 2
            task_ids = [task.task_id for task in active_tasks]
            assert task_id1 in task_ids
            assert task_id2 in task_ids
    
    def test_get_completed_tasks(self):
        """Test getting completed tasks"""
        with patch('task_manager.Celery', side_effect=Exception("Celery not available")):
            task_manager = TaskManager()
            
            # Submit and complete a task
            task_id = task_manager.submit_task("test_task")
            task_status = task_manager.active_tasks[task_id]
            task_status.status = "completed"
            task_manager._move_to_completed(task_id)
            
            completed_tasks = task_manager.get_completed_tasks()
            
            assert len(completed_tasks) == 1
            assert completed_tasks[0].task_id == task_id
            assert completed_tasks[0].status == "completed"
    
    def test_cleanup_old_tasks(self):
        """Test cleaning up old tasks"""
        with patch('task_manager.Celery', side_effect=Exception("Celery not available")):
            task_manager = TaskManager()
            
            # Submit and complete a task
            task_id = task_manager.submit_task("test_task")
            task_status = task_manager.active_tasks[task_id]
            task_status.status = "completed"
            task_status.updated_at = time.time() - 25 * 3600  # 25 hours ago
            task_manager._move_to_completed(task_id)
            
            # Clean up tasks older than 24 hours
            task_manager.cleanup_old_tasks(max_age_hours=24)
            
            completed_tasks = task_manager.get_completed_tasks()
            assert len(completed_tasks) == 0
    
    def test_get_task_metrics(self):
        """Test getting task metrics"""
        with patch('task_manager.Celery', side_effect=Exception("Celery not available")):
            task_manager = TaskManager()
            
            # Submit tasks in different states
            task_id1 = task_manager.submit_task("task1")
            task_id2 = task_manager.submit_task("task2")
            
            # Mark one as completed
            task_status = task_manager.active_tasks[task_id1]
            task_status.status = "completed"
            task_manager._move_to_completed(task_id1)
            
            # Mark one as failed
            task_status = task_manager.active_tasks[task_id2]
            task_status.status = "failed"
            task_manager._move_to_completed(task_id2)
            
            metrics = task_manager.get_task_metrics()
            
            assert metrics["total_tasks"] == 2
            assert metrics["active_tasks"] == 0
            assert metrics["completed_tasks"] == 2
            assert metrics["status_counts"]["completed"] == 1
            assert metrics["status_counts"]["failed"] == 1
            assert metrics["celery_available"] is False


class TestTaskUI:
    """Test task UI components"""
    
    def test_is_long_running_query_complex_keywords(self):
        """Test detection of long-running queries with complex keywords"""
        complex_queries = [
            "Please analyze this document comprehensively",
            "I need a detailed explanation of this topic",
            "Can you research this subject thoroughly",
            "Please evaluate this situation step by step",
            "I want a thorough examination of this data"
        ]
        
        for query in complex_queries:
            assert is_long_running_query(query, "Standard") is True
    
    def test_is_long_running_query_long_text(self):
        """Test detection of long-running queries based on length"""
        long_query = "This is a very long query that contains many words and should be considered as a long-running task because it has more than twenty words in it which exceeds the threshold"
        
        assert is_long_running_query(long_query, "Standard") is True
    
    def test_is_long_running_query_complex_modes(self):
        """Test detection of long-running queries for complex reasoning modes"""
        simple_query = "What is 2+2?"
        
        # Should be long-running for complex modes
        assert is_long_running_query(simple_query, "Multi-Step") is True
        assert is_long_running_query(simple_query, "Agent-Based") is True
        
        # Should not be long-running for simple modes
        assert is_long_running_query(simple_query, "Standard") is False
        assert is_long_running_query(simple_query, "Chain-of-Thought") is False
    
    def test_is_long_running_query_explicit_requests(self):
        """Test detection of long-running queries with explicit requests"""
        explicit_queries = [
            "Please take your time to answer this",
            "I need a detailed analysis of this",
            "Can you provide a comprehensive answer"
        ]
        
        for query in explicit_queries:
            assert is_long_running_query(query, "Standard") is True
    
    def test_should_use_background_task_enabled(self):
        """Test background task decision when enabled"""
        # Mock config with background tasks enabled
        mock_config = Mock()
        mock_config.enable_background_tasks = True
        
        complex_query = "Please analyze this comprehensively"
        
        assert should_use_background_task(complex_query, "Multi-Step", mock_config) is True
    
    def test_should_use_background_task_disabled(self):
        """Test background task decision when disabled"""
        # Mock config with background tasks disabled
        mock_config = Mock()
        mock_config.enable_background_tasks = False
        
        complex_query = "Please analyze this comprehensively"
        
        assert should_use_background_task(complex_query, "Multi-Step", mock_config) is False
    
    def test_create_task_message(self):
        """Test creating task messages"""
        task_id = "test-task-123"
        task_type = "Reasoning"
        
        message = create_task_message(task_id, task_type, query="test query")
        
        assert message["role"] == "assistant"
        assert message["is_task"] is True
        assert message["task_id"] == task_id
        assert task_type in message["content"]
        assert task_id in message["content"]
        assert "background" in message["content"].lower()
        assert message["metadata"]["query"] == "test query"


class TestTaskIntegration:
    """Integration tests for task management"""
    
    def test_task_lifecycle(self):
        """Test complete task lifecycle"""
        with patch('task_manager.Celery', side_effect=Exception("Celery not available")):
            task_manager = TaskManager()
            
            # Submit task
            task_id = task_manager.submit_task("test_task", param="value")
            
            # Check initial state
            task_status = task_manager.get_task_status(task_id)
            assert task_status.status == "running"
            assert task_status.progress == 0.1
            
            # Simulate task completion (in real scenario, this would be done by Celery)
            task_status.status = "completed"
            task_status.progress = 1.0
            task_status.result = {"test": "result"}
            task_manager._move_to_completed(task_id)
            
            # Check final state
            task_status = task_manager.get_task_status(task_id)
            assert task_status.status == "completed"
            assert task_status.progress == 1.0
            assert task_status.result == {"test": "result"}
            
            # Check metrics
            metrics = task_manager.get_task_metrics()
            assert metrics["total_tasks"] == 1
            assert metrics["completed_tasks"] == 1
            assert metrics["status_counts"]["completed"] == 1
    
    def test_task_error_handling(self):
        """Test task error handling"""
        with patch('task_manager.Celery', side_effect=Exception("Celery not available")):
            task_manager = TaskManager()
            
            # Submit task
            task_id = task_manager.submit_task("test_task")
            
            # Simulate task failure
            task_status = task_manager.get_task_status(task_id)
            task_status.status = "failed"
            task_status.error = "Test error message"
            task_manager._move_to_completed(task_id)
            
            # Check error state
            task_status = task_manager.get_task_status(task_id)
            assert task_status.status == "failed"
            assert task_status.error == "Test error message"
            
            # Check metrics
            metrics = task_manager.get_task_metrics()
            assert metrics["status_counts"]["failed"] == 1


if __name__ == "__main__":
    pytest.main([__file__]) 