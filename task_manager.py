"""
Task Manager for BasicChat long-running tasks
Handles task submission, status tracking, and result management
"""

import uuid
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

from celery import Celery
from celery.result import AsyncResult

logger = logging.getLogger(__name__)

@dataclass
class TaskStatus:
    """Represents the status of a long-running task"""
    task_id: str
    status: str  # pending, running, completed, failed, cancelled
    progress: float  # 0.0 to 1.0
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    celery_task_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'task_id': self.task_id,
            'status': self.status,
            'progress': self.progress,
            'result': self.result,
            'error': self.error,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'metadata': self.metadata,
            'celery_task_id': self.celery_task_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskStatus':
        """Create from dictionary"""
        return cls(**data)

class TaskManager:
    """Manages long-running tasks with Celery backend"""
    
    def __init__(self):
        """Initialize the task manager"""
        try:
            self.celery_app = Celery('basic_chat')
            self.celery_app.config_from_object('celery_config')
            self.celery_available = True
            logger.info("Celery initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Celery: {e}")
            self.celery_available = False
        
        # In-memory task tracking (fallback when Redis is not available)
        self.active_tasks: Dict[str, TaskStatus] = {}
        self.completed_tasks: Dict[str, TaskStatus] = {}
        self.max_completed_tasks = 100  # Keep last 100 completed tasks
    
    def submit_task(self, task_type: str, **kwargs) -> str:
        """Submit a new long-running task"""
        task_id = str(uuid.uuid4())
        
        # Create task status
        task_status = TaskStatus(
            task_id=task_id,
            status="pending",
            progress=0.0,
            metadata={
                "task_type": task_type,
                "submitted_at": datetime.now().isoformat(),
                **kwargs
            }
        )
        
        self.active_tasks[task_id] = task_status
        
        # Submit to Celery if available
        if self.celery_available:
            try:
                celery_task = self._submit_celery_task(task_type, task_id, **kwargs)
                if celery_task:
                    task_status.celery_task_id = celery_task.id
                    task_status.status = "running"
                    logger.info(f"Task {task_id} submitted to Celery: {celery_task.id}")
                else:
                    task_status.status = "failed"
                    task_status.error = "Failed to submit to Celery"
                    logger.error(f"Failed to submit task {task_id} to Celery")
            except Exception as e:
                task_status.status = "failed"
                task_status.error = f"Celery submission error: {str(e)}"
                logger.error(f"Celery submission error for task {task_id}: {e}")
        else:
            # Fallback: simulate task processing
            logger.info(f"Celery not available, using fallback for task {task_id}")
            self._simulate_task_processing(task_id, task_type, **kwargs)
        
        return task_id
    
    def _submit_celery_task(self, task_type: str, task_id: str, **kwargs) -> Optional[Any]:
        """Submit task to Celery"""
        try:
            if task_type == "reasoning":
                return self.celery_app.send_task(
                    'tasks.run_reasoning',
                    args=[task_id, kwargs.get('query', ''), kwargs.get('mode', 'Standard')],
                    kwargs={'context': kwargs.get('context', '')}
                )
            elif task_type == "deep_research":
                return self.celery_app.send_task(
                    'tasks.run_deep_research',
                    args=[task_id, kwargs.get('query', '')],
                    kwargs={'research_depth': kwargs.get('research_depth', 'comprehensive')}
                )
            elif task_type == "document_analysis":
                return self.celery_app.send_task(
                    'tasks.analyze_document',
                    args=[task_id, kwargs.get('file_path', '')],
                    kwargs={'file_type': kwargs.get('file_type', 'unknown')}
                )
            elif task_type == "document_processing":
                return self.celery_app.send_task(
                    'tasks.process_document',
                    args=[task_id, kwargs.get('file_path', '')],
                    kwargs={'file_type': kwargs.get('file_type', 'unknown')}
                )
            else:
                logger.warning(f"Unknown task type: {task_type}")
                return None
        except Exception as e:
            logger.error(f"Error submitting Celery task: {e}")
            return None
    
    def _simulate_task_processing(self, task_id: str, task_type: str, **kwargs):
        """Simulate task processing when Celery is not available"""
        import threading
        
        def simulate_task():
            task_status = self.active_tasks.get(task_id)
            if not task_status:
                return
            
            try:
                # Simulate processing steps
                steps = [
                    (0.1, "Initializing"),
                    (0.3, "Processing"),
                    (0.6, "Analyzing"),
                    (0.8, "Finalizing"),
                    (1.0, "Completed")
                ]
                
                for progress, status in steps:
                    if task_id in self.active_tasks:
                        self.active_tasks[task_id].progress = progress
                        self.active_tasks[task_id].status = "running"
                        self.active_tasks[task_id].metadata["status"] = status
                        self.active_tasks[task_id].updated_at = time.time()
                        time.sleep(2)  # Simulate work
                    else:
                        return  # Task was cancelled
                
                # Mark as completed
                if task_id in self.active_tasks:
                    self.active_tasks[task_id].status = "completed"
                    self.active_tasks[task_id].result = {
                        "simulated": True,
                        "task_type": task_type,
                        "message": "Task completed (simulated mode)"
                    }
                    self.active_tasks[task_id].updated_at = time.time()
                    
                    # Move to completed tasks
                    self._move_to_completed(task_id)
                    
            except Exception as e:
                if task_id in self.active_tasks:
                    self.active_tasks[task_id].status = "failed"
                    self.active_tasks[task_id].error = str(e)
                    self.active_tasks[task_id].updated_at = time.time()
        
        # Start simulation in background thread
        thread = threading.Thread(target=simulate_task, daemon=True)
        thread.start()
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get current task status"""
        # Check active tasks first
        if task_id in self.active_tasks:
            task_status = self.active_tasks[task_id]
            
            # Update from Celery if available
            if self.celery_available and task_status.celery_task_id:
                self._update_from_celery(task_status)
            
            return task_status
        
        # Check completed tasks
        return self.completed_tasks.get(task_id)
    
    def _update_from_celery(self, task_status: TaskStatus):
        """Update task status from Celery result"""
        try:
            if not task_status.celery_task_id:
                return
            
            celery_result = AsyncResult(task_status.celery_task_id, app=self.celery_app)
            
            if celery_result.ready():
                task_status.updated_at = time.time()
                
                if celery_result.successful():
                    result = celery_result.result
                    task_status.status = "completed"
                    task_status.result = result.get('result') if isinstance(result, dict) else result
                    task_status.progress = 1.0
                    
                    # Move to completed tasks
                    self._move_to_completed(task_status.task_id)
                    
                else:
                    task_status.status = "failed"
                    task_status.error = str(celery_result.info)
                    
            elif celery_result.state == 'PENDING':
                task_status.status = "pending"
            elif celery_result.state == 'STARTED':
                task_status.status = "running"
                # Try to get progress from result
                if hasattr(celery_result, 'info') and celery_result.info:
                    if isinstance(celery_result.info, dict):
                        task_status.progress = celery_result.info.get('progress', task_status.progress)
                        task_status.metadata['status'] = celery_result.info.get('status', 'Running')
                        
        except Exception as e:
            logger.error(f"Error updating task status from Celery: {e}")
    
    def _move_to_completed(self, task_id: str):
        """Move task from active to completed"""
        if task_id in self.active_tasks:
            task_status = self.active_tasks.pop(task_id)
            self.completed_tasks[task_id] = task_status
            
            # Clean up old completed tasks
            if len(self.completed_tasks) > self.max_completed_tasks:
                oldest_task_id = min(self.completed_tasks.keys(), 
                                   key=lambda k: self.completed_tasks[k].created_at)
                self.completed_tasks.pop(oldest_task_id)
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task"""
        task_status = self.get_task_status(task_id)
        if not task_status:
            return False
        
        if task_status.status in ["pending", "running"]:
            # Cancel in Celery if available
            if self.celery_available and task_status.celery_task_id:
                try:
                    celery_result = AsyncResult(task_status.celery_task_id, app=self.celery_app)
                    celery_result.revoke(terminate=True)
                except Exception as e:
                    logger.error(f"Error cancelling Celery task: {e}")
            
            # Update status
            task_status.status = "cancelled"
            task_status.updated_at = time.time()
            return True
        
        return False
    
    def get_active_tasks(self) -> List[TaskStatus]:
        """Get all active tasks"""
        return list(self.active_tasks.values())
    
    def get_completed_tasks(self) -> List[TaskStatus]:
        """Get all completed tasks"""
        return list(self.completed_tasks.values())
    
    def cleanup_old_tasks(self, max_age_hours: int = 24):
        """Clean up old completed tasks"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        # Clean up completed tasks
        old_task_ids = [
            task_id for task_id, task_status in self.completed_tasks.items()
            if current_time - task_status.updated_at > max_age_seconds
        ]
        
        for task_id in old_task_ids:
            self.completed_tasks.pop(task_id, None)
        
        logger.info(f"Cleaned up {len(old_task_ids)} old completed tasks")
    
    def get_task_metrics(self) -> Dict[str, Any]:
        """Get task metrics"""
        active_tasks = self.get_active_tasks()
        completed_tasks = self.get_completed_tasks()
        
        status_counts = {
            "pending": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0
        }
        
        # Count active tasks
        for task in active_tasks:
            status_counts[task.status] += 1
        
        # Count completed tasks
        for task in completed_tasks:
            status_counts[task.status] += 1
        
        # Calculate average completion time
        completion_times = []
        for task in completed_tasks:
            if task.status == "completed" and task.updated_at and task.created_at:
                completion_times.append(task.updated_at - task.created_at)
        
        avg_completion_time = sum(completion_times) / len(completion_times) if completion_times else 0
        
        return {
            "status_counts": status_counts,
            "total_tasks": len(active_tasks) + len(completed_tasks),
            "active_tasks": len(active_tasks),
            "completed_tasks": len(completed_tasks),
            "avg_completion_time": avg_completion_time,
            "celery_available": self.celery_available
        } 