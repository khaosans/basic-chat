"""
Background task management for BasicChat.

This module handles background tasks, task scheduling, and task monitoring.
"""

from .task_manager import TaskManager
from .task_ui import (
    display_task_status,
    create_task_message,
    display_task_result,
    display_task_metrics,
    display_active_tasks,
    should_use_background_task,
    create_deep_research_message
)
from .tasks import *

__all__ = [
    "TaskManager",
    "display_task_status",
    "create_task_message", 
    "display_task_result",
    "display_task_metrics",
    "display_active_tasks",
    "should_use_background_task",
    "create_deep_research_message"
]
