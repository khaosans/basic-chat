"""
Celery configuration for BasicChat long-running tasks
"""

import os
from dotenv import load_dotenv

load_dotenv(".env.local")

# Broker and result backend configuration
broker_url = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
result_backend = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

# Serialization
task_serializer = 'json'
result_serializer = 'json'
accept_content = ['json']
timezone = 'UTC'
enable_utc = True

# Task routing
task_routes = {
    'tasks.run_reasoning': {'queue': 'reasoning'},
    'tasks.run_deep_research': {'queue': 'reasoning'},
    'tasks.analyze_document': {'queue': 'documents'},
    'tasks.process_document': {'queue': 'documents'},
}

# Task annotations (rate limiting, timeouts)
task_annotations = {
    'tasks.run_reasoning': {
        'rate_limit': '10/m',  # 10 tasks per minute
        'time_limit': 300,     # 5 minutes
        'soft_time_limit': 240  # 4 minutes soft limit
    },
    'tasks.run_deep_research': {
        'rate_limit': '5/m',   # 5 tasks per minute
        'time_limit': 900,     # 15 minutes
        'soft_time_limit': 840  # 14 minutes soft limit
    },
    'tasks.analyze_document': {
        'rate_limit': '5/m',   # 5 tasks per minute
        'time_limit': 600,     # 10 minutes
        'soft_time_limit': 540  # 9 minutes soft limit
    },
    'tasks.process_document': {
        'rate_limit': '5/m',   # 5 tasks per minute
        'time_limit': 600,     # 10 minutes
        'soft_time_limit': 540  # 9 minutes soft limit
    }
}

# Worker configuration
worker_prefetch_multiplier = 1
worker_max_tasks_per_child = 1000
worker_disable_rate_limits = False

# Result backend configuration
result_expires = 3600  # 1 hour
result_persistent = True

# Task result configuration
task_ignore_result = False
task_store_errors_even_if_ignored = True

# Logging
worker_log_format = '[%(asctime)s: %(levelname)s/%(processName)s] %(message)s'
worker_task_log_format = '[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s' 
