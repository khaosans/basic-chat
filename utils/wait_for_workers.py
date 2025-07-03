import time
import sys
from task_manager import TaskManager

timeout = 60
interval = 2
start = time.time()
print("[wait_for_workers] Waiting for Celery worker to become available...")
while time.time() - start < timeout:
    tm = TaskManager()
    if getattr(tm, 'celery_available', False):
        print("[wait_for_workers] Celery worker is available! Proceeding.")
        sys.exit(0)
    print("[wait_for_workers] Not available yet, retrying...")
    time.sleep(interval)
print("[wait_for_workers] ERROR: Celery worker not available after 60 seconds.")
sys.exit(1) 