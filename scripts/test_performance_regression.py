#!/usr/bin/env python3
"""
Deterministic Performance Regression Test for LLM Judge

Measures evaluation time and memory usage for the LLM Judge evaluator.
Fails if thresholds are exceeded. Designed for CI/CD pipelines.

- Uses psutil for cross-platform memory measurement if available, otherwise falls back to resource (Unix-only).

Usage:
    python scripts/test_performance_regression.py

Environment Variables:
    PERF_TIME_THRESHOLD: Max allowed seconds (default: 30.0)
    PERF_MEM_THRESHOLD: Max allowed MB (default: 600.0)

Notes:
    - The default memory threshold (600MB) is set based on typical LLM evaluation memory usage.
      Adjust this value if your model or workload changes.
"""

import time
import json
import os
import sys
from pathlib import Path

# Try to import psutil for cross-platform memory measurement
try:
    import psutil
    _USE_PSUTIL = True
except ImportError:
    _USE_PSUTIL = False
    import resource  # Unix only

# Add the parent directory to the path so we can import from app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluators.check_llm_judge import LLMJudgeEvaluator

THRESHOLD_SECONDS = float(os.getenv("PERF_TIME_THRESHOLD", "30.0"))  # e.g., 30s
# Default memory threshold set to 600MB for LLM evaluation workloads
THRESHOLD_MB = float(os.getenv("PERF_MEM_THRESHOLD", "600.0"))      # e.g., 600MB


def get_memory_mb():
    if _USE_PSUTIL:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)  # MB
    else:
        # ru_maxrss is in kilobytes on Linux, bytes on macOS
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            mem = mem / (1024 * 1024)  # bytes to MB
        else:
            mem = mem / 1024  # kB to MB
        return mem

def main():
    evaluator = LLMJudgeEvaluator(quick_mode=True)
    start_time = time.time()
    start_mem = get_memory_mb()

    # Run the evaluation (do not print results to avoid CI log noise)
    evaluator.run_evaluation()

    end_time = time.time()
    end_mem = get_memory_mb()

    elapsed = end_time - start_time
    mem_used = max(0.0, end_mem - start_mem)

    metrics = {
        "elapsed_seconds": round(elapsed, 2),
        "memory_mb": round(mem_used, 2),
        "threshold_seconds": THRESHOLD_SECONDS,
        "threshold_mb": THRESHOLD_MB,
        "status": "PASS" if elapsed <= THRESHOLD_SECONDS and mem_used <= THRESHOLD_MB else "FAIL"
    }

    # Output results for CI artifact
    with open("performance_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n===== Performance Regression Metrics =====")
    print(json.dumps(metrics, indent=2))
    print("========================================\n")

    # Robust CI failure: assertion + sys.exit(1)
    assert metrics["status"] == "PASS", (
        f"Performance regression: time={elapsed:.2f}s, mem={mem_used:.2f}MB"
    )
    if metrics["status"] != "PASS":
        print(f"Performance regression: time={elapsed:.2f}s, mem={mem_used:.2f}MB", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 