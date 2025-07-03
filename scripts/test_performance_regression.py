#!/usr/bin/env python3
"""
Deterministic Performance Regression Test for LLM Judge

Measures evaluation time and memory usage for the LLM Judge evaluator.
Fails if thresholds are exceeded. Designed for CI/CD pipelines.

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
import resource  # Unix only; use psutil for cross-platform
from pathlib import Path

# Add the parent directory to the path so we can import from app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluators.check_llm_judge import LLMJudgeEvaluator

THRESHOLD_SECONDS = float(os.getenv("PERF_TIME_THRESHOLD", "30.0"))  # e.g., 30s
# Default memory threshold set to 600MB for LLM evaluation workloads
THRESHOLD_MB = float(os.getenv("PERF_MEM_THRESHOLD", "600.0"))      # e.g., 600MB


def main():
    evaluator = LLMJudgeEvaluator(quick_mode=True)
    start_time = time.time()
    start_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB

    # Run the evaluation (do not print results to avoid CI log noise)
    evaluator.run_evaluation()

    end_time = time.time()
    end_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB

    elapsed = end_time - start_time
    mem_used = max(0.0, end_mem - start_mem)

    metrics = {
        "elapsed_seconds": round(elapsed, 2),
        "memory_mb": round(mem_used, 2),
        "threshold_seconds": THRESHOLD_SECONDS,
        "threshold_mb": THRESHOLD_MB,
        "status": "PASS" if elapsed < THRESHOLD_SECONDS and mem_used < THRESHOLD_MB else "FAIL"
    }

    # Output results for CI artifact
    with open("performance_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n===== Performance Regression Metrics =====")
    print(json.dumps(metrics, indent=2))
    print("========================================\n")

    assert metrics["status"] == "PASS", (
        f"Performance regression: time={elapsed:.2f}s, mem={mem_used:.2f}MB"
    )

if __name__ == "__main__":
    main() 