#!/usr/bin/env python3
"""
Deterministic Performance Regression Test for LLM Judge (OpenAI, Ollama, or Hugging Face)

Measures evaluation time and memory usage for the LLM Judge evaluator using OpenAI API, local Ollama, or Hugging Face models.
Fails if thresholds are exceeded. Designed for CI/CD pipelines and local dev.

- Uses OpenAI's gpt-3.5-turbo by default for cost efficiency (frugal mode).
- Uses psutil for cross-platform memory measurement if available, otherwise falls back to resource (Unix-only).
- Select backend with LLM_JUDGE_BACKEND env var: "OPENAI" (default), "OLLAMA", or "HUGGINGFACE"

Usage:
    python scripts/test_performance_regression.py

Environment Variables:
    PERF_TIME_THRESHOLD: Max allowed seconds (default: 30.0)
    PERF_MEM_THRESHOLD: Max allowed MB (default: 600.0)
    OPENAI_API_KEY: Your OpenAI API key (required for OpenAI backend)
    OPENAI_MODEL: OpenAI model to use (default: gpt-3.5-turbo)
    LLM_JUDGE_BACKEND: "OPENAI" (default), "OLLAMA", or "HUGGINGFACE"
    HF_API_KEY: Hugging Face API key (optional for public models)
    HF_MODEL: Hugging Face model to use (default: microsoft/DialoGPT-medium)
    HF_DEVICE: Device to use (default: auto, options: cpu, cuda, mps)

Notes:
    - The default memory threshold (600MB) is set based on typical LLM evaluation memory usage.
      Adjust this value if your model or workload changes.
    - This script uses OpenAI for CI compatibility and cost control, but can use Ollama or Hugging Face locally.
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

from evaluators.check_llm_judge_openai import OpenAIEvaluator
from evaluators.check_llm_judge import LLMJudgeEvaluator

THRESHOLD_SECONDS = float(os.getenv("PERF_TIME_THRESHOLD", "30.0"))  # e.g., 30s
THRESHOLD_MB = float(os.getenv("PERF_MEM_THRESHOLD", "600.0"))      # e.g., 600MB
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BACKEND = os.getenv("LLM_JUDGE_BACKEND", "OPENAI").upper()

# Hugging Face config
def try_import_hf_evaluator():
    try:
        from evaluators.check_llm_judge_huggingface import HuggingFaceEvaluator
        return HuggingFaceEvaluator
    except ImportError as e:
        print("❌ Could not import HuggingFaceEvaluator. Did you create evaluators/check_llm_judge_huggingface.py?", file=sys.stderr)
        print("   Error:", e, file=sys.stderr)
        return None

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
    if BACKEND == "OLLAMA":
        print("[Performance Test] Using Ollama backend (local LLM)")
        evaluator = LLMJudgeEvaluator(quick_mode=True)
    elif BACKEND == "HUGGINGFACE":
        print("[Performance Test] Using Hugging Face backend (transformers)")
        HuggingFaceEvaluator = try_import_hf_evaluator()
        if not HuggingFaceEvaluator:
            print("❌ Hugging Face evaluator not available. Exiting.", file=sys.stderr)
            sys.exit(1)
        # Optionally check for transformers/torch
        try:
            import transformers
            import torch
        except ImportError:
            print("❌ transformers/torch not installed. Install with: pip install transformers torch", file=sys.stderr)
            sys.exit(1)
        # Check for model env
        hf_model = os.getenv("HF_MODEL", "microsoft/DialoGPT-medium")
        # API key is optional for public models
        evaluator = HuggingFaceEvaluator(quick_mode=True, model=hf_model)
    else:
        print("[Performance Test] Using OpenAI backend (frugal, CI-friendly)")
        if not OPENAI_API_KEY:
            print("Error: OPENAI_API_KEY environment variable is required for OpenAI backend.", file=sys.stderr)
            sys.exit(1)
        evaluator = OpenAIEvaluator(quick_mode=True, model=OPENAI_MODEL)

    start_time = time.time()
    start_mem = get_memory_mb()

    # Run the evaluation (do not print results to avoid CI log noise)
    evaluator.run_evaluation()

    end_time = time.time()
    end_mem = get_memory_mb()

    elapsed = end_time - start_time
    mem_used = max(0.0, end_mem - start_mem)

    metrics = {
        "backend": BACKEND,
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