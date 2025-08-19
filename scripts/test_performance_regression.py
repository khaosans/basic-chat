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

from basicchat.evaluation.evaluators.check_llm_judge_openai import OpenAIEvaluator
from basicchat.evaluation.evaluators.check_llm_judge import LLMJudgeEvaluator

THRESHOLD_SECONDS = float(os.getenv("PERF_TIME_THRESHOLD", "30.0"))  # e.g., 30s
THRESHOLD_MB = float(os.getenv("PERF_MEM_THRESHOLD", "600.0"))      # e.g., 600MB
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BACKEND = os.getenv("LLM_JUDGE_BACKEND", "OPENAI").upper()

# Hugging Face config
def try_import_hf_evaluator():
    try:
        from basicchat.evaluation.evaluators.check_llm_judge_huggingface import HuggingFaceEvaluator
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

    print(f"\n🚀 Starting Performance Regression Test")
    print(f"📅 Test Date: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print(f"🔧 Backend: {BACKEND}")
    print(f"⚡ Quick Mode: Enabled")
    print(f"🎯 Time Threshold: {THRESHOLD_SECONDS}s")
    print(f"💾 Memory Threshold: {THRESHOLD_MB}MB")
    print(f"🤖 Model: {OPENAI_MODEL if BACKEND == 'OPENAI' else 'Local Model'}")
    print("-" * 60)

    start_time = time.time()
    start_mem = get_memory_mb()

    print(f"📊 Initial Memory Usage: {start_mem:.2f}MB")
    print(f"⏱️  Starting evaluation at: {time.strftime('%H:%M:%S')}")

    # Run the evaluation (do not print results to avoid CI log noise)
    evaluator.run_evaluation()

    end_time = time.time()
    end_mem = get_memory_mb()

    elapsed = end_time - start_time
    mem_used = max(0.0, end_mem - start_mem)
    mem_peak = end_mem

    print(f"⏱️  Evaluation completed at: {time.strftime('%H:%M:%S')}")
    print(f"📊 Final Memory Usage: {end_mem:.2f}MB")

    # Calculate performance ratios
    time_ratio = (elapsed / THRESHOLD_SECONDS) * 100
    memory_ratio = (mem_used / THRESHOLD_MB) * 100

    # Determine performance grade
    if elapsed <= THRESHOLD_SECONDS * 0.5 and mem_used <= THRESHOLD_MB * 0.5:
        grade = "🟢 EXCELLENT"
    elif elapsed <= THRESHOLD_SECONDS * 0.8 and mem_used <= THRESHOLD_MB * 0.8:
        grade = "🟡 GOOD"
    elif elapsed <= THRESHOLD_SECONDS and mem_used <= THRESHOLD_MB:
        grade = "🟠 ACCEPTABLE"
    else:
        grade = "🔴 FAILED"

    metrics = {
        "test_info": {
            "date": time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
            "backend": BACKEND,
            "model": OPENAI_MODEL if BACKEND == 'OPENAI' else 'Local Model',
            "quick_mode": True,
            "test_type": "LLM Judge Evaluation Performance"
        },
        "performance": {
            "elapsed_seconds": round(elapsed, 3),
            "memory_mb": round(mem_used, 3),
            "memory_peak_mb": round(mem_peak, 3),
            "time_ratio_percent": round(time_ratio, 1),
            "memory_ratio_percent": round(memory_ratio, 1)
        },
        "thresholds": {
            "time_seconds": THRESHOLD_SECONDS,
            "memory_mb": THRESHOLD_MB
        },
        "status": {
            "overall": "PASS" if elapsed <= THRESHOLD_SECONDS and mem_used <= THRESHOLD_MB else "FAIL",
            "time_status": "PASS" if elapsed <= THRESHOLD_SECONDS else "FAIL",
            "memory_status": "PASS" if mem_used <= THRESHOLD_MB else "FAIL",
            "grade": grade
        }
    }

    # Output results for CI artifact
    with open("performance_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Print detailed results
    print(f"\n{'='*60}")
    print(f"📈 PERFORMANCE REGRESSION TEST RESULTS")
    print(f"{'='*60}")
    print(f"📅 Test Date: {metrics['test_info']['date']}")
    print(f"🔧 Backend: {metrics['test_info']['backend']}")
    print(f"🤖 Model: {metrics['test_info']['model']}")
    print(f"⚡ Mode: Quick Evaluation")
    print(f"")
    print(f"⏱️  EXECUTION TIME:")
    print(f"   • Elapsed: {metrics['performance']['elapsed_seconds']}s")
    print(f"   • Threshold: {metrics['thresholds']['time_seconds']}s")
    print(f"   • Usage: {metrics['performance']['time_ratio_percent']}% of threshold")
    print(f"   • Status: {metrics['status']['time_status']}")
    print(f"")
    print(f"💾 MEMORY USAGE:")
    print(f"   • Used: {metrics['performance']['memory_mb']}MB")
    print(f"   • Peak: {metrics['performance']['memory_peak_mb']}MB")
    print(f"   • Threshold: {metrics['thresholds']['memory_mb']}MB")
    print(f"   • Usage: {metrics['performance']['memory_ratio_percent']}% of threshold")
    print(f"   • Status: {metrics['status']['memory_status']}")
    print(f"")
    print(f"🎯 OVERALL RESULT:")
    print(f"   • Grade: {metrics['status']['grade']}")
    print(f"   • Status: {metrics['status']['overall']}")
    print(f"")
    
    if metrics['status']['overall'] == "PASS":
        print(f"✅ PERFORMANCE TEST PASSED")
        if grade == "🟢 EXCELLENT":
            print(f"   🎉 Excellent performance! Well under thresholds.")
        elif grade == "🟡 GOOD":
            print(f"   👍 Good performance within safe margins.")
        else:
            print(f"   ⚠️  Acceptable performance, but close to thresholds.")
    else:
        print(f"❌ PERFORMANCE TEST FAILED")
        print(f"   🚨 Performance regression detected!")
        if metrics['status']['time_status'] == "FAIL":
            print(f"   ⏱️  Time exceeded threshold by {elapsed - THRESHOLD_SECONDS:.2f}s")
        if metrics['status']['memory_status'] == "FAIL":
            print(f"   💾 Memory exceeded threshold by {mem_used - THRESHOLD_MB:.2f}MB")
    
    print(f"{'='*60}")
    print(f"📄 Results saved to: performance_metrics.json")
    print(f"📊 CI Artifact: performance-metrics.zip")
    print(f"{'='*60}\n")

    # Robust CI failure: assertion + sys.exit(1)
    if metrics['status']['overall'] != "PASS":
        print(f"❌ PERFORMANCE REGRESSION DETECTED", file=sys.stderr)
        print(f"   Time: {elapsed:.3f}s (threshold: {THRESHOLD_SECONDS}s)", file=sys.stderr)
        print(f"   Memory: {mem_used:.3f}MB (threshold: {THRESHOLD_MB}MB)", file=sys.stderr)
        sys.exit(1)
    
    print(f"✅ Performance test completed successfully!")

if __name__ == "__main__":
    main() 