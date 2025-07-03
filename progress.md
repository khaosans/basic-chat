## 2025-07-03 — Deterministic Performance Regression Test for LLM Judge (by SourC)

- Added `scripts/test_performance_regression.py` to measure LLM Judge evaluation time and memory usage.
- Test fails if elapsed time > 30s or memory usage > 600MB (default, configurable).
- Integrated as a new job in CI after llm-judge, uploads `performance_metrics.json` artifact.
- Thresholds and rationale are documented in both the script and workflow.
- Ensures no performance regressions are merged. 