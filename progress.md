## 2025-07-03 — Deterministic Performance Regression Test for LLM Judge (by SourC)

- Added `scripts/test_performance_regression.py` to measure LLM Judge evaluation time and memory usage.
- Test fails if elapsed time > 30s or memory usage > 600MB (default, configurable).
- Integrated as a required check in CI under the 'verifyExpected' workflow, uploads `performance_metrics.json` artifact. The old 'llm-judge' check is no longer required.
- Thresholds and rationale are documented in both the script and workflow.
- Ensures no performance regressions are merged. 