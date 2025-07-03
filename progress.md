## 2025-07-03 — Deterministic Performance Regression Test for LLM Judge (by SourC)

- Added `scripts/test_performance_regression.py` to measure LLM Judge evaluation time and memory usage.
- Test fails if elapsed time > 30s or memory usage > 600MB (default, configurable).
- Integrated as a required check in CI under the 'verifyExpected' workflow, uploads `performance_metrics.json` artifact. The old 'llm-judge' check is no longer required.
- Thresholds and rationale are documented in both the script and workflow.
- Ensures no performance regressions are merged.

# Progress Log

## [Date: YYYY-MM-DD]

- Added a new CI Build Reference doc (`docs/CI_BUILD.md`) as the main entry point for CI build documentation. This doc links to the workflow YAML, CI optimization strategy, and testing guide, and includes a workflow diagram, step breakdown, and troubleshooting section.
- Updated `docs/CI_OPTIMIZATION.md` to add a prominent link to the new CI Build Reference at the top.
- Ensured no duplicate content; all docs cross-link for clarity and discoverability.
- All changes are backwards compatible and follow the documentation pattern. 