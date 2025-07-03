## 2025-07-03 — Deterministic Performance Regression Test for LLM Judge (by SourC)

- Added `scripts/test_performance_regression.py` to measure LLM Judge evaluation time and memory usage.
- Test fails if elapsed time > 30s or memory usage > 600MB (default, configurable).
- Integrated as a new job in CI after llm-judge, uploads `performance_metrics.json` artifact.
- Thresholds and rationale are documented in both the script and workflow.
- Ensures no performance regressions are merged.

## 2025-07-03 — Playwright E2E Testing Setup (by SourC)

- Created branch feature/playwright-e2e-testing for Playwright E2E testing setup
- Added package.json and playwright.config.ts for Playwright
- To be followed by E2E test suites, fixtures, and CI integration

# Progress Log

## [Date: YYYY-MM-DD]

- Added `scripts/generate_final_report.py` to aggregate test, coverage, LLM Judge, and performance results into a single Markdown report (`final_test_report.md`).
- Updated `.github/workflows/verify.yml` to run the report script after all tests and upload the report as an artifact for every job.
- The report includes: test summary, coverage, LLM Judge results, performance metrics, recommendations, and a stub for comparisons to previous runs.
- All steps are robust to missing files and always generate a report for CI/CD visibility.

# E2E Test Hardening & Infra Health Check (2024-06)

## 🚦 New: E2E Infra Health Check Script
- **File:** `scripts/e2e_health_check.py`
- **Purpose:** Checks if all required services are up before running E2E tests:
  - Streamlit app (`localhost:8501`)
  - Ollama API (`localhost:11434`, models: `mistral`, `nomic-embed-text`)
  - ChromaDB (`localhost:8000`)
  - Redis (`localhost:6379`)
- **Usage:**
  ```bash
  poetry run python scripts/e2e_health_check.py
  # or
  python scripts/e2e_health_check.py
  ```
- **Behavior:**
  - Prints ✅/❌ for each service
  - Fails fast with clear error if any service is down
  - Summary at the end (ALL SERVICES HEALTHY 🟢 or SOME SERVICES UNHEALTHY 🔴)

## 🛡️ E2E Selector Hardening & Retry Logic
- **Refactored:** `tests/e2e/specs/basic-e2e.spec.ts`
  - Uses robust `data-testid` selectors via `ChatHelper` for all main flows
  - DRY: All common actions (send message, upload, select mode) use helpers
  - Retry logic and explicit waits for LLM and document upload steps
- **Benefits:**
  - Less flakiness from UI changes or slow infra
  - Easier to maintain and debug

## 📝 Best Practices
- **Always run the health check before E2E:**
  ```bash
  poetry run python scripts/e2e_health_check.py
  ```
- **If a test fails:**
  - Check the Playwright HTML report for logs/screenshots
  - Ensure all infra is up and healthy
  - If infra is slow, increase timeouts in helpers

## 🏁 Next Steps
- Harden other E2E specs to use helpers/selectors (optional)
- Add more health checks as needed (e.g., S3, Supabase)
- Keep this doc updated with new E2E patterns 