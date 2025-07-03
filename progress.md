# 🚦 Fast, Repeatable Test Automation with Docker 🐳

## ✅ One-Liner (Fast, Parallel, Mocked)

```sh
# Run from project root:
docker compose run --rm app python scripts/run_tests.py --mode fast --parallel
```

- **Runs only the fastest, mocked tests**
- **Parallelized** for speed (uses pytest-xdist)
- **Consistent & repeatable**: Works locally and in CI
- **Unified Markdown report**: See `final_test_report.md` artifact after each CI run

---

## 🛠️ Makefile Target

Add to your `Makefile`:

```makefile
test-fast:
	docker compose run --rm app python scripts/run_tests.py --mode fast --parallel
```

Then run:
```sh
make test-fast
```

---

## 🤖 How It Works (ASCII Diagram)

```
+-------------------+         +-------------------+
|  Your Host Mac    |         |   Docker Image    |
|-------------------|         |-------------------|
|  docker compose   |  --->   |  app service      |
|  run ...          |         |  (runs test cmd)  |
+-------------------+         +-------------------+
```

---

## 📝 For Full Test Coverage

```sh
docker compose run --rm app python scripts/run_tests.py --mode all --parallel
```

---

## 💡 Pro Tips
- Use `--mode fast` for quick feedback in dev/PRs
- Use `--mode all` in CI for full coverage
- All test logic is in `scripts/run_tests.py` (supports unit, integration, slow, coverage, etc.)
- No need to run `pytest` directly!
- **Unified test report**: Download `final_test_report.md` from CI artifacts for a full summary

---

## 2025-07-03 — Deterministic Performance Regression Test for LLM Judge (by SourC)

- Added `scripts/test_performance_regression.py` to measure LLM Judge evaluation time and memory usage.
- Test fails if elapsed time > 30s or memory usage > 600MB (default, configurable).
- Integrated as a new job in CI after llm-judge, uploads `performance_metrics.json` artifact.
- Thresholds and rationale are documented in both the script and workflow.
- Ensures no performance regressions are merged.

# Progress Log

## [2025-07-03]

- Added `scripts/generate_final_report.py` to aggregate test, coverage, LLM Judge, and performance results into a single Markdown report (`final_test_report.md`).
- Updated `.github/workflows/verify.yml` to run the report script after all tests and upload the report as an artifact for every job.
- The report includes: test summary, coverage, LLM Judge results, performance metrics, recommendations, and a stub for comparisons to previous runs.
- All steps are robust to missing files and always generate a report for CI/CD visibility.
- **Parallelized all tests** for speed and reliability.
- **Documented fast, repeatable build process** in this file. 