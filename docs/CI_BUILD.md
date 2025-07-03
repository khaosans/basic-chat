[🏠 Documentation Home](../README.md#documentation)

---

# 🚦 CI Build Reference

> **Your quick guide to the CI build process for this repository.**

---

## 📝 Overview

This project uses **GitHub Actions** for CI/CD. Every push or pull request to `main` triggers a multi-stage workflow that runs unit tests, integration tests, and performance regression checks. All jobs must pass for a PR to be merged.

- **Workflow file:** [verify.yml](../.github/workflows/verify.yml)
- **Test details:** [TESTING.md](./TESTING.md)
- **Optimization strategy:** [CI_OPTIMIZATION.md](./CI_OPTIMIZATION.md)

---

## 🗺️ Workflow Diagram

```ascii
+-------------------+
|   unit-tests      |
+-------------------+
          |
          v
+-------------------+
| integration-tests |
+-------------------+
          |
          v
+--------------------------+
| performance-regression   |
+--------------------------+
```

- **unit-tests**: Always runs first, required for all code changes
- **integration-tests**: Runs after unit-tests, required for main branch and PRs with [run-integration] in title/commit
- **performance-regression**: Runs in parallel, checks for performance regressions

---

## 🛠️ Workflow Steps

1. **Detect code changes**: Only runs jobs if relevant files changed
2. **Set up Python 3.11**: Ensures consistent environment
3. **Cache dependencies**: Speeds up repeated builds
4. **Install dependencies**: From `requirements.txt`
5. **Prepare test environment**: Creates test dirs, assets
6. **Run tests**: Unit, integration, and performance
7. **Upload artifacts**: Coverage and performance reports
8. **Cleanup**: Removes temp files after integration tests

---

## 🔗 Related Docs & Links

- [CI Optimization Strategy](./CI_OPTIMIZATION.md)
- [Testing Guide](./TESTING.md)
- [verify.yml workflow](../.github/workflows/verify.yml)
- [Architecture Overview](./ARCHITECTURE.md)

---

## 🚦 How to Trigger CI

- **Push to `main`**: Triggers all jobs
- **Pull request to `main`**: Triggers all jobs
- **[run-integration]** in PR title/commit: Forces integration tests

**Required checks for PRs:**
- `verifyExpected / unit-tests (3.11)`
- `verifyExpected / integration-tests`
- `verifyExpected / performance-regression`

---

## 🐞 Troubleshooting

- **Failing tests?** See [TESTING.md](./TESTING.md) for test structure and debugging tips
- **CI not running?** Ensure your branch/PR targets `main` and modifies relevant files
- **Performance regression?** See [CI_OPTIMIZATION.md](./CI_OPTIMIZATION.md) for tuning tips
- **Old checks (llm-judge) stuck?** Remove from branch protection rules (see CI_OPTIMIZATION.md)

For more, see the [CI Optimization Strategy](./CI_OPTIMIZATION.md) and [Testing Guide](./TESTING.md).

---

[🏠 Back to Documentation Home](../README.md#documentation) 