# CI/CD Fixes Summary

## Issue
The CI/CD pipeline was failing because it was still trying to use the old `requirements.txt` file and module structure that was removed during the repository reorganization.

## Error Message
```
ERROR: Could not open requirements file: [Errno 2] No such file or directory: 'requirements.txt'
```

## Root Cause
During the repository reorganization, we:
1. Removed `requirements.txt` (replaced with `pyproject.toml` + Poetry)
2. Moved all Python modules into the `basicchat/` package structure
3. Moved one-off scripts to `temp/one-off-scripts/`
4. Changed the main entry point from `app.py` to `main.py`

## Fixes Applied

### 1. Updated Dependency Installation
**Before:**
```yaml
- name: Install dependencies
  run: |
    python -m pip install --upgrade pip
    pip install -r requirements.txt
```

**After:**
```yaml
- name: Install dependencies
  run: |
    python -m pip install --upgrade pip
    pip install poetry
    poetry install
```

### 2. Fixed Cache Keys
**Before:**
```yaml
key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
```

**After:**
```yaml
key: ${{ runner.os }}-pip-${{ hashFiles('pyproject.toml') }}
```

### 3. Updated Test Commands
**Before:**
```yaml
- name: Run unit tests only
  run: |
    python -m pytest -n auto tests/ -m "unit or fast" --ignore=tests/integration -v --tb=short --cov=app --cov=reasoning_engine --cov=document_processor --cov=utils --cov=task_manager --cov=task_ui --cov=tasks --cov-report=term-missing --cov-report=html:htmlcov
```

**After:**
```yaml
- name: Run unit tests only
  run: |
    poetry run pytest -n auto tests/ -m "unit or fast" --ignore=tests/integration -v --tb=short --cov=basicchat --cov-report=term-missing --cov-report=html:htmlcov
```

### 4. Fixed Script Paths
**Before:**
```yaml
- name: Generate test fixtures
  run: |
    python scripts/generate_test_assets.py || echo "Test assets generation failed, continuing..."
```

**After:**
```yaml
- name: Generate test fixtures
  run: |
    poetry run python temp/one-off-scripts/generate_test_assets.py || echo "Test assets generation failed, continuing..."
```

### 5. Updated Streamlit Entry Point
**Before:**
```yaml
- name: Start Streamlit app (background)
  run: streamlit run app.py --server.port 8501 --server.headless true --server.address 0.0.0.0 &
```

**After:**
```yaml
- name: Start Streamlit app (background)
  run: poetry run streamlit run main.py --server.port 8501 --server.headless true --server.address 0.0.0.0 &
```

## Files Modified
- `.github/workflows/verify.yml` - Main CI workflow
- `.github/workflows/e2e-smoke.yml` - E2E smoke test workflow

## Benefits
1. **✅ CI/CD now works with Poetry** - Uses modern Python dependency management
2. **✅ Proper package structure** - Tests use the new `basicchat/` package
3. **✅ Correct script paths** - All scripts reference the new locations
4. **✅ Updated coverage** - Coverage reports now target the `basicchat` package
5. **✅ Consistent with reorganization** - CI/CD matches the new repository structure

## Status
- **✅ All CI/CD workflows updated**
- **✅ Poetry integration complete**
- **✅ Package structure compatible**
- **✅ Ready for automated testing**

The CI/CD pipeline should now pass successfully with the reorganized repository structure!
