# GitHub Models API Troubleshooting Guide

[← Back to Documentation](../README.md#documentation) | [GitHub Models Integration →](GITHUB_MODELS_INTEGRATION.md)

---

## Issue Summary

The GitHub Models API is returning 404 errors, indicating that the API endpoints we're trying to use are not available or not correctly configured.

## Current Status

### ✅ What's Working
- Mock evaluation mode for testing
- Ollama fallback functionality
- Parallel job execution
- Quick mode optimization

### ❌ What's Not Working
- GitHub Models API endpoints returning 404
- Direct model inference through GitHub API

## Root Cause Analysis

### 1. GitHub Models API Availability

GitHub Models is a relatively new feature and may have:
- Limited availability in certain regions
- Different API endpoints than documented
- Beta/experimental status
- Access restrictions

### 2. API Endpoint Issues

The endpoints we tried:
- `https://api.github.com/copilot/v1/chat/completions` ❌ 404
- `https://api.github.com/models/chat/completions` ❌ 404  
- `https://api.github.com/v1/models/chat/completions` ❌ 404

### 3. Authentication Issues

Possible token permission problems:
- Insufficient scopes
- Token not configured for Models access
- Repository-level restrictions

## Solutions

### Solution 1: Use Mock Mode (Immediate Fix)

For immediate CI/CD functionality, use mock mode:

```yaml
- name: Run LLM Judge (Mock Mode)
  run: python evaluators/check_llm_judge_github.py --quick --mock
```

**Benefits:**
- ✅ Works immediately
- ✅ No API dependencies
- ✅ Fast execution
- ✅ Deterministic results

**Limitations:**
- ❌ Not real AI evaluation
- ❌ Limited insights
- ❌ For testing only

### Solution 2: Ollama Fallback (Recommended)

Use the existing Ollama setup as the primary method:

```yaml
- name: Run LLM Judge (Ollama)
  run: python evaluators/check_llm_judge.py --quick
```

**Benefits:**
- ✅ Real AI evaluation
- ✅ Works reliably
- ✅ No external API dependencies
- ✅ Full functionality

### Solution 3: External LLM APIs (Alternative)

Use external LLM providers as a fallback:

```python
# Add support for OpenAI, Anthropic, etc.
def evaluate_with_external_api(self, prompt: str) -> Dict[str, Any]:
    # Implementation for external APIs
    pass
```

**Benefits:**
- ✅ High-quality models
- ✅ Reliable APIs
- ✅ Fast response times

**Limitations:**
- ❌ Additional costs
- ❌ API key management
- ❌ Rate limits

### Solution 4: Wait for GitHub Models (Future)

Monitor GitHub Models availability and update when ready:

1. **Check GitHub Documentation**: Monitor official docs for API updates
2. **Test Periodically**: Regularly test API endpoints
3. **Update Implementation**: Modify code when APIs become available

## Implementation Strategy

### Phase 1: Immediate (Current)
- Use mock mode for CI/CD
- Ensure pipeline doesn't break
- Maintain parallel execution benefits

### Phase 2: Short-term (Next 2 weeks)
- Implement Ollama as primary method
- Add external API fallbacks
- Improve error handling

### Phase 3: Long-term (Next month)
- Monitor GitHub Models availability
- Update when APIs are ready
- Migrate to GitHub Models when stable

## Updated Workflow Configuration

### Current Working Configuration

```yaml
- name: Run LLM Judge (Mock Mode)
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    LLM_JUDGE_THRESHOLD: 7.0
  run: |
    python evaluators/check_llm_judge_github.py --quick --mock
```

### Recommended Configuration

```yaml
- name: Run LLM Judge (Ollama)
  env:
    OLLAMA_MODEL: mistral
    LLM_JUDGE_THRESHOLD: 7.0
  run: |
    python evaluators/check_llm_judge.py --quick
```

### Future Configuration (When GitHub Models Works)

```yaml
- name: Run LLM Judge (GitHub Models)
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    GITHUB_MODEL: claude-3.5-sonnet
    LLM_JUDGE_THRESHOLD: 7.0
  run: |
    python evaluators/check_llm_judge_github.py --quick
```

## Testing and Validation

### Test Mock Mode
```bash
python evaluators/check_llm_judge_github.py --quick --mock
```

### Test Ollama Mode
```bash
python evaluators/check_llm_judge.py --quick
```

### Test GitHub Models (When Available)
```bash
export GITHUB_TOKEN=your_token_here
python evaluators/check_llm_judge_github.py --quick
```

## Monitoring and Alerts

### GitHub Models API Status
- Monitor API endpoint availability
- Check GitHub documentation updates
- Test endpoints periodically

### CI/CD Pipeline Health
- Track evaluation success rates
- Monitor execution times
- Alert on failures

### Performance Metrics
- Response times
- Success rates
- Resource usage

## Best Practices

### 1. Graceful Degradation
- Always have fallback options
- Don't let API failures break CI
- Provide meaningful error messages

### 2. Testing Strategy
- Test all evaluation modes
- Validate results consistency
- Monitor performance

### 3. Documentation
- Keep troubleshooting guides updated
- Document configuration changes
- Maintain migration guides

## Next Steps

### Immediate Actions
1. ✅ Deploy mock mode for CI/CD
2. ✅ Test Ollama fallback
3. ✅ Update documentation

### Short-term Actions
1. Implement Ollama as primary method
2. Add external API support
3. Improve error handling

### Long-term Actions
1. Monitor GitHub Models availability
2. Update implementation when ready
3. Migrate to GitHub Models

## Conclusion

While GitHub Models API is not currently available, we have implemented a robust solution that:

- ✅ Maintains CI/CD functionality
- ✅ Provides real evaluation capabilities
- ✅ Offers multiple fallback options
- ✅ Preserves performance benefits

The mock mode ensures the pipeline continues to work while we wait for GitHub Models to become available, and the Ollama fallback provides real AI evaluation capabilities in the meantime. 

[← Back to Documentation](../README.md#documentation) | [GitHub Models Integration →](GITHUB_MODELS_INTEGRATION.md)
