# LLM Judge Evaluation Flow

[← Back to Documentation](../README.md#documentation) | [Evaluators →](EVALUATORS.md)

---

## Overview

The LLM Judge evaluation system uses OpenAI's cost-effective models to assess code quality, test coverage, documentation, and overall project health. This provides automated quality assurance for the codebase.

## Current Setup

### OpenAI-Only Evaluation

The system now uses **OpenAI API exclusively** with the most cost-effective model available:

- **Default Model**: `gpt-3.5-turbo`
- **Cost**: ~$0.0015 per 1K input tokens
- **Quality**: Good for code evaluation
- **Speed**: Fast response times

### Configuration

The evaluation can be configured via GitHub Actions variables:

```yaml
OPENAI_MODEL: gpt-3.5-turbo  # Default: cheapest chat model
LLM_JUDGE_THRESHOLD: 7.0     # Default: minimum passing score
```

### Available Models

| Model | Cost per 1K tokens | Quality | Speed | Max Tokens |
|-------|-------------------|---------|-------|------------|
| `gpt-3.5-turbo` | $0.0015 | Good | Fast | 4,096 |
| `gpt-3.5-turbo-16k` | $0.003 | Good | Fast | 16,384 |
| `gpt-4-turbo` | $0.01 | Excellent | Fast | 128,000 |
| `gpt-4` | $0.03 | Excellent | Medium | 8,192 |

## Workflow Integration

### GitHub Actions

The LLM Judge runs as a parallel job in the CI/CD pipeline:

```yaml
llm-judge:
  runs-on: ubuntu-latest
  if: |
    (github.event_name == 'push' && github.ref == 'refs/heads/main') ||
    (github.event_name == 'pull_request' && github.event.pull_request.head.repo.full_name == github.repository)
```

### Triggers

- **Main branch pushes**: Full evaluation
- **PRs from same repo**: Quick evaluation
- **External PRs**: Skipped for security

### Quick Mode

For faster CI runs, the evaluator supports a `--quick` flag that:
- Focuses on critical files only
- Skips detailed test coverage analysis
- Provides faster feedback
- Reduces API costs

## Evaluation Criteria

The LLM Judge evaluates the following aspects (1-10 scale):

1. **Code Quality**: Structure, naming, complexity, Python best practices
2. **Test Coverage**: Comprehensiveness, quality, effectiveness
3. **Documentation**: README quality, inline docs, project documentation
4. **Architecture**: Design patterns, modularity, scalability
5. **Security**: Potential vulnerabilities, security best practices
6. **Performance**: Code efficiency, optimization opportunities

## Results

### Output Format

Results are saved to `llm_judge_results.json`:

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "scores": {
    "code_quality": {"score": 8, "justification": "..."},
    "test_coverage": {"score": 7, "justification": "..."},
    "documentation": {"score": 6, "justification": "..."},
    "architecture": {"score": 8, "justification": "..."},
    "security": {"score": 7, "justification": "..."},
    "performance": {"score": 7, "justification": "..."}
  },
  "overall_score": 7.2,
  "recommendations": ["..."],
  "model_used": "gpt-3.5-turbo",
  "estimated_cost": 0.0001
}
```

### Artifacts

- Results are uploaded as GitHub Actions artifacts
- Retention: 30 days
- Accessible via workflow runs

## Cost Optimization

### Strategies

1. **Quick Mode**: Reduces token usage by 60-80%
2. **Cheap Model**: Uses `gpt-3.5-turbo` by default
3. **Smart Triggering**: Only runs on relevant changes
4. **Parallel Execution**: Doesn't block other CI jobs

### Estimated Costs

- **Quick Mode**: ~$0.0001-0.0003 per evaluation
- **Full Mode**: ~$0.0005-0.001 per evaluation
- **Monthly (100 evaluations)**: ~$0.01-0.10

## Setup Requirements

### Environment Variables

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### GitHub Secrets

Add your OpenAI API key to repository secrets:
- Go to Settings → Secrets and variables → Actions
- Add `OPENAI_API_KEY` with your API key

### GitHub Variables (Optional)

Set these in repository variables for customization:
- `OPENAI_MODEL`: Model to use (default: gpt-3.5-turbo)
- `LLM_JUDGE_THRESHOLD`: Minimum passing score (default: 7.0)

## Testing

### Local Testing

```bash
# Set your API key
export OPENAI_API_KEY="your-key-here"

# Test the evaluator
python scripts/test_openai_evaluation.py

# Or run directly
python evaluators/check_llm_judge_openai.py --quick
```

### CI Testing

The workflow automatically tests the evaluator on:
- Main branch pushes
- PRs from the same repository
- Skips external PRs for security

## Troubleshooting

### Common Issues

1. **Missing API Key**: Set `OPENAI_API_KEY` in GitHub secrets
2. **Model Not Found**: Check model name in `COST_EFFECTIVE_MODELS`
3. **Timeout**: Increase timeout in workflow or use quick mode
4. **JSON Parse Error**: Model response format issues

### Debug Mode

Enable debug output by setting environment variable:
```bash
export DEBUG=1
python evaluators/check_llm_judge_openai.py --quick
```

## Future Enhancements

- [ ] Support for other LLM providers
- [ ] Custom evaluation criteria
- [ ] Historical trend analysis
- [ ] Integration with code review tools
- [ ] Automated fix suggestions 

[← Back to Documentation](../README.md#documentation) | [Evaluators →](EVALUATORS.md)
