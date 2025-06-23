# LLM Judge Evaluator

This directory contains the LLM-based code quality evaluator for the GitHub Actions CI pipeline.

## Overview

The LLM Judge Evaluator uses OpenAI's GPT models to assess code quality, test coverage, documentation, architecture, security, and performance of the codebase. It runs as a separate job in the CI pipeline after unit tests pass.

## Files

- `check_llm_judge.py` - Main evaluator script
- `evaluator.config.json` - Configuration file with thresholds and settings
- `README.md` - This documentation

## Configuration

### Environment Variables

- `OPENAI_API_KEY` (required): Your OpenAI API key
- `LLM_JUDGE_THRESHOLD` (optional): Minimum score required (default: 7.0)
- `LLM_JUDGE_MODEL` (optional): OpenAI model to use (default: gpt-4)

### GitHub Secrets

Add the following secret to your GitHub repository:
- `OPENAI_API_KEY`: Your OpenAI API key

### GitHub Variables (Optional)

You can set these as repository variables for easier management:
- `LLM_JUDGE_THRESHOLD`: Minimum score threshold
- `LLM_JUDGE_MODEL`: OpenAI model to use

## Usage

### Local Development

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Run the evaluator
python evaluators/check_llm_judge.py
```

### CI/CD Pipeline

The evaluator runs automatically in GitHub Actions after tests pass. It only runs on:
- Pushes to the main branch
- Pull requests from the same repository (for security)

## Evaluation Criteria

The evaluator assesses the following categories:

1. **Code Quality** (1-10): Code structure, naming conventions, complexity, Python best practices
2. **Test Coverage** (1-10): Test comprehensiveness, quality, and effectiveness
3. **Documentation** (1-10): README quality, inline documentation, project docs
4. **Architecture** (1-10): Design patterns, modularity, scalability
5. **Security** (1-10): Security vulnerabilities and best practices
6. **Performance** (1-10): Code efficiency and optimization opportunities

## Output

The evaluator generates:
- Console output with detailed scores and recommendations
- `llm_judge_results.json` file with complete evaluation data
- GitHub Actions artifact with results for historical tracking

## Security Considerations

- Only runs on trusted repositories (same-repo PRs)
- API key is stored as GitHub secret
- No code or sensitive data is sent to OpenAI
- Only metadata and statistics are evaluated

## Customization

You can customize the evaluation by modifying:
- `evaluator.config.json` for thresholds and weights
- The evaluation prompt in `check_llm_judge.py`
- The categories and criteria being assessed

## Troubleshooting

### Common Issues

1. **API Key Missing**: Ensure `OPENAI_API_KEY` is set in GitHub secrets
2. **Low Scores**: Review the recommendations and improve code quality
3. **API Failures**: The evaluator retries up to 3 times automatically
4. **Timeout Issues**: Increase timeout values in the config if needed

### Debug Mode

For debugging, you can run with verbose output:
```bash
export LLM_JUDGE_DEBUG=1
python evaluators/check_llm_judge.py
```

## Best Practices

1. **Start with a lower threshold** (e.g., 6.0) and gradually increase
2. **Review recommendations** regularly and implement improvements
3. **Monitor costs** - the evaluator uses OpenAI API calls
4. **Use caching** - GitHub Actions caches dependencies to speed up runs
5. **Set appropriate timeouts** for your codebase size

## Integration

The evaluator integrates seamlessly with:
- GitHub Actions CI/CD
- Existing test suites
- Code coverage tools
- Documentation systems

## Support

For issues or questions:
1. Check the GitHub Actions logs
2. Review the `llm_judge_results.json` output
3. Verify your OpenAI API key and quota
4. Check the configuration in `evaluator.config.json` 