# LLM Judge Evaluator

This document describes the LLM-based code quality evaluator for the GitHub Actions CI pipeline.

## Overview

The LLM Judge Evaluator uses the built-in Ollama setup to assess code quality, test coverage, documentation, architecture, security, and performance of the codebase. It runs as a separate job in the CI pipeline after unit tests pass.

## Files

- `evaluators/check_llm_judge.py` - Main evaluator script
- `evaluators/evaluator.config.json` - Configuration file with thresholds and settings
- `evaluators/check_llm_judge_github.py` - GitHub Models API evaluator
- `evaluators/check_llm_judge_openai.py` - OpenAI API evaluator

## Configuration

### Environment Variables

- `OLLAMA_API_URL` (optional): Ollama API URL (default: http://localhost:11434/api)
- `OLLAMA_MODEL` (optional): Ollama model to use (default: mistral)
- `LLM_JUDGE_THRESHOLD` (optional): Minimum score required (default: 7.0)

### GitHub Variables (Optional)

You can set these as repository variables for easier management:
- `OLLAMA_API_URL`: Ollama API URL
- `OLLAMA_MODEL`: Ollama model to use (e.g., mistral, llama2, codellama)
- `LLM_JUDGE_THRESHOLD`: Minimum score threshold

## Usage

### Local Development

```bash
# Set your Ollama configuration (optional, uses defaults)
export OLLAMA_API_URL="http://localhost:11434/api"
export OLLAMA_MODEL="mistral"

# Run the evaluator
python evaluators/check_llm_judge.py
```

### CI/CD Pipeline

The evaluator runs automatically in GitHub Actions after tests pass. It only runs on:
- Pushes to the main branch
- Pull requests from the same repository (for security)

The CI automatically:
- Installs Ollama
- Pulls the specified model
- Runs the evaluation

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
- Uses local Ollama instance in CI
- No external API keys required
- Only metadata and statistics are evaluated

## Customization

You can customize the evaluation by modifying:
- `evaluators/evaluator.config.json` for thresholds and weights
- The evaluation prompt in `evaluators/check_llm_judge.py`
- The categories and criteria being assessed

## Troubleshooting

### Common Issues

1. **Ollama Not Running**: Ensure Ollama is installed and running locally
2. **Model Not Found**: Pull the required model with `ollama pull <model-name>`
3. **Low Scores**: Review the recommendations and improve code quality
4. **API Failures**: Check Ollama service status and model availability
5. **Timeout Issues**: Increase timeout values in the config if needed

### Debug Mode

For debugging, you can run with verbose output:
```bash
export LLM_JUDGE_DEBUG=1
python evaluators/check_llm_judge.py
```

## Best Practices

1. **Start with a lower threshold** (e.g., 6.0) and gradually increase
2. **Review recommendations** regularly and implement improvements
3. **Use appropriate models** - consider using CodeLlama for code-specific evaluations
4. **Use caching** - GitHub Actions caches dependencies to speed up runs
5. **Set appropriate timeouts** for your codebase size

## Integration

The evaluator integrates seamlessly with:
- GitHub Actions CI/CD
- Existing test suites
- Code coverage tools
- Documentation systems
- Built-in Ollama setup

## Related Documentation

- [CI Optimization](CI_OPTIMIZATION.md) - CI/CD pipeline optimization details
- [GitHub Models Integration](GITHUB_MODELS_INTEGRATION.md) - GitHub Models API integration
- [LLM Judge Flow](LLM_JUDGE_FLOW.md) - Detailed evaluation workflow
- [Testing Strategy](TESTING_STRATEGY.md) - Testing approach and methodology

## Support

For issues or questions:
1. Check the GitHub Actions logs
2. Review the `llm_judge_results.json` output
3. Verify Ollama is running and the model is available
4. Check the configuration in `evaluators/evaluator.config.json`

---

[← Back to Documentation](../README.md#documentation) | [Architecture →](ARCHITECTURE.md) | [Development →](DEVELOPMENT.md) 
