#!/usr/bin/env python3
"""
LLM Judge Evaluator using OpenAI API

This script evaluates the codebase using OpenAI's API to assess:
- Code quality and maintainability
- Test coverage and effectiveness
- Documentation quality
- Overall project health

Uses OpenAI's cost-effective models for reliable evaluation.

Usage:
    python evaluators/check_llm_judge_openai.py [--quick] [--model gpt-3.5-turbo]

Environment Variables:
    OPENAI_API_KEY: OpenAI API key
    OPENAI_MODEL: OpenAI model to use (default: gpt-3.5-turbo)
    LLM_JUDGE_THRESHOLD: Minimum score required (default: 7.0)
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import requests

# Add the parent directory to the path so we can import from app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basicchat.core.config import config
from basicchat.evaluation.evaluators.consistency import LLMJudgeConsistency

# Configuration
DEFAULT_THRESHOLD = 7.0
DEFAULT_MODEL = "gpt-3.5-turbo"
MAX_RETRIES = 3

# Cost-effective model options
COST_EFFECTIVE_MODELS = {
    "gpt-3.5-turbo": {
        "cost_per_1k_tokens": 0.0015,  # $0.0015 per 1K input tokens
        "quality": "Good",
        "speed": "Fast",
        "max_tokens": 4096
    },
    "gpt-3.5-turbo-16k": {
        "cost_per_1k_tokens": 0.003,   # $0.003 per 1K input tokens
        "quality": "Good",
        "speed": "Fast",
        "max_tokens": 16384
    },
    "gpt-4": {
        "cost_per_1k_tokens": 0.03,    # $0.03 per 1K input tokens
        "quality": "Excellent",
        "speed": "Medium",
        "max_tokens": 8192
    },
    "gpt-4-turbo": {
        "cost_per_1k_tokens": 0.01,    # $0.01 per 1K input tokens
        "quality": "Excellent",
        "speed": "Fast",
        "max_tokens": 128000
    }
}

class OpenAIEvaluator:
    """LLM-based code evaluator using OpenAI API"""
    
    def __init__(self, quick_mode: bool = False, model: str = None):
        self.model = model or os.getenv('OPENAI_MODEL', DEFAULT_MODEL)
        self.threshold = float(os.getenv('LLM_JUDGE_THRESHOLD', DEFAULT_THRESHOLD))
        self.quick_mode = quick_mode
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.consistency = LLMJudgeConsistency()
        
        # OpenAI API endpoint
        self.api_url = "https://api.openai.com/v1/chat/completions"
        
        # Validate model
        if self.model not in COST_EFFECTIVE_MODELS:
            print(f"⚠️  Warning: Model {self.model} not in cost-effective list. Using {DEFAULT_MODEL}")
            self.model = DEFAULT_MODEL
        
        # Initialize results
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'scores': {},
            'details': {},
            'recommendations': [],
            'overall_score': 0.0,
            'evaluation_mode': 'quick' if quick_mode else 'full',
            'model_used': self.model,
            'estimated_cost': 0.0
        }
    
    def collect_codebase_info(self) -> Dict[str, Any]:
        """Collect information about the codebase for evaluation"""
        info = {
            'file_count': 0,
            'lines_of_code': 0,
            'test_files': 0,
            'test_coverage': 0.0,
            'documentation_files': 0,
            'dependencies': [],
            'recent_changes': [],
            'test_analysis': {
                'test_files_content': [],
                'test_functions': [],
                'test_categories': {},
                'test_coverage_details': {}
            }
        }
        
        # In quick mode, focus on key files only
        if self.quick_mode:
            key_files = [
                'app.py', 'config.py', 'requirements.txt', 'README.md',
                'reasoning_engine.py', 'document_processor.py'
            ]
            test_dirs = ['tests/']
            
            for file in key_files:
                if os.path.exists(file):
                    info['file_count'] += 1
                    try:
                        with open(file, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            info['lines_of_code'] += len(lines)
                    except Exception:
                        pass
            
            # Enhanced test directory analysis
            for test_dir in test_dirs:
                if os.path.exists(test_dir):
                    for root, dirs, files in os.walk(test_dir):
                        for file in files:
                            if file.endswith('.py') and ('test' in file.lower() or file.startswith('test_')):
                                info['test_files'] += 1
                                file_path = os.path.join(root, file)
                                
                                # Read and analyze test file content
                                try:
                                    with open(file_path, 'r', encoding='utf-8') as f:
                                        content = f.read()
                                        lines = content.split('\n')
                                        
                                        # Extract test functions
                                        test_functions = []
                                        test_categories = set()
                                        
                                        for i, line in enumerate(lines):
                                            # Find test function definitions
                                            if line.strip().startswith('def test_'):
                                                func_name = line.strip().split('def ')[1].split('(')[0]
                                                test_functions.append({
                                                    'name': func_name,
                                                    'file': file,
                                                    'line': i + 1
                                                })
                                            
                                            # Find test categories/markers
                                            if '@pytest.mark.' in line:
                                                marker = line.strip().split('@pytest.mark.')[1].split('(')[0]
                                                test_categories.add(marker)
                                        
                                        # Store test file analysis
                                        info['test_analysis']['test_files_content'].append({
                                            'file': file,
                                            'path': file_path,
                                            'lines': len(lines),
                                            'test_functions': test_functions,
                                            'categories': list(test_categories),
                                            'content_preview': content[:1000] + '...' if len(content) > 1000 else content
                                        })
                                        
                                        # Update test categories
                                        for category in test_categories:
                                            if category not in info['test_analysis']['test_categories']:
                                                info['test_analysis']['test_categories'][category] = 0
                                            info['test_analysis']['test_categories'][category] += 1
                                        
                                except Exception as e:
                                    print(f"Warning: Could not read test file {file_path}: {e}")
        else:
            # Full mode - scan entire codebase
            for root, dirs, files in os.walk('.'):
                # Skip common directories
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['venv', '__pycache__', 'node_modules']]
                
                for file in files:
                    if file.endswith(('.py', '.js', '.ts', '.jsx', '.tsx')):
                        file_path = os.path.join(root, file)
                        info['file_count'] += 1
                        
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                lines = f.readlines()
                                info['lines_of_code'] += len(lines)
                        except Exception:
                            pass
                        
                        # Enhanced test file analysis for full mode
                        if 'test' in file.lower() or file.startswith('test_'):
                            info['test_files'] += 1
                            
                            # Read and analyze test file content
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                    lines = content.split('\n')
                                    
                                    # Extract test functions
                                    test_functions = []
                                    test_categories = set()
                                    
                                    for i, line in enumerate(lines):
                                        # Find test function definitions
                                        if line.strip().startswith('def test_'):
                                            func_name = line.strip().split('def ')[1].split('(')[0]
                                            test_functions.append({
                                                'name': func_name,
                                                'file': file,
                                                'line': i + 1
                                            })
                                        
                                        # Find test categories/markers
                                        if '@pytest.mark.' in line:
                                            marker = line.strip().split('@pytest.mark.')[1].split('(')[0]
                                            test_categories.add(marker)
                                    
                                    # Store test file analysis
                                    info['test_analysis']['test_files_content'].append({
                                        'file': file,
                                        'path': file_path,
                                        'lines': len(lines),
                                        'test_functions': test_functions,
                                        'categories': list(test_categories),
                                        'content_preview': content[:1000] + '...' if len(content) > 1000 else content
                                    })
                                    
                                    # Update test categories
                                    for category in test_categories:
                                        if category not in info['test_analysis']['test_categories']:
                                            info['test_analysis']['test_categories'][category] = 0
                                        info['test_analysis']['test_categories'][category] += 1
                                    
                            except Exception as e:
                                print(f"Warning: Could not read test file {file_path}: {e}")
        
        # Get test coverage if available (skip in quick mode for speed)
        if not self.quick_mode:
            try:
                result = subprocess.run(['python', '-m', 'pytest', '--cov=.', '--cov-report=json'], 
                                      capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    coverage_data = json.loads(result.stdout)
                    if coverage_data and 'totals' in coverage_data:
                        info['test_coverage'] = coverage_data['totals'].get('percent', 0.0)
                        info['test_analysis']['test_coverage_details'] = coverage_data
            except Exception:
                pass
        
        # Count documentation files
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith(('.md', '.rst', '.txt')):
                    info['documentation_files'] += 1
        
        # Get dependencies
        if os.path.exists('requirements.txt'):
            with open('requirements.txt', 'r') as f:
                info['dependencies'] = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        return info
    
    def generate_evaluation_prompt(self, codebase_info: Dict[str, Any]) -> str:
        """Generate the evaluation prompt for the LLM, injecting rubric and version for consistency"""
        mode_note = "QUICK EVALUATION MODE - Focus on critical issues only" if self.quick_mode else "FULL EVALUATION MODE"
        rubric_md = self.consistency.rubric_text()
        version = self.consistency.version
        
        # Build detailed test analysis section
        test_analysis = codebase_info.get('test_analysis', {})
        test_files_content = test_analysis.get('test_files_content', [])
        test_categories = test_analysis.get('test_categories', {})
        
        test_analysis_text = f"""
TEST ANALYSIS:
- Test files found: {len(test_files_content)}
- Test categories: {dict(test_categories)}
- Total test functions: {sum(len(tf.get('test_functions', [])) for tf in test_files_content)}

Test Files Details:"""
        
        for tf in test_files_content:
            test_analysis_text += f"""
  - {tf['file']} ({tf['lines']} lines)
    - Test functions: {len(tf['test_functions'])}
    - Categories: {tf['categories']}
    - Functions: {[f['name'] for f in tf['test_functions']]}
    - Content preview: {tf['content_preview'][:200]}..."""
        
        return f"""
You are an expert software engineer evaluating a Python codebase for quality, maintainability, and best practices.

LLM Judge Rubric Version: {version}

{mode_note}

Codebase Information:
- Total files: {codebase_info['file_count']}
- Lines of code: {codebase_info['lines_of_code']}
- Test files: {codebase_info['test_files']}
- Test coverage: {codebase_info['test_coverage']:.1f}%
- Documentation files: {codebase_info['documentation_files']}
- Dependencies: {len(codebase_info['dependencies'])} packages

{test_analysis_text}

Please evaluate the following aspects and provide scores from 1-10 (where 10 is excellent):

{rubric_md}

**IMPORTANT: Pay special attention to test coverage and quality based on the detailed test analysis above.**

For each category, provide:
- Score (1-10)
- Brief justification
- Specific, actionable recommendations for improvement

**For recommendations:**
- List each as a single, clear, actionable checklist item
- Reference specific files, functions, or areas of the codebase if possible
- Prioritize the most impactful actions first
- Use concise, direct language
- If relevant, provide a brief example or filename
- For test coverage issues, reference specific test files and functions that need improvement

Respond in the following JSON format:
{{
    "scores": {{
        "code_quality": {{"score": 8, "justification": "Well-structured code with good naming conventions"}},
        "test_coverage": {{"score": 7, "justification": "Good test coverage but could be more comprehensive"}},
        "documentation": {{"score": 6, "justification": "Basic documentation present but could be enhanced"}},
        "architecture": {{"score": 8, "justification": "Clean modular design with good separation of concerns"}},
        "security": {{"score": 7, "justification": "No obvious security issues, follows basic security practices"}},
        "performance": {{"score": 7, "justification": "Generally efficient code with room for optimization"}}
    }},
    "overall_score": 7.2,
    "recommendations": [
        "[tests/test_core.py] Add unit tests for edge cases in ReasoningEngine.",
        "[README.md] Expand documentation with usage examples and setup instructions.",
        "[app.py] Refactor function names to follow snake_case convention.",
        "[config.py] Add type hints to all functions.",
        "[ALL] Run a security audit for dependency vulnerabilities."
    ]
}}
"""
    
    def estimate_cost(self, prompt: str, response: str) -> float:
        """Estimate the cost of the API call"""
        # Rough token estimation (1 token ≈ 4 characters for English text)
        input_tokens = len(prompt) / 4
        output_tokens = len(response) / 4
        
        model_info = COST_EFFECTIVE_MODELS[self.model]
        input_cost = (input_tokens / 1000) * model_info['cost_per_1k_tokens']
        output_cost = (output_tokens / 1000) * model_info['cost_per_1k_tokens'] * 2  # Output is typically more expensive
        
        return input_cost + output_cost
    
    def evaluate_with_openai(self, prompt: str) -> Dict[str, Any]:
        """Evaluate the codebase using OpenAI API"""
        if not self.api_key:
            raise Exception("OPENAI_API_KEY environment variable is required")
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        model_info = COST_EFFECTIVE_MODELS[self.model]
        
        payload = {
            'model': self.model,
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are an expert software engineer evaluating code quality. Provide detailed, actionable feedback in JSON format.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'max_tokens': min(2000, model_info['max_tokens']),
            'temperature': 0.1
        }
        
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    
                    # Estimate cost
                    estimated_cost = self.estimate_cost(prompt, content)
                    self.results['estimated_cost'] = estimated_cost
                    
                    # Try to parse JSON response
                    try:
                        # Find JSON in the response
                        start = content.find('{')
                        end = content.rfind('}') + 1
                        if start != -1 and end != 0:
                            json_str = content[start:end]
                            return json.loads(json_str)
                        else:
                            raise ValueError("No JSON found in response")
                    except json.JSONDecodeError as e:
                        print(f"Failed to parse JSON response (attempt {attempt + 1}): {e}")
                        print(f"Response: {content}")
                        if attempt == MAX_RETRIES - 1:
                            raise
                        continue
                else:
                    print(f"OpenAI API call failed (attempt {attempt + 1}): {response.status_code}")
                    print(f"Response: {response.text}")
                    if attempt == MAX_RETRIES - 1:
                        raise Exception(f"OpenAI API failed with status {response.status_code}")
                    continue
                    
            except requests.exceptions.RequestException as e:
                print(f"OpenAI API request failed (attempt {attempt + 1}): {e}")
                if attempt == MAX_RETRIES - 1:
                    raise
                continue
        
        raise Exception("Failed to get valid response from OpenAI after all retries")
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run the complete evaluation process"""
        mode_text = "QUICK" if self.quick_mode else "FULL"
        print(f"🔍 Collecting codebase information ({mode_text} mode)...")
        codebase_info = self.collect_codebase_info()
        
        print("🤖 Generating evaluation prompt...")
        prompt = self.generate_evaluation_prompt(codebase_info)
        
        print(f"🧠 Running LLM evaluation with OpenAI Model: {self.model}...")
        evaluation = self.evaluate_with_openai(prompt)
        
        # Store results
        self.results.update(evaluation)
        self.results['codebase_info'] = codebase_info
        
        return self.results
    
    def print_results(self, results: Dict[str, Any]):
        """Print evaluation results in a readable format"""
        mode_text = "QUICK" if self.quick_mode else "FULL"
        print("\n" + "="*60)
        print(f"🤖 LLM JUDGE EVALUATION RESULTS (OpenAI) - {mode_text} MODE")
        print("="*60)
        
        scores = results.get('scores', {})
        overall_score = results.get('overall_score', 0.0)
        estimated_cost = results.get('estimated_cost', 0.0)
        
        print(f"\n📊 OVERALL SCORE: {overall_score:.1f}/10")
        print(f"🤖 MODEL USED: {self.model}")
        print(f"💰 ESTIMATED COST: ${estimated_cost:.4f}")
        
        model_info = COST_EFFECTIVE_MODELS[self.model]
        print(f"📈 MODEL INFO: {model_info['quality']} quality, {model_info['speed']} speed")
        
        print("\n📈 DETAILED SCORES:")
        for category, data in scores.items():
            if isinstance(data, dict):
                score = data.get('score', 0)
                justification = data.get('justification', 'No justification provided')
                print(f"  {category.replace('_', ' ').title()}: {score}/10")
                print(f"    {justification}")
        
        print(f"\n🎯 THRESHOLD: {self.threshold}/10")
        
        if overall_score >= self.threshold:
            print("✅ EVALUATION PASSED")
            status = "PASS"
        else:
            print("❌ EVALUATION FAILED")
            status = "FAIL"
        
        recommendations = results.get('recommendations', [])
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        # Save results to file
        output_file = "llm_judge_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📄 Results saved to: {output_file}")
        
        return status, overall_score
    
    def run(self) -> int:
        """Main execution method"""
        try:
            mode_text = "QUICK" if self.quick_mode else "FULL"
            print(f"🚀 Starting LLM Judge Evaluation (OpenAI) - {mode_text} MODE...")
            print(f"📋 Using model: {self.model}")
            print(f"🔗 OpenAI API: {self.api_url}")
            
            if not self.api_key:
                print("❌ OPENAI_API_KEY environment variable is required")
                return 1
            
            # Show model info
            model_info = COST_EFFECTIVE_MODELS[self.model]
            print(f"💰 Cost per 1K tokens: ${model_info['cost_per_1k_tokens']}")
            print(f"📈 Quality: {model_info['quality']}, Speed: {model_info['speed']}")
            
            results = self.run_evaluation()
            status, score = self.print_results(results)
            
            if status == "FAIL":
                print(f"\n❌ Evaluation failed: Score {score:.1f} is below threshold {self.threshold}")
                return 1
            else:
                print(f"\n✅ Evaluation passed: Score {score:.1f} meets threshold {self.threshold}")
                return 0
                
        except Exception as e:
            print(f"❌ Evaluation failed with error: {e}")
            return 1

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='LLM Judge Evaluator using OpenAI API')
    parser.add_argument('--quick', action='store_true', 
                       help='Run in quick mode for faster CI evaluation')
    parser.add_argument('--model', type=str, default=None,
                       help=f'OpenAI model to use (default: {DEFAULT_MODEL})')
    args = parser.parse_args()
    
    try:
        evaluator = OpenAIEvaluator(quick_mode=args.quick, model=args.model)
        exit_code = evaluator.run()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️ Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
