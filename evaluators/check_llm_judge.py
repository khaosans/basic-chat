#!/usr/bin/env python3
"""
LLM Judge Evaluator for GitHub Actions CI

This script evaluates the codebase using an LLM to assess:
- Code quality and maintainability
- Test coverage and effectiveness
- Documentation quality
- Overall project health

Usage:
    python evaluators/check_llm_judge.py

Environment Variables:
    OPENAI_API_KEY: OpenAI API key (required)
    LLM_JUDGE_THRESHOLD: Minimum score required (default: 7.0)
    LLM_JUDGE_MODEL: OpenAI model to use (default: gpt-4)
"""

import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
import openai
from datetime import datetime

# Configuration
DEFAULT_THRESHOLD = 7.0
DEFAULT_MODEL = "gpt-4"
MAX_RETRIES = 3

class LLMJudgeEvaluator:
    """LLM-based code evaluator for CI/CD pipelines"""
    
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.threshold = float(os.getenv('LLM_JUDGE_THRESHOLD', DEFAULT_THRESHOLD))
        self.model = os.getenv('LLM_JUDGE_MODEL', DEFAULT_MODEL)
        
        openai.api_key = self.api_key
        
        # Initialize results
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'scores': {},
            'details': {},
            'recommendations': [],
            'overall_score': 0.0
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
            'recent_changes': []
        }
        
        # Count files and lines
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
                    
                    # Count test files
                    if 'test' in file.lower() or file.startswith('test_'):
                        info['test_files'] += 1
        
        # Get test coverage if available
        try:
            result = subprocess.run(['python', '-m', 'pytest', '--cov=.', '--cov-report=json'], 
                                  capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                coverage_data = json.loads(result.stdout)
                if coverage_data and 'totals' in coverage_data:
                    info['test_coverage'] = coverage_data['totals'].get('percent', 0.0)
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
        """Generate the evaluation prompt for the LLM"""
        return f"""
You are an expert software engineer evaluating a Python codebase for quality, maintainability, and best practices.

Codebase Information:
- Total files: {codebase_info['file_count']}
- Lines of code: {codebase_info['lines_of_code']}
- Test files: {codebase_info['test_files']}
- Test coverage: {codebase_info['test_coverage']:.1f}%
- Documentation files: {codebase_info['documentation_files']}
- Dependencies: {len(codebase_info['dependencies'])} packages

Please evaluate the following aspects and provide scores from 1-10 (where 10 is excellent):

1. **Code Quality** (1-10): Assess code structure, naming conventions, complexity, and adherence to Python best practices
2. **Test Coverage** (1-10): Evaluate test comprehensiveness, quality, and effectiveness
3. **Documentation** (1-10): Assess README quality, inline documentation, and overall project documentation
4. **Architecture** (1-10): Evaluate overall design patterns, modularity, and scalability
5. **Security** (1-10): Assess potential security vulnerabilities and best practices
6. **Performance** (1-10): Evaluate code efficiency and optimization opportunities

For each category, provide:
- Score (1-10)
- Brief justification
- Specific recommendations for improvement

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
        "Add more comprehensive integration tests",
        "Enhance API documentation with examples",
        "Consider adding type hints throughout the codebase"
    ]
}}
"""
    
    def evaluate_with_llm(self, prompt: str) -> Dict[str, Any]:
        """Evaluate the codebase using OpenAI's API"""
        for attempt in range(MAX_RETRIES):
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert software engineer evaluating code quality. Provide detailed, constructive feedback."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=2000
                )
                
                content = response.choices[0].message.content
                
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
                    
            except Exception as e:
                print(f"API call failed (attempt {attempt + 1}): {e}")
                if attempt == MAX_RETRIES - 1:
                    raise
                continue
        
        raise Exception("Failed to get valid response from LLM after all retries")
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run the complete evaluation process"""
        print("🔍 Collecting codebase information...")
        codebase_info = self.collect_codebase_info()
        
        print("🤖 Generating evaluation prompt...")
        prompt = self.generate_evaluation_prompt(codebase_info)
        
        print("🧠 Running LLM evaluation...")
        evaluation = self.evaluate_with_llm(prompt)
        
        # Store results
        self.results.update(evaluation)
        self.results['codebase_info'] = codebase_info
        
        return self.results
    
    def print_results(self, results: Dict[str, Any]):
        """Print evaluation results in a readable format"""
        print("\n" + "="*60)
        print("🤖 LLM JUDGE EVALUATION RESULTS")
        print("="*60)
        
        scores = results.get('scores', {})
        overall_score = results.get('overall_score', 0.0)
        
        print(f"\n📊 OVERALL SCORE: {overall_score:.1f}/10")
        
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
            print("🚀 Starting LLM Judge Evaluation...")
            
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
    try:
        evaluator = LLMJudgeEvaluator()
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