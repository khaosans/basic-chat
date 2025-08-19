#!/usr/bin/env python3
"""
LLM Judge Evaluator for GitHub Actions CI

This script evaluates the codebase using an LLM to assess:
- Code quality and maintainability
- Test coverage and effectiveness
- Documentation quality
- Overall project health

Uses the built-in Ollama setup instead of external APIs.

Usage:
    python evaluators/check_llm_judge.py [--quick]

Environment Variables:
    OLLAMA_API_URL: Ollama API URL (default: http://localhost:11434/api)
    OLLAMA_MODEL: Ollama model to use (default: mistral)
    LLM_JUDGE_THRESHOLD: Minimum score required (default: 7.0)
"""

import os
import sys
import json
import subprocess
import tempfile
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime

# Add the parent directory to the path so we can import from app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basicchat.core.app import OllamaChat
from basicchat.core.config import config

# Configuration
DEFAULT_THRESHOLD = 7.0
DEFAULT_MODEL = "mistral"
MAX_RETRIES = 3

class LLMJudgeEvaluator:
    """LLM-based code evaluator for CI/CD pipelines using built-in Ollama"""
    
    def __init__(self, quick_mode: bool = False):
        self.ollama_url = os.getenv('OLLAMA_API_URL', 'http://localhost:11434/api')
        self.model = os.getenv('OLLAMA_MODEL', DEFAULT_MODEL)
        self.threshold = float(os.getenv('LLM_JUDGE_THRESHOLD', DEFAULT_THRESHOLD))
        self.quick_mode = quick_mode
        
        # Initialize Ollama chat client
        self.ollama_chat = OllamaChat(model_name=self.model)
        
        # Initialize results
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'scores': {},
            'details': {},
            'recommendations': [],
            'overall_score': 0.0,
            'evaluation_mode': 'quick' if quick_mode else 'full'
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
            
            # Count test files in test directories
            for test_dir in test_dirs:
                if os.path.exists(test_dir):
                    for root, dirs, files in os.walk(test_dir):
                        for file in files:
                            if file.endswith('.py') and ('test' in file.lower() or file.startswith('test_')):
                                info['test_files'] += 1
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
                        
                        # Count test files
                        if 'test' in file.lower() or file.startswith('test_'):
                            info['test_files'] += 1
        
        # Get test coverage if available (skip in quick mode for speed)
        if not self.quick_mode:
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
        mode_note = "QUICK EVALUATION MODE - Focus on critical issues only" if self.quick_mode else "FULL EVALUATION MODE"
        
        return f"""
You are an expert software engineer evaluating a Python codebase for quality, maintainability, and best practices.

{mode_note}

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

{"In QUICK MODE, focus on major issues and provide brief justifications." if self.quick_mode else "Provide detailed analysis with specific examples."}

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
        """Evaluate the codebase using built-in Ollama"""
        for attempt in range(MAX_RETRIES):
            try:
                # Use the built-in OllamaChat
                payload = {"inputs": prompt}
                response = self.ollama_chat.query(payload)
                
                if not response:
                    raise Exception("No response received from Ollama")
                
                # Try to parse JSON response
                try:
                    # Find JSON in the response
                    start = response.find('{')
                    end = response.rfind('}') + 1
                    if start != -1 and end != 0:
                        json_str = response[start:end]
                        return json.loads(json_str)
                    else:
                        raise ValueError("No JSON found in response")
                except json.JSONDecodeError as e:
                    print(f"Failed to parse JSON response (attempt {attempt + 1}): {e}")
                    print(f"Response: {response}")
                    if attempt == MAX_RETRIES - 1:
                        raise
                    continue
                    
            except Exception as e:
                print(f"Ollama API call failed (attempt {attempt + 1}): {e}")
                if attempt == MAX_RETRIES - 1:
                    raise
                continue
        
        raise Exception("Failed to get valid response from LLM after all retries")
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run the complete evaluation process"""
        mode_text = "QUICK" if self.quick_mode else "FULL"
        print(f"🔍 Collecting codebase information ({mode_text} mode)...")
        codebase_info = self.collect_codebase_info()
        
        print("🤖 Generating evaluation prompt...")
        prompt = self.generate_evaluation_prompt(codebase_info)
        
        print("🧠 Running LLM evaluation with Ollama...")
        evaluation = self.evaluate_with_llm(prompt)
        
        # Store results
        self.results.update(evaluation)
        self.results['codebase_info'] = codebase_info
        
        return self.results
    
    def print_results(self, results: Dict[str, Any]):
        """Print evaluation results in a readable format"""
        mode_text = "QUICK" if self.quick_mode else "FULL"
        print("\n" + "="*60)
        print(f"🤖 LLM JUDGE EVALUATION RESULTS (Ollama) - {mode_text} MODE")
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
            mode_text = "QUICK" if self.quick_mode else "FULL"
            print(f"🚀 Starting LLM Judge Evaluation (Ollama) - {mode_text} MODE...")
            print(f"📋 Using model: {self.model}")
            print(f"🔗 Ollama URL: {self.ollama_url}")
            
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
    parser = argparse.ArgumentParser(description='LLM Judge Evaluator')
    parser.add_argument('--quick', action='store_true', 
                       help='Run in quick mode for faster CI evaluation')
    args = parser.parse_args()
    
    try:
        evaluator = LLMJudgeEvaluator(quick_mode=args.quick)
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
