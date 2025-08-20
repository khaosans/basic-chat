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
        
        # Load evaluation rules
        self.rules = self.load_evaluation_rules()
        
        # Initialize Ollama chat client
        self.ollama_chat = OllamaChat(model_name=self.model)
        
        # Initialize results
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'scores': {},
            'details': {},
            'recommendations': [],
            'overall_score': 0.0,
            'evaluation_mode': 'quick' if quick_mode else 'full',
            'rules_version': self.rules.get('version', '1.0.0'),
            'consistency_checks': {}
        }
    
    def load_evaluation_rules(self) -> Dict[str, Any]:
        """Load evaluation rules from configuration file"""
        rules_file = os.path.join(os.path.dirname(__file__), 'llm_judge_rules.json')
        try:
            with open(rules_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Rules file not found at {rules_file}, using defaults")
            return {
                "version": "1.0.0",
                "thresholds": {"overall_minimum": 7.0},
                "categories": {}
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
            'file_types': {},
            'complexity_metrics': {}
        }
        
        # Get file patterns from rules
        patterns = self.rules.get('file_patterns', {})
        include_extensions = patterns.get('include', ['.py', '.js', '.ts', '.jsx', '.tsx'])
        exclude_dirs = patterns.get('exclude', ['.git', 'venv', '__pycache__', 'node_modules'])
        doc_extensions = patterns.get('documentation', ['.md', '.rst', '.txt', '.adoc'])
        test_patterns = patterns.get('test_files', ['test_*', '*_test', '*test*'])
        
        # In quick mode, focus on key files only
        if self.quick_mode:
            key_files = [
                'main.py', 'basicchat/core/app.py', 'basicchat/core/config.py', 
                'pyproject.toml', 'README.md', 'basicchat/core/reasoning_engine.py', 
                'basicchat/services/document_processor.py'
            ]
            
            for file in key_files:
                if os.path.exists(file):
                    info['file_count'] += 1
                    try:
                        with open(file, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            info['lines_of_code'] += len(lines)
                            
                            # Count file types
                            ext = os.path.splitext(file)[1]
                            info['file_types'][ext] = info['file_types'].get(ext, 0) + 1
                    except Exception:
                        pass
            
            # Count test files in test directories
            test_dirs = ['tests/']
            for test_dir in test_dirs:
                if os.path.exists(test_dir):
                    for root, dirs, files in os.walk(test_dir):
                        dirs[:] = [d for d in dirs if d not in exclude_dirs]
                        for file in files:
                            if any(pattern.replace('*', '') in file for pattern in test_patterns):
                                info['test_files'] += 1
        else:
            # Full mode - scan entire codebase
            for root, dirs, files in os.walk('.'):
                # Skip excluded directories
                dirs[:] = [d for d in dirs if d not in exclude_dirs]
                
                for file in files:
                    if any(file.endswith(ext) for ext in include_extensions):
                        file_path = os.path.join(root, file)
                        info['file_count'] += 1
                        
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                lines = f.readlines()
                                info['lines_of_code'] += len(lines)
                                
                                # Count file types
                                ext = os.path.splitext(file)[1]
                                info['file_types'][ext] = info['file_types'].get(ext, 0) + 1
                        except Exception:
                            pass
                        
                        # Count test files
                        if any(pattern.replace('*', '') in file for pattern in test_patterns):
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
                if any(file.endswith(ext) for ext in doc_extensions):
                    info['documentation_files'] += 1
        
        # Get dependencies
        if os.path.exists('pyproject.toml'):
            try:
                with open('pyproject.toml', 'r') as f:
                    content = f.read()
                    # Simple parsing for dependencies
                    if '[tool.poetry.dependencies]' in content:
                        info['dependencies'] = ['poetry-managed']
            except Exception:
                pass
        
        return info
    
    def generate_evaluation_prompt(self, codebase_info: Dict[str, Any]) -> str:
        """Generate the evaluation prompt for the LLM using rules"""
        mode_note = "QUICK EVALUATION MODE - Focus on critical issues only" if self.quick_mode else "FULL EVALUATION MODE"
        
        # Get categories and rules from configuration
        categories = self.rules.get('categories', {})
        best_practices = self.rules.get('best_practices', {})
        
        prompt = f"""
You are an expert software engineer evaluating a Python codebase for quality, maintainability, and best practices.

{mode_note}

Codebase Information:
- Total files: {codebase_info['file_count']}
- Lines of code: {codebase_info['lines_of_code']}
- Test files: {codebase_info['test_files']}
- Test coverage: {codebase_info['test_coverage']:.1f}%
- Documentation files: {codebase_info['documentation_files']}
- File types: {codebase_info['file_types']}

Evaluation Rules and Standards:
"""

        # Add category-specific rules
        for category_name, category_config in categories.items():
            prompt += f"\n{category_name.replace('_', ' ').title()}:\n"
            rules = category_config.get('rules', [])
            for rule in rules:
                prompt += f"- {rule}\n"
            
            rubric = category_config.get('rubric', {})
            prompt += f"Rubric:\n"
            for score, description in rubric.items():
                prompt += f"- {score}: {description}\n"

        prompt += f"""

Python Best Practices:
"""
        for practice in best_practices.get('python', []):
            prompt += f"- {practice}\n"

        prompt += f"""

General Best Practices:
"""
        for practice in best_practices.get('general', []):
            prompt += f"- {practice}\n"

        prompt += f"""

Please evaluate the following aspects and provide scores from 1-10 (where 10 is excellent):

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
    ],
    "consistency_confidence": 0.95
}}
"""
        
        return prompt
    
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
                        result = json.loads(json_str)
                        
                        # Validate consistency
                        if self.rules.get('consistency_checks', {}).get('enabled', False):
                            result = self.validate_consistency(result)
                        
                        return result
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
    
    def validate_consistency(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate evaluation consistency"""
        scores = result.get('scores', {})
        overall_score = result.get('overall_score', 0.0)
        
        # Calculate weighted average
        categories = self.rules.get('categories', {})
        weighted_sum = 0.0
        total_weight = 0.0
        
        for category_name, category_config in categories.items():
            if category_name in scores:
                weight = category_config.get('weight', 1.0)
                score = scores[category_name].get('score', 0) if isinstance(scores[category_name], dict) else scores[category_name]
                weighted_sum += score * weight
                total_weight += weight
        
        if total_weight > 0:
            calculated_score = weighted_sum / total_weight
            score_diff = abs(calculated_score - overall_score)
            
            if score_diff > self.rules.get('consistency_checks', {}).get('max_score_variance', 1.0):
                print(f"Warning: Score inconsistency detected. Calculated: {calculated_score:.2f}, Reported: {overall_score:.2f}")
                result['overall_score'] = calculated_score
                result['consistency_confidence'] = max(0.5, 1.0 - score_diff)
            else:
                result['consistency_confidence'] = 1.0
        
        return result
    
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
        consistency_confidence = results.get('consistency_confidence', 1.0)
        rules_version = results.get('rules_version', '1.0.0')
        
        print(f"\n📊 OVERALL SCORE: {overall_score:.1f}/10")
        print(f"🎯 THRESHOLD: {self.threshold}/10")
        print(f"📋 RULES VERSION: {rules_version}")
        print(f"✅ CONSISTENCY CONFIDENCE: {consistency_confidence:.2f}")
        
        print("\n📈 DETAILED SCORES:")
        for category, data in scores.items():
            if isinstance(data, dict):
                score = data.get('score', 0)
                justification = data.get('justification', 'No justification provided')
                print(f"  {category.replace('_', ' ').title()}: {score}/10")
                print(f"    {justification}")
        
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
            print(f"📋 Rules version: {self.rules.get('version', '1.0.0')}")
            
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
