#!/usr/bin/env python3
"""
Smart LLM Judge Evaluator

This script automatically chooses the best backend:
- Ollama for local development (when available)
- OpenAI for remote/CI environments
- Fallback to OpenAI if Ollama fails

Usage:
    python evaluators/check_llm_judge_smart.py [--quick]

Environment Variables:
    OPENAI_API_KEY: OpenAI API key (required for remote/CI)
    LLM_JUDGE_THRESHOLD: Minimum score required (default: 7.0)
    LLM_JUDGE_FORCE_BACKEND: Force specific backend (OLLAMA/OPENAI)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add the parent directory to the path so we can import from app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basicchat.core.config import config

# Configuration
DEFAULT_THRESHOLD = 7.0
MAX_RETRIES = 3

class SmartLLMJudgeEvaluator:
    """Smart LLM-based code evaluator that automatically chooses the best backend"""
    
    def __init__(self, quick_mode: bool = False):
        self.threshold = float(os.getenv('LLM_JUDGE_THRESHOLD', DEFAULT_THRESHOLD))
        self.quick_mode = quick_mode
        self.force_backend = os.getenv('LLM_JUDGE_FORCE_BACKEND', '').upper()
        
        # Load evaluation rules
        self.rules = self.load_evaluation_rules()
        
        # Determine backend
        self.backend = self.determine_backend()
        self.evaluator = self.create_evaluator()
        
        # Initialize results
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'scores': {},
            'details': {},
            'recommendations': [],
            'overall_score': 0.0,
            'evaluation_mode': 'quick' if quick_mode else 'full',
            'rules_version': self.rules.get('version', '1.0.0'),
            'backend_used': self.backend,
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
    
    def determine_backend(self) -> str:
        """Determine the best backend to use"""
        # If backend is forced, use it
        if self.force_backend in ['OLLAMA', 'OPENAI']:
            print(f"🔧 Using forced backend: {self.force_backend}")
            return self.force_backend
        
        # Check if we're in a CI environment
        if self.is_ci_environment():
            print("🔧 CI environment detected, using OpenAI backend")
            return 'OPENAI'
        
        # Check if OpenAI API key is available
        if os.getenv('OPENAI_API_KEY'):
            print("🔧 OpenAI API key found")
            # Still try Ollama first for local development
            if self.test_ollama_connection():
                print("🔧 Ollama available, using Ollama backend")
                return 'OLLAMA'
            else:
                print("🔧 Ollama not available, using OpenAI backend")
                return 'OPENAI'
        
        # Try Ollama for local development
        if self.test_ollama_connection():
            print("🔧 Ollama available, using Ollama backend")
            return 'OLLAMA'
        
        # Fallback to OpenAI if API key is available
        if os.getenv('OPENAI_API_KEY'):
            print("🔧 Using OpenAI backend as fallback")
            return 'OPENAI'
        
        # No backend available
        raise Exception("No suitable backend available. Please ensure either Ollama is running or OPENAI_API_KEY is set.")
    
    def is_ci_environment(self) -> bool:
        """Check if we're running in a CI environment"""
        ci_indicators = [
            'CI', 'GITHUB_ACTIONS', 'GITLAB_CI', 'JENKINS_URL', 
            'TRAVIS', 'CIRCLECI', 'BUILDKITE', 'DRONE'
        ]
        return any(os.getenv(indicator) for indicator in ci_indicators)
    
    def test_ollama_connection(self) -> bool:
        """Test if Ollama is available and responding"""
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                # Check if mistral model is available
                models = response.json().get('models', [])
                return any('mistral' in model.get('name', '').lower() for model in models)
            return False
        except Exception:
            return False
    
    def create_evaluator(self):
        """Create the appropriate evaluator based on backend"""
        if self.backend == 'OLLAMA':
            from basicchat.evaluation.evaluators.check_llm_judge import LLMJudgeEvaluator
            return LLMJudgeEvaluator(quick_mode=self.quick_mode)
        elif self.backend == 'OPENAI':
            from basicchat.evaluation.evaluators.check_llm_judge_openai import OpenAIEvaluator
            return OpenAIEvaluator(quick_mode=self.quick_mode)
        else:
            raise Exception(f"Unknown backend: {self.backend}")
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run the complete evaluation process"""
        mode_text = "QUICK" if self.quick_mode else "FULL"
        print(f"🔍 Collecting codebase information ({mode_text} mode)...")
        
        # Use the appropriate evaluator's methods
        if hasattr(self.evaluator, 'collect_codebase_info'):
            codebase_info = self.evaluator.collect_codebase_info()
        else:
            # Fallback to basic info collection
            codebase_info = self.collect_basic_codebase_info()
        
        print("🤖 Generating evaluation prompt...")
        if hasattr(self.evaluator, 'generate_evaluation_prompt'):
            prompt = self.evaluator.generate_evaluation_prompt(codebase_info)
        else:
            prompt = self.generate_basic_evaluation_prompt(codebase_info)
        
        print(f"🧠 Running LLM evaluation with {self.backend}...")
        if hasattr(self.evaluator, 'evaluate_with_llm'):
            evaluation = self.evaluator.evaluate_with_llm(prompt)
        else:
            evaluation = self.evaluator.run_evaluation()
        
        # Store results
        self.results.update(evaluation)
        self.results['codebase_info'] = codebase_info
        
        return self.results
    
    def collect_basic_codebase_info(self) -> Dict[str, Any]:
        """Basic codebase info collection as fallback"""
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
        
        # Simple file counting
        for root, dirs, files in os.walk('.'):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['venv', '__pycache__', 'node_modules']]
            
            for file in files:
                if file.endswith(('.py', '.js', '.ts', '.jsx', '.tsx')):
                    info['file_count'] += 1
                    try:
                        with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            info['lines_of_code'] += len(lines)
                    except Exception:
                        pass
                
                if file.endswith(('.md', '.rst', '.txt')):
                    info['documentation_files'] += 1
        
        return info
    
    def generate_basic_evaluation_prompt(self, codebase_info: Dict[str, Any]) -> str:
        """Basic evaluation prompt as fallback"""
        mode_note = "QUICK EVALUATION MODE - Focus on critical issues only" if self.quick_mode else "FULL EVALUATION MODE"
        
        return f"""
You are an expert software engineer evaluating a Python codebase for quality, maintainability, and best practices.

{mode_note}

Codebase Information:
- Total files: {codebase_info['file_count']}
- Lines of code: {codebase_info['lines_of_code']}
- Documentation files: {codebase_info['documentation_files']}

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
    
    def print_results(self, results: Dict[str, Any]):
        """Print evaluation results in a readable format"""
        mode_text = "QUICK" if self.quick_mode else "FULL"
        print("\n" + "="*60)
        print(f"🤖 LLM JUDGE EVALUATION RESULTS ({self.backend}) - {mode_text} MODE")
        print("="*60)
        
        scores = results.get('scores', {})
        overall_score = results.get('overall_score', 0.0)
        backend_used = results.get('backend_used', self.backend)
        rules_version = results.get('rules_version', '1.0.0')
        
        print(f"\n📊 OVERALL SCORE: {overall_score:.1f}/10")
        print(f"🎯 THRESHOLD: {self.threshold}/10")
        print(f"📋 RULES VERSION: {rules_version}")
        print(f"🔧 BACKEND USED: {backend_used}")
        
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
            print(f"🚀 Starting Smart LLM Judge Evaluation - {mode_text} MODE...")
            print(f"🔧 Backend: {self.backend}")
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
    parser = argparse.ArgumentParser(description='Smart LLM Judge Evaluator')
    parser.add_argument('--quick', action='store_true', 
                       help='Run in quick mode for faster CI evaluation')
    args = parser.parse_args()
    
    try:
        evaluator = SmartLLMJudgeEvaluator(quick_mode=args.quick)
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
