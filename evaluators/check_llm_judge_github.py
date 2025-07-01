#!/usr/bin/env python3
"""
LLM Judge Evaluator using GitHub Models API

This script evaluates the codebase using GitHub Models (via Azure AI Inference SDK) to assess:
- Code quality and maintainability
- Test coverage and effectiveness
- Documentation quality
- Overall project health

Uses GitHub Models for free evaluation with rate limits.

Usage:
    python evaluators/check_llm_judge_github.py [--quick] [--model deepseek/DeepSeek-V3-0324]

Environment Variables:
    GITHUB_TOKEN: GitHub personal access token with models:read permissions
    GITHUB_MODEL: GitHub model to use (default: deepseek/DeepSeek-V3-0324)
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
import time

# Add the parent directory to the path so we can import from app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config

# Configuration
DEFAULT_THRESHOLD = 7.0
DEFAULT_MODEL = "microsoft/phi-3.5-mini"
MAX_RETRIES = 3
RATE_LIMIT_DELAY = 60  # seconds to wait if rate limited

# GitHub Models available for evaluation
GITHUB_MODELS = {
    "deepseek/DeepSeek-V3-0324": {
        "type": "High",
        "quality": "Excellent",
        "speed": "Fast",
        "max_tokens": 8000,
        "rate_limit": "10 requests/minute, 50 requests/day"
    },
    "deepseek/deepseek-coder-33b-instruct": {
        "type": "High",
        "quality": "Excellent",
        "speed": "Fast",
        "max_tokens": 8000,
        "rate_limit": "10 requests/minute, 50 requests/day"
    },
    "deepseek/deepseek-coder-6.7b-instruct": {
        "type": "High", 
        "quality": "Excellent",
        "speed": "Fast",
        "max_tokens": 8000,
        "rate_limit": "10 requests/minute, 50 requests/day"
    },
    "microsoft/phi-3.5": {
        "type": "Low",
        "quality": "Good",
        "speed": "Very Fast",
        "max_tokens": 8000,
        "rate_limit": "15 requests/minute, 150 requests/day"
    },
    "microsoft/phi-3.5-mini": {
        "type": "Low",
        "quality": "Good", 
        "speed": "Very Fast",
        "max_tokens": 8000,
        "rate_limit": "15 requests/minute, 150 requests/day"
    },
    "microsoft/phi-2": {
        "type": "Low",
        "quality": "Good",
        "speed": "Very Fast", 
        "max_tokens": 8000,
        "rate_limit": "15 requests/minute, 150 requests/day"
    }
}

class GitHubModelsEvaluator:
    """LLM-based code evaluator using GitHub Models API"""
    
    def __init__(self, quick_mode: bool = False, model: str = None):
        self.model = model or os.getenv('GITHUB_MODEL', DEFAULT_MODEL)
        self.threshold = float(os.getenv('LLM_JUDGE_THRESHOLD', DEFAULT_THRESHOLD))
        self.quick_mode = quick_mode
        self.token = os.getenv('GITHUB_TOKEN')
        
        # GitHub Models API endpoint
        self.endpoint = "https://models.github.ai/inference"
        
        # Validate model
        if self.model not in GITHUB_MODELS:
            print(f"⚠️  Warning: Model {self.model} not in supported list. Using {DEFAULT_MODEL}")
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
            'api_calls': 0,
            'rate_limited': False
        }
        
        # Initialize Azure AI client
        try:
            from azure.ai.inference import ChatCompletionsClient
            from azure.core.credentials import AzureKeyCredential
            
            self.client = ChatCompletionsClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.token),
            )
            print(f"✅ GitHub Models client initialized with model: {self.model}")
        except ImportError:
            print("❌ Azure AI Inference SDK not installed. Install with: pip install azure-ai-inference")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Failed to initialize GitHub Models client: {e}")
            sys.exit(1)
    
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

1. Code Quality (structure, readability, best practices)
2. Test Coverage (adequacy and quality of tests)
3. Documentation (completeness and clarity)
4. Architecture (design patterns, modularity)
5. Security (potential vulnerabilities, best practices)
6. Performance (efficiency, optimization opportunities)
7. Maintainability (ease of future development)
8. Overall Project Health

For each category, provide:
- Score (1-10)
- Brief explanation
- Specific recommendations for improvement

Respond in JSON format:
{{
    "scores": {{
        "code_quality": {{"score": X, "explanation": "...", "recommendations": ["..."]}},
        "test_coverage": {{"score": X, "explanation": "...", "recommendations": ["..."]}},
        "documentation": {{"score": X, "explanation": "...", "recommendations": ["..."]}},
        "architecture": {{"score": X, "explanation": "...", "recommendations": ["..."]}},
        "security": {{"score": X, "explanation": "...", "recommendations": ["..."]}},
        "performance": {{"score": X, "explanation": "...", "recommendations": ["..."]}},
        "maintainability": {{"score": X, "explanation": "...", "recommendations": ["..."]}},
        "overall_health": {{"score": X, "explanation": "...", "recommendations": ["..."]}}
    }},
    "overall_score": X.X,
    "summary": "...",
    "critical_issues": ["..."],
    "next_steps": ["..."]
}}
"""
    
    def evaluate_with_github_models(self, prompt: str) -> Dict[str, Any]:
        """Evaluate using GitHub Models API with retry logic"""
        from azure.ai.inference.models import SystemMessage, UserMessage
        
        # Try multiple models in order of preference
        models_to_try = [
            self.model,
            "microsoft/phi-3.5-mini",
            "microsoft/phi-3.5",
            "microsoft/phi-2"
        ]
        
        for model_to_try in models_to_try:
            print(f"🔄 Trying model: {model_to_try}")
            
            for attempt in range(MAX_RETRIES):
                try:
                    print(f"🔄 Attempt {attempt + 1}/{MAX_RETRIES} - Calling GitHub Models API...")
                    
                    response = self.client.complete(
                        messages=[
                            SystemMessage("You are an expert software engineer evaluating code quality. Respond only with valid JSON."),
                            UserMessage(prompt),
                        ],
                        temperature=0.3,  # Lower temperature for more consistent evaluation
                        top_p=0.9,
                        max_tokens=2000,
                        model=model_to_try
                    )
                    
                    self.results['api_calls'] += 1
                    self.results['model_used'] = model_to_try
                    
                    # Extract response content
                    content = response.choices[0].message.content.strip()
                    
                    # Try to parse JSON response
                    try:
                        # Clean up the response if it has markdown formatting
                        if content.startswith('```json'):
                            content = content[7:]
                        if content.endswith('```'):
                            content = content[:-3]
                        
                        result = json.loads(content)
                        print(f"✅ Successfully parsed GitHub Models response with model: {model_to_try}")
                        return result
                        
                    except json.JSONDecodeError as e:
                        print(f"⚠️  JSON parsing failed: {e}")
                        print(f"Raw response: {content[:200]}...")
                        
                        # If it's the last attempt for this model, try the next model
                        if attempt == MAX_RETRIES - 1:
                            break
                        
                        continue
                        
                except Exception as e:
                    error_msg = str(e).lower()
                    
                    if 'no_access' in error_msg or 'access denied' in error_msg:
                        print(f"❌ No access to model {model_to_try}, trying next model...")
                        break  # Try next model
                    elif 'rate limit' in error_msg or '429' in error_msg:
                        print(f"⚠️  Rate limited. Waiting {RATE_LIMIT_DELAY} seconds...")
                        self.results['rate_limited'] = True
                        time.sleep(RATE_LIMIT_DELAY)
                        continue
                    elif 'unauthorized' in error_msg or '401' in error_msg:
                        print("❌ Unauthorized. Check your GITHUB_TOKEN has models:read permissions")
                        return self.create_fallback_response("Authentication failed")
                    elif 'quota' in error_msg:
                        print("❌ Quota exceeded. Consider upgrading or waiting for reset")
                        return self.create_fallback_response("Quota exceeded")
                    else:
                        print(f"⚠️  API call failed: {e}")
                        if attempt == MAX_RETRIES - 1:
                            break  # Try next model
                        time.sleep(2 ** attempt)  # Exponential backoff
        
        # If all models failed, return fallback response
        return self.create_fallback_response("All GitHub Models failed - no access to any models")
    
    def extract_scores_manually(self, content: str) -> Dict[str, Any]:
        """Extract scores manually if JSON parsing fails"""
        print("🔄 Attempting manual score extraction...")
        
        # Look for score patterns in the text
        import re
        
        scores = {}
        score_pattern = r'(\d+(?:\.\d+)?)/10|score[:\s]*(\d+(?:\.\d+)?)'
        matches = re.findall(score_pattern, content, re.IGNORECASE)
        
        if matches:
            # Extract the first 8 scores found
            numeric_scores = []
            for match in matches:
                score = match[0] if match[0] else match[1]
                if score:
                    numeric_scores.append(float(score))
            
            if len(numeric_scores) >= 8:
                categories = ['code_quality', 'test_coverage', 'documentation', 'architecture', 
                            'security', 'performance', 'maintainability', 'overall_health']
                
                for i, category in enumerate(categories):
                    scores[category] = {
                        "score": numeric_scores[i],
                        "explanation": f"Extracted from response",
                        "recommendations": ["Review response manually for specific recommendations"]
                    }
                
                overall_score = sum(numeric_scores) / len(numeric_scores)
                
                return {
                    "scores": scores,
                    "overall_score": overall_score,
                    "summary": "Scores extracted manually from response",
                    "critical_issues": ["Manual extraction used - review response"],
                    "next_steps": ["Review full response for detailed recommendations"]
                }
        
        return self.create_fallback_response("Could not extract scores from response")
    
    def create_fallback_response(self, reason: str) -> Dict[str, Any]:
        """Create a fallback response when API fails"""
        return {
            "scores": {
                "code_quality": {"score": 5.0, "explanation": reason, "recommendations": ["Check API connectivity"]},
                "test_coverage": {"score": 5.0, "explanation": reason, "recommendations": ["Check API connectivity"]},
                "documentation": {"score": 5.0, "explanation": reason, "recommendations": ["Check API connectivity"]},
                "architecture": {"score": 5.0, "explanation": reason, "recommendations": ["Check API connectivity"]},
                "security": {"score": 5.0, "explanation": reason, "recommendations": ["Check API connectivity"]},
                "performance": {"score": 5.0, "explanation": reason, "recommendations": ["Check API connectivity"]},
                "maintainability": {"score": 5.0, "explanation": reason, "recommendations": ["Check API connectivity"]},
                "overall_health": {"score": 5.0, "explanation": reason, "recommendations": ["Check API connectivity"]}
            },
            "overall_score": 5.0,
            "summary": f"Evaluation failed: {reason}",
            "critical_issues": [f"API error: {reason}"],
            "next_steps": ["Check GitHub token permissions", "Verify model availability", "Review rate limits"]
        }
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run the complete evaluation"""
        print(f"🚀 Starting GitHub Models LLM Judge evaluation...")
        print(f"📊 Model: {self.model}")
        print(f"⚡ Mode: {'Quick' if self.quick_mode else 'Full'}")
        print(f"🎯 Threshold: {self.threshold}")
        
        # Collect codebase information
        print("📁 Collecting codebase information...")
        codebase_info = self.collect_codebase_info()
        
        # Generate evaluation prompt
        print("📝 Generating evaluation prompt...")
        prompt = self.generate_evaluation_prompt(codebase_info)
        
        # Run evaluation
        print("🤖 Running LLM evaluation...")
        evaluation_result = self.evaluate_with_github_models(prompt)
        
        # Process results
        self.results.update(evaluation_result)
        
        # Calculate overall score if not provided
        if 'overall_score' not in self.results or self.results['overall_score'] == 0:
            scores = self.results.get('scores', {})
            if scores:
                total_score = sum(score.get('score', 0) for score in scores.values())
                self.results['overall_score'] = total_score / len(scores)
        
        return self.results
    
    def print_results(self, results: Dict[str, Any]):
        """Print evaluation results in a formatted way"""
        print("\n" + "="*60)
        print("🔍 GITHUB MODELS LLM JUDGE EVALUATION RESULTS")
        print("="*60)
        
        # Model and mode info
        print(f"🤖 Model: {results.get('model_used', 'Unknown')}")
        print(f"⚡ Mode: {results.get('evaluation_mode', 'Unknown').title()}")
        print(f"📅 Timestamp: {results.get('timestamp', 'Unknown')}")
        print(f"📊 API Calls: {results.get('api_calls', 0)}")
        
        if results.get('rate_limited'):
            print("⚠️  Rate limited during evaluation")
        
        # Overall score
        overall_score = results.get('overall_score', 0)
        print(f"\n🎯 OVERALL SCORE: {overall_score:.1f}/10")
        
        # Score breakdown
        print("\n📊 DETAILED SCORES:")
        scores = results.get('scores', {})
        for category, data in scores.items():
            if isinstance(data, dict):
                score = data.get('score', 0)
                explanation = data.get('explanation', 'No explanation provided')
                print(f"  {category.replace('_', ' ').title()}: {score:.1f}/10")
                print(f"    💡 {explanation}")
        
        # Summary
        if 'summary' in results:
            print(f"\n📝 SUMMARY:")
            print(f"  {results['summary']}")
        
        # Critical issues
        critical_issues = results.get('critical_issues', [])
        if critical_issues:
            print(f"\n🚨 CRITICAL ISSUES:")
            for issue in critical_issues:
                print(f"  • {issue}")
        
        # Next steps
        next_steps = results.get('next_steps', [])
        if next_steps:
            print(f"\n✅ NEXT STEPS:")
            for step in next_steps:
                print(f"  • {step}")
        
        # Pass/fail status
        threshold = self.threshold
        if overall_score >= threshold:
            print(f"\n✅ PASSED: Score {overall_score:.1f} >= {threshold}")
        else:
            print(f"\n❌ FAILED: Score {overall_score:.1f} < {threshold}")
        
        print("="*60)
    
    def save_results(self, results: Dict[str, Any]):
        """Save results to JSON file"""
        output_file = 'llm_judge_results.json'
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 Results saved to {output_file}")
        except Exception as e:
            print(f"⚠️  Failed to save results: {e}")
    
    def run(self) -> int:
        """Main execution method"""
        try:
            # Validate token
            if not self.token:
                print("❌ GITHUB_TOKEN environment variable not set")
                print("💡 Set it with: export GITHUB_TOKEN='your-token-here'")
                return 1
            
            # Run evaluation
            results = self.run_evaluation()
            
            # Print results
            self.print_results(results)
            
            # Save results
            self.save_results(results)
            
            # Return exit code based on threshold
            overall_score = results.get('overall_score', 0)
            if overall_score >= self.threshold:
                return 0
            else:
                return 1
                
        except KeyboardInterrupt:
            print("\n⚠️  Evaluation interrupted by user")
            return 1
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            return 1

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='GitHub Models LLM Judge Evaluator')
    parser.add_argument('--quick', action='store_true', help='Run in quick evaluation mode')
    parser.add_argument('--model', type=str, help='GitHub model to use')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD, help='Minimum score threshold')
    
    args = parser.parse_args()
    
    # Set environment variables from args
    if args.threshold != DEFAULT_THRESHOLD:
        os.environ['LLM_JUDGE_THRESHOLD'] = str(args.threshold)
    
    evaluator = GitHubModelsEvaluator(quick_mode=args.quick, model=args.model)
    return evaluator.run()

if __name__ == '__main__':
    sys.exit(main()) 
