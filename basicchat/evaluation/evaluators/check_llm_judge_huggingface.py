#!/usr/bin/env python3
"""
LLM Judge Evaluator using Hugging Face Models

This script evaluates the codebase using Hugging Face models to assess:
- Code quality and maintainability
- Test coverage and effectiveness
- Documentation quality
- Overall project health

Compatible with CI and local dev. Uses transformers pipeline for local or API-based inference.

Usage:
    python evaluators/check_llm_judge_huggingface.py [--quick] [--model microsoft/DialoGPT-medium]

Environment Variables:
    HF_API_KEY: Hugging Face API key (optional for public models)
    HF_MODEL: Hugging Face model to use (default: microsoft/DialoGPT-medium)
    LLM_JUDGE_THRESHOLD: Minimum score required (default: 7.0)
    HF_DEVICE: Device to use (default: auto, options: cpu, cuda, mps)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import time

# Add the parent directory to the path so we can import from app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basicchat.core.config import config
from basicchat.evaluation.evaluators.consistency import LLMJudgeConsistency

DEFAULT_THRESHOLD = 7.0
DEFAULT_MODEL = "microsoft/DialoGPT-medium"
MAX_RETRIES = 2

class HuggingFaceEvaluator:
    """LLM-based code evaluator using Hugging Face models"""
    def __init__(self, quick_mode: bool = False, model: Optional[str] = None):
        self.model = model or os.getenv('HF_MODEL', DEFAULT_MODEL)
        self.threshold = float(os.getenv('LLM_JUDGE_THRESHOLD', DEFAULT_THRESHOLD))
        self.quick_mode = quick_mode
        self.api_key = os.getenv('HF_API_KEY')
        self.device = os.getenv('HF_DEVICE', 'auto')
        self.consistency = LLMJudgeConsistency()
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'scores': {},
            'details': {},
            'recommendations': [],
            'overall_score': 0.0,
            'evaluation_mode': 'quick' if quick_mode else 'full',
            'model_used': self.model,
            'inference_time': 0.0
        }
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
            import torch
            self._torch = torch
            self._pipeline = pipeline
            self._AutoTokenizer = AutoTokenizer
            self._AutoModelForCausalLM = AutoModelForCausalLM
        except ImportError:
            print("❌ transformers/torch not installed. Install with: pip install transformers torch", file=sys.stderr)
            sys.exit(1)
        self._load_model()

    def _load_model(self):
        print(f"🤗 Loading Hugging Face model: {self.model}")
        device = self.device
        if device == 'auto':
            if self._torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(self._torch.backends, 'mps') and self._torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = device
        print(f"🔧 Using device: {self.device}")
        try:
            self.tokenizer = self._AutoTokenizer.from_pretrained(self.model)
            self.model_obj = self._AutoModelForCausalLM.from_pretrained(self.model)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.generator = self._pipeline(
                "text-generation",
                model=self.model_obj,
                tokenizer=self.tokenizer,
                device=0 if self.device == 'cuda' else -1
            )
            # Detect max length from tokenizer or set default
            self.max_length = getattr(self.tokenizer, 'model_max_length', 1024)
            if self.max_length is None or self.max_length > 100_000:
                self.max_length = 1024  # fallback for some tokenizers
            print(f"✅ Model loaded: {self.model} (max context: {self.max_length} tokens)")
        except Exception as e:
            print(f"❌ Failed to load model: {e}", file=sys.stderr)
            sys.exit(1)

    def _truncate_prompt(self, prompt: str) -> str:
        # Truncate prompt to fit model's max context length
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(tokens) > self.max_length:
            # Try to keep the last part (rubric + JSON format)
            # Heuristic: keep last 60% of tokens
            keep_tokens = tokens[-int(self.max_length * 0.6):]
            truncated_prompt = self.tokenizer.decode(keep_tokens, skip_special_tokens=True)
            print(f"⚠️  Prompt truncated from {len(tokens)} to {len(keep_tokens)} tokens for model context window ({self.max_length}).")
            return truncated_prompt
        return prompt

    def collect_codebase_info(self) -> Dict[str, Any]:
        # Mimic OpenAI evaluator's quick/full mode
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
            for test_dir in test_dirs:
                if os.path.exists(test_dir):
                    for root, dirs, files in os.walk(test_dir):
                        for file in files:
                            if file.endswith('.py') and ('test' in file.lower() or file.startswith('test_')):
                                info['test_files'] += 1
        else:
            for root, dirs, files in os.walk('.'):
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
        return info

    def generate_evaluation_prompt(self, codebase_info: Dict[str, Any]) -> str:
        # Reuse OpenAI/consistency prompt
        mode_note = "QUICK EVALUATION MODE - Focus on critical issues only" if self.quick_mode else "FULL EVALUATION MODE"
        rubric_md = self.consistency.rubric_text()
        version = self.consistency.version
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

Please evaluate the following aspects and provide scores from 1-10 (where 10 is excellent):

{rubric_md}

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

    def evaluate_with_hf(self, prompt: str) -> Dict[str, Any]:
        # Truncate prompt if needed
        prompt = self._truncate_prompt(prompt)
        for attempt in range(MAX_RETRIES):
            try:
                start = time.time()
                outputs = self.generator(prompt, max_new_tokens=512, pad_token_id=self.tokenizer.eos_token_id)
                end = time.time()
                self.results['inference_time'] = round(end - start, 2)
                content = outputs[0]['generated_text']
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start != -1 and json_end != 0:
                    json_str = content[json_start:json_end]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError as e:
                        print(f"⚠️  JSON parse error: {e}")
                        if attempt == MAX_RETRIES - 1:
                            raise
                        continue
                else:
                    print("⚠️  No JSON found in output.")
                    if attempt == MAX_RETRIES - 1:
                        raise ValueError("No JSON found in model output")
            except Exception as e:
                print(f"❌ Model inference failed: {e}")
                if attempt == MAX_RETRIES - 1:
                    raise
                continue
        raise Exception("Failed to get valid response from Hugging Face model after all retries")

    def run_evaluation(self) -> Dict[str, Any]:
        print(f"🔍 Collecting codebase info ({'QUICK' if self.quick_mode else 'FULL'})...")
        codebase_info = self.collect_codebase_info()
        print("🤖 Generating evaluation prompt...")
        prompt = self.generate_evaluation_prompt(codebase_info)
        print(f"🧠 Running LLM evaluation with Hugging Face Model: {self.model}...")
        evaluation = self.evaluate_with_hf(prompt)
        self.results.update(evaluation)
        self.results['codebase_info'] = codebase_info
        return self.results

    def print_results(self, results: Dict[str, Any]):
        print("\n" + "="*60)
        print(f"🤖 LLM JUDGE EVALUATION RESULTS (HuggingFace)")
        print("="*60)
        scores = results.get('scores', {})
        overall_score = results.get('overall_score', 0.0)
        for k, v in scores.items():
            print(f"{k}: {v}")
        print(f"Overall Score: {overall_score}")
        print(f"Recommendations: {results.get('recommendations', [])}")
        print(f"Inference Time: {self.results.get('inference_time', 0.0)}s")
        print("="*60)

    def run(self) -> int:
        try:
            results = self.run_evaluation()
            self.print_results(results)
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
    parser = argparse.ArgumentParser(description='LLM Judge Evaluator using Hugging Face Models')
    parser.add_argument('--quick', action='store_true', help='Run in quick mode for faster CI evaluation')
    parser.add_argument('--model', type=str, default=None, help=f'Hugging Face model to use (default: {DEFAULT_MODEL})')
    args = parser.parse_args()
    evaluator = HuggingFaceEvaluator(quick_mode=args.quick, model=args.model)
    exit_code = evaluator.run()
    sys.exit(exit_code)

if __name__ == "__main__":
    main() 