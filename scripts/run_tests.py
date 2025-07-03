#!/usr/bin/env python3
"""
Test runner script for BasicChat application.
Provides different test execution modes for development and CI.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}")
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
    else:
        print(f"❌ {description} failed")
        if result.stderr:
            print(result.stderr)
        if result.stdout:
            print(result.stdout)
        sys.exit(result.returncode)
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Run BasicChat tests")
    parser.add_argument(
        "--mode",
        choices=["unit", "integration", "all", "fast", "slow"],
        default="unit",
        help="Test mode to run"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Test timeout in seconds"
    )
    
    args = parser.parse_args()
    
    # Base pytest command
    cmd = ["python3", "-m", "pytest", "tests/"]
    
    # Add mode-specific options
    if args.mode == "unit":
        cmd.extend(["-m", "unit or fast"])
        print("🧪 Running UNIT TESTS (fast, isolated)")
    elif args.mode == "integration":
        cmd.extend(["-m", "integration"])
        print("🧪 Running INTEGRATION TESTS (external dependencies)")
    elif args.mode == "fast":
        cmd.extend(["-m", "fast"])
        print("🧪 Running FAST TESTS (mocked only)")
    elif args.mode == "slow":
        cmd.extend(["-m", "slow"])
        print("🧪 Running SLOW TESTS (LLM calls)")
    elif args.mode == "all":
        print("🧪 Running ALL TESTS")
    
    # Add parallel execution
    if args.parallel and args.mode != "slow":
        cmd.extend(["-n", "auto", "--dist=worksteal"])
        print("⚡ Running tests in parallel")
    
    # Add coverage
    if args.coverage:
        cmd.extend([
            "--cov=app",
            "--cov=reasoning_engine", 
            "--cov=document_processor",
            "--cov=utils",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov"
        ])
        print("📊 Generating coverage report")
    
    # Add verbosity
    if args.verbose:
        cmd.extend(["-v", "-s"])
    
    # Add timeout
    cmd.extend(["--timeout", str(args.timeout)])
    
    # Add other options
    cmd.extend([
        "--tb=short",
        "--color=yes"
    ])
    
    # Set environment variables
    env = os.environ.copy()
    env.update({
        'TESTING': 'true',
        'CHROMA_PERSIST_DIR': './test_chroma_db',
        'MOCK_EXTERNAL_SERVICES': 'true' if args.mode in ['unit', 'fast'] else 'false'
    })
    
    print(f"\n🚀 Starting test run with mode: {args.mode}")
    print(f"Command: {' '.join(cmd)}")
    
    # Run the tests
    try:
        result = subprocess.run(cmd, env=env, check=True)
        print(f"\n🎉 All tests passed!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Tests failed with exit code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print(f"\n⏹️  Test run interrupted by user")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
