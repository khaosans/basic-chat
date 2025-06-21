#!/usr/bin/env python3
"""
Comprehensive static analysis script for the basic-chat-template project.

This script runs multiple static analysis tools and provides a unified report
of code quality, security issues, and potential problems.

Usage:
    python scripts/lint.py [--fix] [--verbose] [--tools TOOL1,TOOL2,...]

Examples:
    python scripts/lint.py                    # Run all tools
    python scripts/lint.py --fix              # Run tools that can auto-fix
    python scripts/lint.py --tools mypy,flake8  # Run specific tools
    python scripts/lint.py --verbose          # Show detailed output
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LintResult:
    """Container for linting results"""
    
    def __init__(self, tool: str, success: bool, output: str = "", errors: List[str] = None):
        self.tool = tool
        self.success = success
        self.output = output
        self.errors = errors or []
        self.duration = 0.0

class StaticAnalyzer:
    """Main static analysis orchestrator"""
    
    def __init__(self, verbose: bool = False, fix: bool = False):
        self.verbose = verbose
        self.fix = fix
        self.results: List[LintResult] = []
        self.project_root = Path(__file__).parent.parent
        
    def run_command(self, cmd: List[str], timeout: int = 300) -> Tuple[bool, str, List[str]]:
        """Run a command and return success status, output, and errors"""
        try:
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.project_root
            )
            duration = time.time() - start_time
            
            success = result.returncode == 0
            output = result.stdout
            errors = result.stderr.split('\n') if result.stderr else []
            
            if self.verbose:
                logger.info(f"Command: {' '.join(cmd)}")
                logger.info(f"Duration: {duration:.2f}s")
                logger.info(f"Return code: {result.returncode}")
                if output:
                    logger.info(f"Output: {output[:500]}...")
                if errors:
                    logger.info(f"Errors: {errors[:5]}...")
            
            return success, output, errors
            
        except subprocess.TimeoutExpired:
            return False, "", [f"Command timed out after {timeout} seconds"]
        except FileNotFoundError:
            return False, "", [f"Command not found: {cmd[0]}"]
        except Exception as e:
            return False, "", [f"Unexpected error: {str(e)}"]
    
    def run_black(self) -> LintResult:
        """Run Black code formatter"""
        logger.info("Running Black...")
        cmd = ["black", "--check", "--diff", "."]
        if self.fix:
            cmd = ["black", "."]
        
        success, output, errors = self.run_command(cmd)
        return LintResult("Black", success, output, errors)
    
    def run_isort(self) -> LintResult:
        """Run isort import sorter"""
        logger.info("Running isort...")
        cmd = ["isort", "--check-only", "--diff", "."]
        if self.fix:
            cmd = ["isort", "."]
        
        success, output, errors = self.run_command(cmd)
        return LintResult("isort", success, output, errors)
    
    def run_flake8(self) -> LintResult:
        """Run flake8 linter"""
        logger.info("Running flake8...")
        cmd = ["flake8", "."]
        success, output, errors = self.run_command(cmd)
        return LintResult("flake8", success, output, errors)
    
    def run_mypy(self) -> LintResult:
        """Run mypy type checker"""
        logger.info("Running mypy...")
        cmd = ["mypy", "."]
        success, output, errors = self.run_command(cmd)
        return LintResult("mypy", success, output, errors)
    
    def run_pylint(self) -> LintResult:
        """Run pylint"""
        logger.info("Running pylint...")
        cmd = ["pylint", "--rcfile=pyproject.toml", "."]
        success, output, errors = self.run_command(cmd)
        return LintResult("pylint", success, output, errors)
    
    def run_bandit(self) -> LintResult:
        """Run bandit security linter"""
        logger.info("Running bandit...")
        cmd = ["bandit", "-r", ".", "-f", "json", "-o", "/tmp/bandit-report.json"]
        success, output, errors = self.run_command(cmd)
        
        # Read bandit report if it exists
        bandit_report = Path("/tmp/bandit-report.json")
        if bandit_report.exists():
            try:
                with open(bandit_report, 'r') as f:
                    report = json.load(f)
                issues = report.get('results', [])
                if issues:
                    errors.extend([f"{issue['filename']}:{issue['line_number']} - {issue['issue_text']}" 
                                 for issue in issues])
            except Exception as e:
                errors.append(f"Failed to parse bandit report: {e}")
        
        return LintResult("bandit", success, output, errors)
    
    def run_safety(self) -> LintResult:
        """Run safety security checker"""
        logger.info("Running safety...")
        cmd = ["safety", "check", "--json"]
        success, output, errors = self.run_command(cmd)
        
        # Parse safety output
        if output:
            try:
                vulns = json.loads(output)
                if vulns:
                    errors.extend([f"{v['package']} {v['installed_version']} - {v['advisory']}" 
                                 for v in vulns])
            except json.JSONDecodeError:
                pass
        
        return LintResult("safety", success, output, errors)
    
    def run_pytest(self) -> LintResult:
        """Run pytest"""
        logger.info("Running pytest...")
        cmd = ["pytest", "--tb=short", "--quiet"]
        success, output, errors = self.run_command(cmd)
        return LintResult("pytest", success, output, errors)
    
    def run_radon(self) -> LintResult:
        """Run radon complexity analyzer"""
        logger.info("Running radon...")
        cmd = ["radon", "cc", ".", "-a", "-nc"]
        success, output, errors = self.run_command(cmd)
        return LintResult("radon", success, output, errors)
    
    def run_xenon(self) -> LintResult:
        """Run xenon complexity checker"""
        logger.info("Running xenon...")
        cmd = ["xenon", ".", "--max-absolute=A", "--max-modules=A", "--max-average=A"]
        success, output, errors = self.run_command(cmd)
        return LintResult("xenon", success, output, errors)
    
    def run_autoflake(self) -> LintResult:
        """Run autoflake to remove unused imports"""
        logger.info("Running autoflake...")
        cmd = ["autoflake", "--in-place", "--remove-all-unused-imports", "--remove-unused-variables", "."]
        success, output, errors = self.run_command(cmd)
        return LintResult("autoflake", success, output, errors)
    
    def run_pyright(self) -> LintResult:
        """Run pyright type checker"""
        logger.info("Running pyright...")
        cmd = ["pyright", "."]
        success, output, errors = self.run_command(cmd)
        return LintResult("pyright", success, output, errors)
    
    def run_all_tools(self, tools: Optional[List[str]] = None) -> List[LintResult]:
        """Run all specified tools or all available tools"""
        tool_map = {
            'black': self.run_black,
            'isort': self.run_isort,
            'flake8': self.run_flake8,
            'mypy': self.run_mypy,
            'pylint': self.run_pylint,
            'bandit': self.run_bandit,
            'safety': self.run_safety,
            'pytest': self.run_pytest,
            'radon': self.run_radon,
            'xenon': self.run_xenon,
            'autoflake': self.run_autoflake,
            'pyright': self.run_pyright,
        }
        
        if tools:
            available_tools = [tool for tool in tools if tool in tool_map]
            if len(available_tools) != len(tools):
                missing = set(tools) - set(tool_map.keys())
                logger.warning(f"Unknown tools: {missing}")
        else:
            available_tools = list(tool_map.keys())
        
        logger.info(f"Running tools: {', '.join(available_tools)}")
        
        for tool_name in available_tools:
            try:
                result = tool_map[tool_name]()
                self.results.append(result)
            except Exception as e:
                logger.error(f"Error running {tool_name}: {e}")
                self.results.append(LintResult(tool_name, False, "", [str(e)]))
        
        return self.results
    
    def generate_report(self) -> str:
        """Generate a comprehensive report of all results"""
        if not self.results:
            return "No results to report"
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("STATIC ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Summary
        total_tools = len(self.results)
        successful_tools = sum(1 for r in self.results if r.success)
        failed_tools = total_tools - successful_tools
        
        report_lines.append(f"SUMMARY:")
        report_lines.append(f"  Total tools run: {total_tools}")
        report_lines.append(f"  Successful: {successful_tools}")
        report_lines.append(f"  Failed: {failed_tools}")
        report_lines.append(f"  Success rate: {successful_tools/total_tools*100:.1f}%")
        report_lines.append("")
        
        # Detailed results
        report_lines.append("DETAILED RESULTS:")
        report_lines.append("-" * 40)
        
        for result in self.results:
            status = "✅ PASS" if result.success else "❌ FAIL"
            report_lines.append(f"{result.tool:12} {status}")
            
            if result.errors and self.verbose:
                for error in result.errors[:3]:  # Show first 3 errors
                    report_lines.append(f"  {error}")
                if len(result.errors) > 3:
                    report_lines.append(f"  ... and {len(result.errors) - 3} more errors")
        
        report_lines.append("")
        
        # Recommendations
        if failed_tools > 0:
            report_lines.append("RECOMMENDATIONS:")
            report_lines.append("-" * 40)
            
            failed_results = [r for r in self.results if not r.success]
            for result in failed_results:
                if result.tool == "black":
                    report_lines.append("• Run 'black .' to format code")
                elif result.tool == "isort":
                    report_lines.append("• Run 'isort .' to sort imports")
                elif result.tool == "autoflake":
                    report_lines.append("• Run 'autoflake --in-place --remove-all-unused-imports .' to clean imports")
                elif result.tool == "mypy":
                    report_lines.append("• Fix type annotations or add type ignores")
                elif result.tool == "flake8":
                    report_lines.append("• Fix code style issues")
                elif result.tool == "pylint":
                    report_lines.append("• Address code quality issues")
                elif result.tool == "bandit":
                    report_lines.append("• Review security issues")
                elif result.tool == "safety":
                    report_lines.append("• Update vulnerable dependencies")
                elif result.tool == "pytest":
                    report_lines.append("• Fix failing tests")
                elif result.tool in ["radon", "xenon"]:
                    report_lines.append("• Reduce code complexity")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def save_report(self, filename: str = "lint-report.txt"):
        """Save the report to a file"""
        report = self.generate_report()
        report_path = self.project_root / filename
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Report saved to {report_path}")
        return report_path

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run comprehensive static analysis")
    parser.add_argument("--fix", action="store_true", help="Auto-fix issues where possible")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    parser.add_argument("--tools", help="Comma-separated list of tools to run")
    parser.add_argument("--output", help="Output file for report")
    
    args = parser.parse_args()
    
    # Parse tools list
    tools = None
    if args.tools:
        tools = [tool.strip() for tool in args.tools.split(",")]
    
    # Run analysis
    analyzer = StaticAnalyzer(verbose=args.verbose, fix=args.fix)
    results = analyzer.run_all_tools(tools)
    
    # Generate and display report
    report = analyzer.generate_report()
    print(report)
    
    # Save report if requested
    if args.output:
        analyzer.save_report(args.output)
    
    # Exit with appropriate code
    failed_tools = sum(1 for r in results if not r.success)
    if failed_tools > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main() 