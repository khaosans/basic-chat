#!/usr/bin/env python3
"""
Generate actionable LLM Judge report
Converts LLM judge results into an easy-to-follow action plan
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Tuple

def load_results() -> Dict[str, Any]:
    """Load LLM judge results from JSON file"""
    results_file = "llm_judge_results.json"
    if not os.path.exists(results_file):
        print(f"❌ Results file not found: {results_file}")
        sys.exit(1)
    
    try:
        with open(results_file, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse results file: {e}")
        sys.exit(1)

def load_rules() -> Dict[str, Any]:
    """Load evaluation rules"""
    rules_file = "basicchat/evaluation/evaluators/llm_judge_rules.json"
    if not os.path.exists(rules_file):
        print(f"⚠️ Rules file not found: {rules_file}, using defaults")
        return {}
    
    try:
        with open(rules_file, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"⚠️ Failed to parse rules file: {e}")
        return {}

def categorize_issues(scores: Dict[str, Any], rules: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Categorize issues by priority and type"""
    issues = []
    categories = rules.get('categories', {})
    action_items = rules.get('action_items', {})
    
    for category_name, score_data in scores.items():
        if isinstance(score_data, dict):
            score = score_data.get('score', 0)
            justification = score_data.get('justification', '')
        else:
            score = score_data
            justification = ''
        
        category_config = categories.get(category_name, {})
        priority = category_config.get('priority', 'medium')
        is_critical = category_config.get('critical', False)
        
        # Determine issue severity based on score
        if score < 6:
            severity = 'critical' if is_critical else 'high'
        elif score < 7:
            severity = 'high'
        elif score < 8:
            severity = 'medium'
        else:
            severity = 'low'
        
        # Get category-specific rules for actionable items
        category_rules = category_config.get('rules', [])
        
        issues.append({
            'category': category_name,
            'score': score,
            'severity': severity,
            'priority': priority,
            'justification': justification,
            'rules': category_rules,
            'needs_attention': score < 7
        })
    
    return issues

def generate_action_plan(issues: List[Dict[str, Any]], overall_score: float, rules: Dict[str, Any]) -> str:
    """Generate an actionable plan from issues"""
    report = []
    
    # Header
    report.append("# 🤖 LLM Judge Action Plan")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Overall Score: {overall_score:.1f}/10")
    report.append("")
    
    # Summary
    critical_issues = [i for i in issues if i['severity'] == 'critical']
    high_issues = [i for i in issues if i['severity'] == 'high']
    medium_issues = [i for i in issues if i['severity'] == 'medium']
    
    report.append("## 📊 Summary")
    report.append(f"- **Critical Issues**: {len(critical_issues)}")
    report.append(f"- **High Priority Issues**: {len(high_issues)}")
    report.append(f"- **Medium Priority Issues**: {len(medium_issues)}")
    report.append("")
    
    # Priority levels explanation
    priority_levels = rules.get('action_items', {}).get('priority_levels', {})
    if priority_levels:
        report.append("## 🎯 Priority Levels")
        for level, description in priority_levels.items():
            report.append(f"- **{level.title()}**: {description}")
        report.append("")
    
    # Critical Issues
    if critical_issues:
        report.append("## 🚨 Critical Issues (Must Fix Immediately)")
        for issue in critical_issues:
            report.append(f"### {issue['category'].replace('_', ' ').title()}")
            report.append(f"**Score**: {issue['score']}/10")
            report.append(f"**Issue**: {issue['justification']}")
            report.append("")
            report.append("**Action Items**:")
            for rule in issue['rules'][:5]:  # Top 5 rules
                report.append(f"- [ ] {rule}")
            report.append("")
    
    # High Priority Issues
    if high_issues:
        report.append("## ⚠️ High Priority Issues (Should Fix Soon)")
        for issue in high_issues:
            report.append(f"### {issue['category'].replace('_', ' ').title()}")
            report.append(f"**Score**: {issue['score']}/10")
            report.append(f"**Issue**: {issue['justification']}")
            report.append("")
            report.append("**Action Items**:")
            for rule in issue['rules'][:3]:  # Top 3 rules
                report.append(f"- [ ] {rule}")
            report.append("")
    
    # Medium Priority Issues
    if medium_issues:
        report.append("## 📝 Medium Priority Issues (Good to Fix)")
        for issue in medium_issues:
            report.append(f"### {issue['category'].replace('_', ' ').title()}")
            report.append(f"**Score**: {issue['score']}/10")
            report.append(f"**Issue**: {issue['justification']}")
            report.append("")
            report.append("**Action Items**:")
            for rule in issue['rules'][:2]:  # Top 2 rules
                report.append(f"- [ ] {rule}")
            report.append("")
    
    # Quick Wins
    quick_wins = []
    for issue in issues:
        if issue['score'] >= 7 and issue['score'] < 8:
            quick_wins.append(issue)
    
    if quick_wins:
        report.append("## 🚀 Quick Wins (Easy Improvements)")
        for issue in quick_wins:
            report.append(f"- **{issue['category'].replace('_', ' ').title()}**: {issue['justification']}")
        report.append("")
    
    # Best Practices Checklist
    report.append("## ✅ Best Practices Checklist")
    best_practices = rules.get('best_practices', {})
    
    if 'python' in best_practices:
        report.append("### Python Best Practices")
        for practice in best_practices['python']:
            report.append(f"- [ ] {practice}")
        report.append("")
    
    if 'general' in best_practices:
        report.append("### General Best Practices")
        for practice in best_practices['general']:
            report.append(f"- [ ] {practice}")
        report.append("")
    
    # Next Steps
    report.append("## 🎯 Next Steps")
    if critical_issues:
        report.append("1. **Immediate**: Address all critical issues")
    if high_issues:
        report.append("2. **Short-term**: Fix high priority issues")
    if medium_issues:
        report.append("3. **Medium-term**: Improve medium priority areas")
    report.append("4. **Ongoing**: Run LLM Judge regularly to track progress")
    report.append("5. **Continuous**: Follow best practices checklist")
    report.append("")
    
    # Commands
    report.append("## 🔧 Useful Commands")
    report.append("```bash")
    report.append("# Run quick evaluation")
    report.append("./scripts/run_llm_judge.sh quick ollama 7.0")
    report.append("")
    report.append("# Run full evaluation")
    report.append("./scripts/run_llm_judge.sh full ollama 7.0")
    report.append("")
    report.append("# Run with OpenAI (if available)")
    report.append("./scripts/run_llm_judge.sh quick openai 7.0")
    report.append("```")
    report.append("")
    
    # Footer
    report.append("---")
    report.append("*This report was generated automatically by the LLM Judge evaluation system.*")
    report.append("*Review and update this action plan regularly as you implement improvements.*")
    
    return "\n".join(report)

def generate_improvement_tips(issues: List[Dict[str, Any]], rules: Dict[str, Any]) -> str:
    """Generate specific improvement tips"""
    tips = []
    
    for issue in issues:
        if issue['score'] < 7:  # Focus on areas needing improvement
            category = issue['category']
            score = issue['score']
            
            tips.append(f"## {category.replace('_', ' ').title()} (Score: {score}/10)")
            
            if category == 'code_quality':
                tips.extend([
                    "- Run `black` to format code consistently",
                    "- Use `flake8` to check for style issues",
                    "- Add type hints to function signatures",
                    "- Break down large functions into smaller ones",
                    "- Use meaningful variable names"
                ])
            elif category == 'test_coverage':
                tips.extend([
                    "- Run `pytest --cov` to check current coverage",
                    "- Add tests for untested functions",
                    "- Write tests for edge cases",
                    "- Use `pytest-mock` for mocking dependencies",
                    "- Add integration tests for critical paths"
                ])
            elif category == 'documentation':
                tips.extend([
                    "- Update README.md with setup instructions",
                    "- Add docstrings to all functions",
                    "- Create API documentation",
                    "- Include usage examples",
                    "- Document configuration options"
                ])
            elif category == 'architecture':
                tips.extend([
                    "- Review SOLID principles implementation",
                    "- Reduce coupling between modules",
                    "- Use dependency injection",
                    "- Implement proper error handling",
                    "- Consider design patterns for complex logic"
                ])
            elif category == 'security':
                tips.extend([
                    "- Validate all user inputs",
                    "- Use parameterized queries",
                    "- Implement proper authentication",
                    "- Follow OWASP guidelines",
                    "- Keep dependencies updated"
                ])
            elif category == 'performance':
                tips.extend([
                    "- Profile code to identify bottlenecks",
                    "- Optimize database queries",
                    "- Implement caching where appropriate",
                    "- Use async/await for I/O operations",
                    "- Monitor memory usage"
                ])
            
            tips.append("")
    
    return "\n".join(tips)

def main():
    """Main function"""
    print("📋 Generating LLM Judge Action Plan...")
    
    # Load data
    results = load_results()
    rules = load_rules()
    
    # Extract scores
    scores = results.get('scores', {})
    overall_score = results.get('overall_score', 0.0)
    
    # Categorize issues
    issues = categorize_issues(scores, rules)
    
    # Generate reports
    action_plan = generate_action_plan(issues, overall_score, rules)
    improvement_tips = generate_improvement_tips(issues, rules)
    
    # Write action plan
    with open('llm_judge_action_items.md', 'w') as f:
        f.write(action_plan)
    
    # Write improvement tips
    with open('llm_judge_improvement_tips.md', 'w') as f:
        f.write(improvement_tips)
    
    # Print summary
    print("✅ Generated action plan: llm_judge_action_items.md")
    print("✅ Generated improvement tips: llm_judge_improvement_tips.md")
    
    # Print quick summary
    critical_count = len([i for i in issues if i['severity'] == 'critical'])
    high_count = len([i for i in issues if i['severity'] == 'high'])
    
    print(f"\n📊 Quick Summary:")
    print(f"- Overall Score: {overall_score:.1f}/10")
    print(f"- Critical Issues: {critical_count}")
    print(f"- High Priority Issues: {high_count}")
    
    if critical_count > 0:
        print("🚨 Critical issues found - review llm_judge_action_items.md immediately!")
    elif high_count > 0:
        print("⚠️ High priority issues found - plan to address them soon.")
    else:
        print("✅ No critical or high priority issues found!")

if __name__ == "__main__":
    main()
