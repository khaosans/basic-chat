#!/usr/bin/env python3
"""
Aggregate all test, coverage, LLM Judge, and performance results into a single Markdown report for CI/CD.
"""
import os
import json
from datetime import datetime

def read_json(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        print(f"⚠️  Could not read {path}")
        return None

def read_coverage():
    # Try to read coverage summary from htmlcov or coverage.xml
    summary = {}
    if os.path.exists('htmlcov/index.html'):
        # Parse HTML for total coverage (simple regex)
        try:
            with open('htmlcov/index.html') as f:
                html = f.read()
            import re
            m = re.search(r'TOTAL.*?(\d+)%', html)
            if m:
                summary['total'] = int(m.group(1))
        except Exception:
            pass
    if os.path.exists('coverage.xml'):
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse('coverage.xml')
            root = tree.getroot()
            summary['total'] = float(root.attrib.get('line-rate', 0)) * 100
        except Exception:
            pass
    return summary

def read_pytest_results():
    # Try to read pytest output from last run (if available)
    log_path = 'ci_build_doc_test.log'
    if not os.path.exists(log_path):
        return None
    summary = {}
    with open(log_path) as f:
        lines = f.readlines()
    for line in lines:
        if 'collected' in line and 'items' in line:
            summary['collected'] = int(line.split('collected')[1].split('items')[0].strip())
        if 'passed' in line and 'skipped' in line:
            import re
            m = re.findall(r'(\d+) passed', line)
            if m:
                summary['passed'] = int(m[0])
            m = re.findall(r'(\d+) skipped', line)
            if m:
                summary['skipped'] = int(m[0])
    return summary

def main():
    now = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    report = [f"# 🧪 Final Test Report\n\n*Generated: {now}*\n"]

    # Test summary
    pytest_results = read_pytest_results()
    if pytest_results:
        report.append("## ✅ Test Results Summary\n")
        report.append(f"- Total tests collected: {pytest_results.get('collected','?')}")
        report.append(f"- Passed: {pytest_results.get('passed','?')}")
        report.append(f"- Skipped: {pytest_results.get('skipped','?')}")
        report.append("")
    else:
        report.append("## ✅ Test Results Summary\n- ⚠️  No pytest summary found.\n")

    # Coverage summary
    coverage = read_coverage()
    if coverage and 'total' in coverage:
        report.append(f"## 📊 Coverage Summary\n- Total coverage: **{coverage['total']}%**\n")
    else:
        report.append("## 📊 Coverage Summary\n- ⚠️  No coverage data found.\n")

    # LLM Judge results
    llm_judge = read_json('llm_judge_results.json')
    if llm_judge:
        report.append("## 🤖 LLM Judge Results\n")
        score = llm_judge.get('overall_score', '?')
        report.append(f"- Overall Score: **{score}/10**")
        if 'scores' in llm_judge:
            report.append("- Score Breakdown:")
            for k, v in llm_judge['scores'].items():
                if isinstance(v, dict):
                    report.append(f"    - {k}: {v.get('score','?')}/10 — {v.get('justification','')}")
                else:
                    report.append(f"    - {k}: {v}")
        if 'recommendations' in llm_judge:
            report.append("- Recommendations:")
            for rec in llm_judge['recommendations']:
                report.append(f"    - {rec}")
        if 'next_steps' in llm_judge:
            report.append("- Next Steps:")
            for step in llm_judge['next_steps']:
                report.append(f"    - {step}")
        report.append("")
    else:
        report.append("## 🤖 LLM Judge Results\n- ⚠️  No LLM Judge results found.\n")

    # Performance metrics
    perf = read_json('performance_metrics.json')
    if perf:
        report.append("## 🚦 Performance Metrics\n")
        for k in ['elapsed_seconds','memory_mb','threshold_seconds','threshold_mb','status']:
            if k in perf:
                report.append(f"- {k.replace('_',' ').title()}: {perf[k]}")
        report.append("")
    else:
        report.append("## 🚦 Performance Metrics\n- ⚠️  No performance metrics found.\n")

    # Recommendations
    report.append("## 📝 Recommendations\n")
    if llm_judge and 'recommendations' in llm_judge:
        for rec in llm_judge['recommendations']:
            report.append(f"- {rec}")
    else:
        report.append("- No recommendations from LLM Judge.\n")
    if coverage and coverage.get('total',0) < 50:
        report.append("- 🚨 Coverage is below 50%. Add more tests!")
    if perf and perf.get('status') == 'FAIL':
        report.append("- 🚨 Performance regression detected. Optimize code or dependencies.")
    report.append("")

    # Comparison to previous run (stub)
    report.append("## 🔄 Comparison to Previous Run\n- (Comparison feature coming soon)\n")

    # Save report
    with open('final_test_report.md','w') as f:
        f.write('\n'.join(report))
    print("✅ Final test report generated: final_test_report.md")

if __name__ == "__main__":
    main() 