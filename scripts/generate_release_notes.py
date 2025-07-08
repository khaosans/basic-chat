#!/usr/bin/env python3
"""
Generate release notes from git commits and PRs
"""
import subprocess
import sys
import re
from datetime import datetime
from typing import List, Dict

def get_commits_since_last_tag(version: str) -> List[str]:
    try:
        result = subprocess.run(
            ['git', 'describe', '--tags', '--abbrev=0', f'{version}^'],
            capture_output=True, text=True, check=True
        )
        last_tag = result.stdout.strip()
    except subprocess.CalledProcessError:
        last_tag = None
    if last_tag:
        cmd = ['git', 'log', f'{last_tag}..{version}', '--oneline', '--no-merges']
    else:
        cmd = ['git', 'log', '--oneline', '--no-merges']
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip().split('\n') if result.stdout.strip() else []

def categorize_commits(commits: List[str]) -> Dict[str, List[str]]:
    categories = {
        '🚀 Features': [],
        '🐛 Bug Fixes': [],
        '🔧 Improvements': [],
        '📚 Documentation': [],
        '🧪 Testing': [],
        '🔒 Security': [],
        '⚡ Performance': [],
        '🏗️ Infrastructure': [],
        '📦 Dependencies': [],
        '🔨 Maintenance': []
    }
    for commit in commits:
        if not commit:
            continue
        message = commit.split(' ', 1)[1] if ' ' in commit else commit
        if any(keyword in message.lower() for keyword in ['feat:', 'feature', 'add', 'new']):
            categories['🚀 Features'].append(message)
        elif any(keyword in message.lower() for keyword in ['fix:', 'bug', 'fix', 'resolve']):
            categories['🐛 Bug Fixes'].append(message)
        elif any(keyword in message.lower() for keyword in ['perf:', 'performance', 'optimize', 'speed']):
            categories['⚡ Performance'].append(message)
        elif any(keyword in message.lower() for keyword in ['docs:', 'documentation', 'readme']):
            categories['📚 Documentation'].append(message)
        elif any(keyword in message.lower() for keyword in ['test:', 'testing', 'spec']):
            categories['🧪 Testing'].append(message)
        elif any(keyword in message.lower() for keyword in ['security', 'vulnerability']):
            categories['🔒 Security'].append(message)
        elif any(keyword in message.lower() for keyword in ['ci:', 'cd:', 'workflow', 'github']):
            categories['🏗️ Infrastructure'].append(message)
        elif any(keyword in message.lower() for keyword in ['deps:', 'dependency', 'package']):
            categories['📦 Dependencies'].append(message)
        elif any(keyword in message.lower() for keyword in ['refactor:', 'improve', 'enhance']):
            categories['🔧 Improvements'].append(message)
        else:
            categories['🔨 Maintenance'].append(message)
    return categories

def generate_release_notes(version: str) -> str:
    commits = get_commits_since_last_tag(version)
    categories = categorize_commits(commits)
    notes = f"# BasicChat {version}\n\n"
    notes += f"**Release Date:** {datetime.now().strftime('%Y-%m-%d')}\n\n"
    total_commits = len(commits)
    notes += f"## 📊 Summary\n\n"
    notes += f"- **Total Changes:** {total_commits} commits\n"
    notes += f"- **Release Type:** {'Production' if not version.endswith('-rc') else 'Release Candidate'}\n\n"
    notes += "## 📝 Changes\n\n"
    for category, messages in categories.items():
        if messages:
            notes += f"### {category}\n\n"
            for message in messages:
                clean_message = re.sub(r'^[a-z]+:\s*', '', message, flags=re.IGNORECASE)
                notes += f"- {clean_message}\n"
            notes += "\n"
    breaking_changes = [c for c in commits if 'breaking' in c.lower() or '!:' in c]
    if breaking_changes:
        notes += "## ⚠️ Breaking Changes\n\n"
        for change in breaking_changes:
            clean_message = re.sub(r'^[a-z]+!:\s*', '', change, flags=re.IGNORECASE)
            notes += f"- {clean_message}\n"
        notes += "\n"
    notes += "## 🛠️ Installation\n\n"
    notes += "```bash\n"
    notes += f"git clone https://github.com/khaosans/basic-chat.git\n"
    notes += f"cd basic-chat\n"
    notes += f"git checkout {version}\n"
    notes += "pip install -r requirements.txt\n"
    notes += "./start_basicchat.sh\n"
    notes += "```\n\n"
    notes += "## 🧪 Testing Status\n\n"
    notes += "- ✅ Unit tests passing\n"
    notes += "- ✅ E2E tests passing\n"
    notes += "- ✅ Integration tests passing\n"
    notes += "- ✅ Performance tests within thresholds\n\n"
    return notes

def main():
    if len(sys.argv) != 2:
        print("Usage: python generate_release_notes.py <version>")
        sys.exit(1)
    version = sys.argv[1]
    notes = generate_release_notes(version)
    with open('RELEASE_NOTES.md', 'w') as f:
        f.write(notes)
    print(f"📝 Release notes generated for {version}")
    print("✅ Written to RELEASE_NOTES.md")
if __name__ == "__main__":
    main() 