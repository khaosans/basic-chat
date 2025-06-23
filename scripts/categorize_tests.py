#!/usr/bin/env python3
"""
Test categorization script for BasicChat.

This script helps categorize existing tests with appropriate pytest markers
for parallel execution and better CI performance.
"""

import os
import re
from pathlib import Path

# Test categorization rules
TEST_CATEGORIES = {
    'unit': [
        'test_core.py',
        'test_audio.py',
        'test_enhanced_tools.py',
        'test_config.py'
    ],
    'integration': [
        'test_document_processing.py',
        'test_documents.py',
        'test_reasoning.py',
        'test_web_search.py'
    ],
    'slow': [
        'test_llm_judge.py',
        'test_openai_evaluation.py',
        'test_github_models.py'
    ],
    'isolated': [
        'test_upload.py',
        'test_voice.py'
    ]
}

# Keywords that indicate test type
KEYWORDS = {
    'unit': ['mock', 'patch', 'fast', 'simple', 'basic', 'initialization'],
    'integration': ['database', 'file', 'network', 'api', 'external', 'real'],
    'slow': ['llm', 'openai', 'github', 'judge', 'evaluation', 'model'],
    'isolated': ['upload', 'file_system', 'temp', 'cleanup', 'isolation']
}

def categorize_test_file(file_path):
    """Categorize a test file based on its content and name"""
    content = file_path.read_text()
    file_name = file_path.name
    
    # Check explicit categorization first
    for category, files in TEST_CATEGORIES.items():
        if file_name in files:
            return category
    
    # Analyze content for keywords
    category_scores = {cat: 0 for cat in KEYWORDS.keys()}
    
    for category, keywords in KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in content.lower():
                category_scores[category] += 1
    
    # Return category with highest score, default to 'unit'
    if max(category_scores.values()) > 0:
        return max(category_scores, key=category_scores.get)
    
    return 'unit'  # Default to unit tests

def add_markers_to_file(file_path, category):
    """Add appropriate pytest markers to a test file"""
    content = file_path.read_text()
    
    # Check if markers already exist
    if '@pytest.mark.' in content:
        print(f"⚠️  {file_path.name} already has markers, skipping...")
        return
    
    # Add markers to class definitions
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        new_lines.append(line)
        
        # Add markers after class definitions
        if line.strip().startswith('class ') and 'Test' in line:
            indent = len(line) - len(line.lstrip())
            marker_indent = ' ' * (indent + 4)
            new_lines.append(f'{marker_indent}@pytest.mark.{category}')
            new_lines.append(f'{marker_indent}@pytest.mark.fast' if category == 'unit' else f'{marker_indent}@pytest.mark.{category}')
    
    # Write back to file
    file_path.write_text('\n'.join(new_lines))
    print(f"✅ Added {category} markers to {file_path.name}")

def main():
    """Main categorization function"""
    tests_dir = Path('tests')
    
    if not tests_dir.exists():
        print("❌ Tests directory not found")
        return
    
    print("🔍 Categorizing test files...")
    print("=" * 50)
    
    for test_file in tests_dir.glob('test_*.py'):
        if test_file.name.startswith('__'):
            continue
            
        category = categorize_test_file(test_file)
        print(f"📁 {test_file.name} → {category}")
        
        # Add markers
        add_markers_to_file(test_file, category)
    
    print("\n" + "=" * 50)
    print("📊 Test Categories Summary:")
    print("=" * 50)
    
    for category, files in TEST_CATEGORIES.items():
        print(f"\n{category.upper()} Tests:")
        for file in files:
            file_path = tests_dir / file
            if file_path.exists():
                print(f"  ✅ {file}")
            else:
                print(f"  ❌ {file} (not found)")
    
    print("\n🚀 Next Steps:")
    print("1. Run: pytest tests/ -m 'unit or fast' -n auto")
    print("2. Run: pytest tests/ -m 'integration' -n auto")
    print("3. Run: pytest tests/ -m 'slow' -n 0")
    print("4. Run: pytest tests/ -m 'isolated' -n 0")

if __name__ == "__main__":
    main() 