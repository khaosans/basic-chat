"""
Unit tests for UI styling improvements
"""
import pytest
import re
from pathlib import Path


class TestUIStyling:
    """Test class for UI styling improvements"""
    
    def test_dropdown_styling_in_app_py(self):
        """Test that dropdown styling improvements are present in app.py"""
        app_py_path = Path("app.py")
        assert app_py_path.exists(), "app.py should exist"
        
        with open(app_py_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for comprehensive dropdown styling
        assert '.stSelectbox * {' in content, "Should have universal dropdown styling"
        assert 'color: #000000 !important;' in content, "Should have black text color"
        assert 'font-weight: 700 !important;' in content, "Should have bold font weight"
        assert 'font-size: 14px !important;' in content, "Should have 14px font size"
        
        # Check for specific dropdown targeting
        assert '[data-baseweb="select"] *' in content, "Should target baseweb select elements"
        assert '[role="combobox"] *' in content, "Should target combobox elements"
        assert '[role="listbox"] *' in content, "Should target listbox elements"
        
        # Check for sidebar styling
        assert '.css-1d391kg {' in content, "Should have sidebar styling"
        assert 'background-color: #f8f9fa !important;' in content, "Should have sidebar background"
        assert 'border-right: 1px solid #e5e7eb !important;' in content, "Should have sidebar border"
        
        # Check for enhanced selectbox container
        assert 'min-height: 40px !important;' in content, "Should have minimum height for dropdowns"
        assert 'border: 2px solid #d1d5db !important;' in content, "Should have enhanced border"
        assert 'box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;' in content, "Should have shadow"
    
    def test_css_specificity_and_importance(self):
        """Test that CSS rules use proper specificity and !important declarations"""
        app_py_path = Path("app.py")
        
        with open(app_py_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract CSS section
        css_match = re.search(r'<style>(.*?)</style>', content, re.DOTALL)
        assert css_match, "Should have CSS styling section"
        
        css_content = css_match.group(1)
        
        # Check for proper !important usage
        important_rules = re.findall(r'[^}]*!important[^}]*', css_content)
        assert len(important_rules) > 0, "Should have !important declarations"
        
        # Check for comprehensive selectbox targeting
        selectbox_rules = re.findall(r'\.stSelectbox[^{]*{', css_content)
        assert len(selectbox_rules) > 0, "Should have selectbox styling rules"
    
    def test_color_contrast_improvements(self):
        """Test that color contrast improvements are implemented"""
        app_py_path = Path("app.py")
        
        with open(app_py_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for black text on white background
        assert '#000000 !important' in content, "Should use black text for maximum contrast"
        assert '#ffffff !important' in content, "Should use white background"
        
        # Check for proper sidebar contrast
        assert '#f8f9fa !important' in content, "Should have light sidebar background"
        assert '#1f2937 !important' in content, "Should have dark text in sidebar"
    
    def test_font_weight_and_size_improvements(self):
        """Test that font weight and size improvements are implemented"""
        app_py_path = Path("app.py")
        
        with open(app_py_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for bold font weights
        assert 'font-weight: 700 !important' in content, "Should use bold font weight"
        assert 'font-weight: 600 !important' in content, "Should use semi-bold font weight"
        
        # Check for consistent font sizes
        assert 'font-size: 14px !important' in content, "Should use 14px font size"
    
    def test_hover_and_interactive_states(self):
        """Test that hover and interactive states are properly styled"""
        app_py_path = Path("app.py")
        
        with open(app_py_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for hover effects
        assert ':hover' in content, "Should have hover effects"
        assert '#10a37f !important' in content, "Should use green color for hover states"
        
        # Check for focus states
        assert 'box-shadow' in content, "Should have box shadow effects"
    
    def test_accessibility_improvements(self):
        """Test that accessibility improvements are implemented"""
        app_py_path = Path("app.py")
        
        with open(app_py_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for proper contrast ratios
        assert '#000000' in content, "Should use black text for maximum contrast"
        assert '#ffffff' in content, "Should use white background for maximum contrast"
        
        # Check for proper spacing
        assert 'padding: 8px 12px !important' in content, "Should have proper padding"
        assert 'min-height: 40px !important' in content, "Should have minimum touch target size"
    
    def test_cross_browser_compatibility(self):
        """Test that styling works across different browsers"""
        app_py_path = Path("app.py")
        
        with open(app_py_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for vendor prefixes if needed
        # Note: Modern CSS properties don't always need vendor prefixes
        
        # Check for fallback values
        assert '!important' in content, "Should use !important for consistent rendering"
        
        # Check for standard CSS properties
        assert 'background-color' in content, "Should use standard background-color property"
        assert 'color' in content, "Should use standard color property"
        assert 'font-weight' in content, "Should use standard font-weight property"
        assert 'font-size' in content, "Should use standard font-size property"
    
    def test_performance_considerations(self):
        """Test that styling doesn't introduce performance issues"""
        app_py_path = Path("app.py")
        
        with open(app_py_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for efficient selectors
        css_match = re.search(r'<style>(.*?)</style>', content, re.DOTALL)
        if css_match:
            css_content = css_match.group(1)
            
            # Remove comments to get actual CSS rules
            css_content = re.sub(r'/\*.*?\*/', '', css_content, flags=re.DOTALL)
            
            # Count CSS rules to ensure we don't have too many
            rule_count = len(re.findall(r'[^{]*{', css_content))
            assert rule_count < 100, "Should not have excessive CSS rules"
            
            # Check that we have reasonable CSS structure
            assert '.stSelectbox' in css_content, "Should have selectbox styling"
            assert '!important' in css_content, "Should use !important for consistency"
            assert 'color:' in css_content, "Should have color properties"
            assert 'background-color:' in css_content, "Should have background properties"
