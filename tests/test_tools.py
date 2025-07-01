"""
Enhanced tools functionality tests
CHANGELOG:
- Merged test_enhanced_tools.py into dedicated tools test file
- Removed redundant calculation tests, kept edge cases
- Focused on tool integration and error handling
- Added parameterized tests for mathematical operations
"""

import pytest
import math
from unittest.mock import patch
import os
import sys

# Skip if timezone library is missing
pytest.importorskip("pytz", reason="pytz not installed")

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.enhanced_tools import EnhancedCalculator, EnhancedTimeTools, CalculationResult, TimeResult

class TestEnhancedCalculator:
    """Test enhanced calculator functionality"""
    
    def setup_method(self):
        """Setup calculator for each test"""
        self.calculator = EnhancedCalculator()
    
    @pytest.mark.parametrize("expression,expected", [
        ("2 + 2", "4.0"),
        ("10 - 5", "5.0"),
        ("3 * 4", "12.0"),
        ("15 / 3", "5.0"),
        ("2 ** 3", "8.0"),
        ("10 % 3", "1.0"),
        ("sqrt(16)", "4.0"),
        ("abs(-5)", "5.0"),
        ("factorial(5)", "120.0"),
        ("sin(0)", "0.0"),
        ("cos(0)", "1.0"),
        ("log(e)", "1.0"),
        ("pi", "3.1"),  # Rounded in output
        ("e", "2.7")    # Rounded in output
    ])
    def test_should_calculate_basic_expressions(self, expression, expected):
        """Should calculate basic mathematical expressions correctly"""
        result = self.calculator.calculate(expression)
        assert result.success is True
        assert result.result == expected
        assert len(result.steps) > 0
    
    @pytest.mark.parametrize("expression", [
        "2 / 0",
        "sqrt(-1)",
        "factorial(-1)",
        "log(0)",
        "undefined_function(1)"
    ])
    def test_should_handle_calculation_errors(self, expression):
        """Should handle calculation errors gracefully"""
        result = self.calculator.calculate(expression)
        assert result.success is False
        assert result.error is not None
    
    @pytest.mark.parametrize("dangerous_expression", [
        "import os",
        "eval('2+2')",
        "exec('print(1)')",
        "open('file.txt')",
        "__import__('os')",
        "globals()",
        "locals()"
    ])
    def test_should_block_dangerous_expressions(self, dangerous_expression):
        """Should block dangerous expressions for security"""
        result = self.calculator.calculate(dangerous_expression)
        assert result.success is False
        assert "unsafe" in result.error.lower() or "error" in result.error.lower()
    
    def test_should_clean_expression_input(self):
        """Should clean and normalize expression input"""
        test_cases = [
            ("2 × 3", "6.0"),
            ("10 ÷ 2", "5.0"),
            ("5²", "25.0"),
            ("2³", "8.0"),
            ("2^3", "8.0"),
            ("2  +  2", "4.0")
        ]
        
        for expression, expected in test_cases:
            result = self.calculator.calculate(expression)
            assert result.success is True
            assert result.result == expected

class TestEnhancedTimeTools:
    """Test enhanced time tools functionality"""
    
    def setup_method(self):
        """Setup time tools for each test"""
        self.time_tools = EnhancedTimeTools()
    
    @pytest.mark.parametrize("timezone", ["UTC", "EST", "PST", "JST", "IST"])
    def test_should_get_current_time_in_timezones(self, timezone):
        """Should get current time in different timezones"""
        result = self.time_tools.get_current_time(timezone)
        assert result.success is True
        assert result.timezone in self.time_tools.common_timezones.values()
        assert result.unix_timestamp > 0
    
    def test_should_get_current_time_in_gmt(self):
        """Should get current time in GMT timezone"""
        result = self.time_tools.get_current_time("GMT")
        assert result.success is True
        # GMT should be handled gracefully (either mapped to UTC or kept as GMT)
        assert result.timezone in ["UTC", "GMT"] or result.timezone in self.time_tools.common_timezones.values()
        assert result.unix_timestamp > 0
    
    def test_should_convert_time_between_timezones(self):
        """Should convert time between different timezones"""
        result = self.time_tools.convert_time(
            "2024-01-01 12:00:00",
            "UTC",
            "EST"
        )
        assert result.success is True
        assert "EST" in result.current_time or "EDT" in result.current_time
    
    def test_should_calculate_time_differences(self):
        """Should calculate time differences correctly"""
        result = self.time_tools.get_time_difference(
            "2024-01-01 12:00:00",
            "2024-01-01 14:00:00",
            "UTC"
        )
        assert result["success"] is True
        assert result["difference_seconds"] == 7200
        assert result["difference_hours"] == 2.0
    
    def test_should_get_comprehensive_time_info(self):
        """Should get comprehensive time information"""
        result = self.time_tools.get_time_info("UTC")
        assert result["success"] is True
        assert "current_time" in result
        assert "timezone" in result
        assert "unix_timestamp" in result
        assert "year" in result
        assert "month" in result
        assert "day" in result
        assert "weekday" in result
        assert "is_weekend" in result
        assert "is_business_day" in result
    
    def test_should_handle_invalid_timezone(self):
        """Should handle invalid timezone gracefully"""
        result = self.time_tools.get_current_time("INVALID_TIMEZONE")
        assert result.success is True
        assert result.timezone == "UTC"  # Should fallback to UTC
    
    def test_should_handle_invalid_time_format(self):
        """Should handle invalid time format gracefully"""
        result = self.time_tools.convert_time(
            "invalid_time_format",
            "UTC",
            "EST"
        )
        assert result.success is False
        assert result.error is not None
    
    @patch('utils.enhanced_tools.pytz')
    def test_should_handle_pytz_import_errors(self, mock_pytz):
        """Should handle pytz import errors gracefully"""
        mock_pytz.timezone.side_effect = ImportError("pytz not available")
        
        result = self.time_tools.get_current_time("UTC")
        assert result.success is False
        assert result.error is not None

class TestDataStructures:
    """Test data structures for tools"""
    
    def test_should_create_calculation_result(self):
        """Should create CalculationResult with all fields"""
        result = CalculationResult(
            result="42",
            expression="40 + 2",
            steps=["Step 1", "Step 2"],
            success=True
        )
        
        assert result.result == "42"
        assert result.expression == "40 + 2"
        assert len(result.steps) == 2
        assert result.success is True
        assert result.error is None
    
    def test_should_create_time_result(self):
        """Should create TimeResult with all fields"""
        result = TimeResult(
            current_time="2024-01-01 12:00:00 UTC",
            timezone="UTC",
            formatted_time="2024-01-01 12:00:00 UTC",
            unix_timestamp=1704110400.0,
            success=True
        )
        
        assert result.current_time == "2024-01-01 12:00:00 UTC"
        assert result.timezone == "UTC"
        assert result.formatted_time == "2024-01-01 12:00:00 UTC"
        assert result.unix_timestamp == 1704110400.0
        assert result.success is True
        assert result.error is None

class TestToolIntegration:
    """Test integration between tools"""
    
    def test_should_integrate_calculator_and_time_tools(self):
        """Should integrate calculator and time tools"""
        calculator = EnhancedCalculator()
        time_tools = EnhancedTimeTools()
        
        # Get current time
        time_result = time_tools.get_current_time("UTC")
        assert time_result.success is True
        
        # Use time in calculation
        unix_time = time_result.unix_timestamp
        calc_result = calculator.calculate(f"floor({unix_time} / 3600)")
        
        assert calc_result.success is True
        # Convert to int for comparison (remove .0 suffix)
        result_value = float(calc_result.result)
        assert int(result_value) > 0 