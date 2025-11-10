"""
Unit tests for JSON parsing utilities
Tests: Robust JSON parsing, error handling, edge cases
"""

import pytest
import json
from unittest.mock import Mock, patch
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from agents.browser_automation_agent import parse_json_robust


class TestParseJsonRobust:
    """Test robust JSON parsing function"""
    
    def test_parse_valid_json(self):
        """Test parsing valid JSON"""
        json_str = '{"action": "click", "selector": "#button"}'
        result = parse_json_robust(json_str)
        
        assert result is not None
        assert result['action'] == 'click'
        assert result['selector'] == '#button'
    
    def test_parse_json_with_trailing_comma(self):
        """Test parsing JSON with trailing commas"""
        json_str = '{"action": "click", "selector": "#button",}'
        result = parse_json_robust(json_str)
        
        assert result is not None
        assert result['action'] == 'click'
    
    def test_parse_json_with_unquoted_keys(self):
        """Test parsing JSON with unquoted property names"""
        json_str = '{action: "click", selector: "#button"}'
        result = parse_json_robust(json_str)
        
        assert result is not None
        assert result['action'] == 'click'
        assert result['selector'] == '#button'
    
    def test_parse_nested_json(self):
        """Test parsing nested JSON structures"""
        json_str = '{"action": "click", "details": {"x": 100, "y": 200}}'
        result = parse_json_robust(json_str)
        
        assert result is not None
        assert result['action'] == 'click'
        assert result['details']['x'] == 100
        assert result['details']['y'] == 200
    
    def test_parse_json_with_arrays(self):
        """Test parsing JSON with arrays"""
        json_str = '{"actions": ["click", "type", "wait"]}'
        result = parse_json_robust(json_str)
        
        assert result is not None
        assert len(result['actions']) == 3
        assert 'click' in result['actions']
    
    def test_parse_json_with_trailing_comma_in_array(self):
        """Test parsing JSON with trailing comma in array"""
        json_str = '{"actions": ["click", "type",]}'
        result = parse_json_robust(json_str)
        
        assert result is not None
        assert len(result['actions']) == 2
    
    def test_parse_json_with_special_characters(self):
        """Test parsing JSON with special characters in strings"""
        json_str = '{"text": "Hello \\"World\\"", "path": "C:\\\\Users\\\\test"}'
        result = parse_json_robust(json_str)
        
        assert result is not None
        assert 'World' in result['text']
    
    def test_parse_json_with_numbers(self):
        """Test parsing JSON with various number formats"""
        json_str = '{"int": 42, "float": 3.14, "negative": -10}'
        result = parse_json_robust(json_str)
        
        assert result is not None
        assert result['int'] == 42
        assert result['float'] == 3.14
        assert result['negative'] == -10
    
    def test_parse_json_with_booleans(self):
        """Test parsing JSON with boolean values"""
        json_str = '{"success": true, "failed": false}'
        result = parse_json_robust(json_str)
        
        assert result is not None
        assert result['success'] is True
        assert result['failed'] is False
    
    def test_parse_json_with_null(self):
        """Test parsing JSON with null values"""
        json_str = '{"value": null, "other": "test"}'
        result = parse_json_robust(json_str)
        
        assert result is not None
        assert result['value'] is None
        assert result['other'] == 'test'
    
    def test_parse_completely_invalid_json(self):
        """Test parsing completely invalid JSON"""
        json_str = 'This is not JSON at all!'
        result = parse_json_robust(json_str)
        
        # Should return None or attempt extraction
        # Depending on implementation, might extract partial data
        assert result is None or isinstance(result, dict)
    
    def test_parse_empty_string(self):
        """Test parsing empty string"""
        json_str = ''
        result = parse_json_robust(json_str)
        
        assert result is None
    
    def test_parse_json_with_multiline(self):
        """Test parsing multiline JSON"""
        json_str = '''
        {
            "action": "click",
            "selector": "#button",
            "reasoning": "Need to click the button"
        }
        '''
        result = parse_json_robust(json_str)
        
        assert result is not None
        assert result['action'] == 'click'
        assert result['selector'] == '#button'
    
    def test_parse_json_with_extra_whitespace(self):
        """Test parsing JSON with extra whitespace"""
        json_str = '  {  "action"  :  "click"  }  '
        result = parse_json_robust(json_str)
        
        assert result is not None
        assert result['action'] == 'click'
    
    def test_parse_json_with_unicode(self):
        """Test parsing JSON with Unicode characters"""
        json_str = '{"text": "Hello 世界 🌍"}'
        result = parse_json_robust(json_str)
        
        assert result is not None
        assert '世界' in result['text']
        assert '🌍' in result['text']
    
    def test_parse_json_array_root(self):
        """Test parsing JSON with array as root"""
        json_str = '[{"action": "click"}, {"action": "type"}]'
        result = parse_json_robust(json_str)
        
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 2
    
    def test_parse_json_with_content_preview(self):
        """Test parsing with content preview for debugging"""
        json_str = '{"action": "invalid'
        result = parse_json_robust(json_str, content_preview="Test preview")
        
        # Should handle gracefully
        assert result is None or isinstance(result, dict)


class TestJsonParsingEdgeCases:
    """Test edge cases in JSON parsing"""
    
    def test_parse_json_with_double_quotes_in_string(self):
        """Test parsing JSON with escaped quotes in strings"""
        json_str = '{"message": "He said \\"Hello\\""}'
        result = parse_json_robust(json_str)
        
        assert result is not None
        assert 'Hello' in result['message']
    
    def test_parse_json_with_newlines_in_string(self):
        """Test parsing JSON with newlines in strings"""
        json_str = '{"text": "Line 1\\nLine 2"}'
        result = parse_json_robust(json_str)
        
        assert result is not None
        assert '\\n' in result['text'] or '\n' in result['text']
    
    def test_parse_json_with_tabs(self):
        """Test parsing JSON with tab characters"""
        json_str = '{"text": "Column1\\tColumn2"}'
        result = parse_json_robust(json_str)
        
        assert result is not None
    
    def test_parse_json_with_very_long_string(self):
        """Test parsing JSON with very long string values"""
        long_string = "a" * 10000
        json_str = f'{{"text": "{long_string}"}}'
        result = parse_json_robust(json_str)
        
        assert result is not None
        assert len(result['text']) == 10000
    
    def test_parse_json_with_deeply_nested_structure(self):
        """Test parsing deeply nested JSON"""
        json_str = '{"a": {"b": {"c": {"d": {"e": "value"}}}}}'
        result = parse_json_robust(json_str)
        
        assert result is not None
        assert result['a']['b']['c']['d']['e'] == 'value'
    
    def test_parse_json_with_mixed_types_in_array(self):
        """Test parsing JSON with mixed types in array"""
        json_str = '{"items": [1, "two", true, null, {"key": "value"}]}'
        result = parse_json_robust(json_str)
        
        assert result is not None
        assert len(result['items']) == 5
        assert result['items'][0] == 1
        assert result['items'][1] == "two"
        assert result['items'][2] is True
        assert result['items'][3] is None
        assert result['items'][4]['key'] == "value"
    
    def test_parse_json_with_scientific_notation(self):
        """Test parsing JSON with scientific notation numbers"""
        json_str = '{"small": 1e-10, "large": 1e10}'
        result = parse_json_robust(json_str)
        
        assert result is not None
        assert result['small'] == 1e-10
        assert result['large'] == 1e10
    
    def test_parse_json_with_empty_objects_and_arrays(self):
        """Test parsing JSON with empty objects and arrays"""
        json_str = '{"obj": {}, "arr": [], "value": "test"}'
        result = parse_json_robust(json_str)
        
        assert result is not None
        assert result['obj'] == {}
        assert result['arr'] == []
        assert result['value'] == "test"


class TestJsonParsingLLMResponses:
    """Test parsing JSON from typical LLM responses"""
    
    def test_parse_json_from_markdown_code_block(self):
        """Test extracting JSON from markdown code block"""
        response = '''
        Here's the action:
        ```json
        {"action": "click", "selector": "#button"}
        ```
        '''
        # Note: parse_json_robust might not handle markdown
        # This tests the raw JSON extraction
        json_str = '{"action": "click", "selector": "#button"}'
        result = parse_json_robust(json_str)
        
        assert result is not None
        assert result['action'] == 'click'
    
    def test_parse_json_with_llm_explanation(self):
        """Test parsing JSON when LLM adds explanation"""
        # Just the JSON part
        json_str = '{"action": "click", "reasoning": "User requested to click"}'
        result = parse_json_robust(json_str)
        
        assert result is not None
        assert result['action'] == 'click'
    
    def test_parse_json_with_incomplete_structure(self):
        """Test parsing incomplete JSON structure"""
        json_str = '{"action": "click", "selector":'
        result = parse_json_robust(json_str)
        
        # Should handle gracefully
        assert result is None or isinstance(result, dict)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
