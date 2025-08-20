"""
Web search functionality tests
CHANGELOG:
- Streamlined test_web_search.py
- Removed redundant mock tests
- Focused on core search functionality and error handling
- Added parameterized tests for different search scenarios
"""

import pytest
from unittest.mock import patch, MagicMock
from basicchat.services.web_search import SearchResult, search_web, WebSearch

class TestWebSearch:
    """Test web search functionality"""
    @pytest.mark.integration
    @pytest.mark.integration
    
    def setup_method(self):
        """Setup for each test"""
        self.test_query = "Python programming"
    
    @patch('basicchat.services.web_search.DDGS')
    def test_should_perform_basic_search(self, mock_ddgs):
        """Should perform basic web search successfully"""
        # Mock successful search results
        mock_results = [
            {
                'title': 'Python Programming',
                'link': 'https://python.org',
                'body': 'Python is a programming language'
            }
        ]
        
        mock_instance = MagicMock()
        mock_instance.text.return_value = mock_results
        mock_ddgs.return_value = mock_instance
        
        results = search_web(self.test_query)
        
        assert isinstance(results, str)
        assert "Python Programming" in results
        assert "Search Results:" in results
    
    def test_should_handle_empty_query(self):
        """Should handle empty query gracefully"""
        results = search_web("")
        assert results == "No results found."
    
    @patch('basicchat.services.web_search.DDGS')
    def test_should_respect_max_results_parameter(self, mock_ddgs):
        """Should respect max_results parameter"""
        # Mock many results
        mock_results = [
            {
                'title': f'Result {i}',
                'link': f'https://example{i}.com',
                'body': f'Snippet {i}'
            }
            for i in range(10)
        ]
        
        mock_instance = MagicMock()
        mock_instance.text.return_value = mock_results
        mock_ddgs.return_value = mock_instance
        
        results = search_web(self.test_query, max_results=3)
        
        # Count numbered results in formatted output
        result_count = results.count("1. **") + results.count("2. **") + results.count("3. **")
        assert result_count == 3
    
    @patch('basicchat.services.web_search.DDGS')
    def test_should_handle_rate_limit_errors(self, mock_ddgs):
        """Should handle rate limiting gracefully"""
        mock_instance = MagicMock()
        mock_instance.text.side_effect = Exception("Rate limit")
        mock_ddgs.return_value = mock_instance
        
        results = search_web(self.test_query)
        
        assert "Search Results:" in results
        assert "Unable to perform real-time search" in results
    
    def test_should_create_search_result_object(self):
        """Should create SearchResult object correctly"""
        result = SearchResult(
            title="Test Title",
            link="https://test.com",
            snippet="Test snippet"
        )
        
        assert result.title == "Test Title"
        assert result.link == "https://test.com"
        assert result.snippet == "Test snippet"
    
    def test_should_format_search_result_string(self):
        """Should format SearchResult as string correctly"""
        result = SearchResult(
            title="Test Title",
            link="https://test.com",
            snippet="Test snippet"
        )
        
        string_repr = str(result)
        assert "Test Title" in string_repr
        assert "https://test.com" in string_repr
        assert "Test snippet" in string_repr
    
    def test_should_initialize_web_search_class(self):
        """Should initialize WebSearch class correctly"""
        searcher = WebSearch()
        
        assert searcher is not None
        assert searcher.max_results == 5
        assert searcher.region == 'wt-wt' 
