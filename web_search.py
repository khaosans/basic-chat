"""
Web search functionality using DuckDuckGo
"""

from typing import List, Dict, Optional
from duckduckgo_search import DDGS
import time
from dataclasses import dataclass

@dataclass
class SearchResult:
    """Structure for search results"""
    title: str
    link: str
    snippet: str

class WebSearch:
    def __init__(self):
        """Initialize the web search with DuckDuckGo"""
        self.ddgs = DDGS()
        self.max_results = 5
        self.region = 'wt-wt'  # Worldwide results
        
    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """
        Perform a web search using DuckDuckGo
        
        Args:
            query: The search query
            max_results: Maximum number of results to return (default: 5)
            
        Returns:
            List of SearchResult objects containing search results
        """
        if not query.strip():
            return []
            
        try:
            results = []
            search_results = self.ddgs.text(
                query,
                region=self.region,
                max_results=max_results
            )
            
            # Convert generator to list to handle potential errors
            search_results = list(search_results)
            
            for r in search_results:
                results.append(SearchResult(
                    title=r.get('title', 'No title'),
                    link=r.get('link', ''),
                    snippet=r.get('body', 'No description available')
                ))
                
            return results
        except Exception as e:
            print(f"Search error: {str(e)}")
            return []
    
    def format_results(self, results: List[SearchResult]) -> str:
        """
        Format search results into a readable string
        
        Args:
            results: List of SearchResult objects
            
        Returns:
            Formatted string with search results
        """
        if not results:
            return "No results found."
            
        formatted = "Search Results:\n\n"
        for i, result in enumerate(results, 1):
            formatted += f"{i}. **{result.title}**\n"
            formatted += f"   {result.snippet}\n"
            formatted += f"   [Link]({result.link})\n\n"
            
        return formatted

def search_web(query: str, max_results: int = 5) -> str:
    """
    Convenience function to search and format results in one call
    
    Args:
        query: The search query
        max_results: Maximum number of results to return
        
    Returns:
        Formatted string with search results
    """
    searcher = WebSearch()
    results = searcher.search(query, max_results)
    return searcher.format_results(results) 