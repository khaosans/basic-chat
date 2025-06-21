"""
Web search functionality using DuckDuckGo with improved error handling
"""

from typing import List, Dict, Optional, Tuple
from duckduckgo_search import DDGS
import time
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
import requests
import json

@dataclass
class SearchResult:
    """Represents a search result"""
    title: str
    url: str
    snippet: str
    source: str

class WebSearch:
    def __init__(self) -> None:
        """Initialize the web search with DuckDuckGo"""
        self.ddgs = DDGS()
        self.max_results = 5
        self.region = 'wt-wt'  # Worldwide results
        self.retry_attempts = 3
        self.retry_delay = 2  # seconds
        self.cache: Dict[str, Tuple[List[SearchResult], datetime]] = {}
        self.cache_duration = timedelta(minutes=5)  # Cache for 5 minutes
        
    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """
        Perform a web search using DuckDuckGo with retry logic and caching
        
        Args:
            query: The search query
            max_results: Maximum number of results to return (default: 5)
            
        Returns:
            List of SearchResult objects containing search results
        """
        if not query.strip():
            return []
        
        # Check cache first
        cache_key = f"{query}_{max_results}"
        if cache_key in self.cache:
            cached_time, cached_results = self.cache[cache_key]
            if datetime.now() - cached_time < self.cache_duration:
                print(f"Returning cached results for: {query}")
                return cached_results
            else:
                # Remove expired cache entry
                del self.cache[cache_key]
            
        for attempt in range(self.retry_attempts):
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
                        url=r.get('link', ''),
                        snippet=r.get('body', 'No description available'),
                        source='DuckDuckGo'
                    ))
                    
                if results:
                    # Cache successful results
                    self.cache[cache_key] = (datetime.now(), results)
                    return results
                    
            except Exception as e:
                error_msg = str(e)
                print(f"Search attempt {attempt + 1} failed: {error_msg}")
                
                # If it's a rate limit, wait longer
                if "rate" in error_msg.lower() or "429" in error_msg or "202" in error_msg:
                    wait_time = self.retry_delay * (attempt + 1) + random.uniform(0, 1)
                    print(f"Rate limited, waiting {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                else:
                    # For other errors, wait a bit and retry
                    time.sleep(1)
        
        # If all attempts failed, return fallback results
        fallback_results = self._get_fallback_results(query)
        # Cache fallback results for a shorter time
        self.cache[cache_key] = (datetime.now(), fallback_results)
        return fallback_results
    
    def _get_fallback_results(self, query: str) -> List[SearchResult]:
        """Provide fallback results when search fails"""
        fallback_results = [
            SearchResult(
                title=f"Search for '{query}'",
                url="https://duckduckgo.com",
                snippet=f"Unable to perform real-time search for '{query}'. Please try again later or visit DuckDuckGo directly.",
                source='DuckDuckGo'
            ),
            SearchResult(
                title="Search Temporarily Unavailable",
                url="https://duckduckgo.com",
                snippet="The search service is currently experiencing high traffic. Please try again in a few minutes.",
                source='DuckDuckGo'
            )
        ]
        return fallback_results
    
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
            formatted += f"   [Link]({result.url})\n\n"
            
        return formatted

def search_web(query: str, num_results: int = 5) -> List[SearchResult]:
    """
    Perform web search using DuckDuckGo Instant Answer API
    
    Args:
        query: Search query string
        num_results: Number of results to return (max 10)
    
    Returns:
        List of SearchResult objects
    """
    try:
        # Use DuckDuckGo Instant Answer API
        url = "https://api.duckduckgo.com/"
        params = {
            'q': query,
            'format': 'json',
            'no_html': '1',
            'skip_disambig': '1'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        # Extract abstract if available
        if data.get('Abstract'):
            results.append(SearchResult(
                title=data.get('AbstractText', 'No title'),
                url=data.get('AbstractURL', ''),
                snippet=data.get('Abstract', ''),
                source='DuckDuckGo Abstract'
            ))
        
        # Extract related topics
        for topic in data.get('RelatedTopics', [])[:num_results-1]:
            if isinstance(topic, dict) and topic.get('Text'):
                results.append(SearchResult(
                    title=topic.get('Text', 'No title')[:100],
                    url=topic.get('FirstURL', ''),
                    snippet=topic.get('Text', ''),
                    source='DuckDuckGo Related'
                ))
        
        return results
        
    except requests.RequestException as e:
        print(f"Search error: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return [] 