"""
Caching utilities for the application
Provides in-memory and persistent caching capabilities
"""

import json
import os
import pickle
import time
from typing import Any, Dict, Optional, Union, List
from datetime import datetime, timedelta
import hashlib
from cachetools import TTLCache, LRUCache
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    """Manages caching operations with both memory and disk storage"""
    
    def __init__(self, cache_dir: str = ".cache", max_memory_size: int = 100, ttl: int = 3600):
        """
        Initialize cache manager
        
        Args:
            cache_dir: Directory for persistent cache files
            max_memory_size: Maximum number of items in memory cache
            ttl: Time to live for cache items in seconds
        """
        self.cache_dir = cache_dir
        self.memory_cache: TTLCache[str, Any] = TTLCache(maxsize=max_memory_size, ttl=ttl)
        self.lru_cache: LRUCache[str, Any] = LRUCache(maxsize=max_memory_size)
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0
        }
    
    def _get_cache_key(self, key: str) -> str:
        """Generate a cache key with hash for file naming"""
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> str:
        """Get the full path for a cache file"""
        cache_key = self._get_cache_key(key)
        return os.path.join(self.cache_dir, f"{cache_key}.cache")
    
    def get(self, key: str) -> Optional[str]:
        """
        Get a value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        # Try memory cache first
        if key in self.memory_cache:
            self.stats['hits'] += 1
            return self.memory_cache[key]
        
        # Try LRU cache
        if key in self.lru_cache:
            self.stats['hits'] += 1
            return self.lru_cache[key]
        
        # Try disk cache
        cache_path = self._get_cache_path(key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Check if expired
                if 'expires' in data and data['expires'] < time.time():
                    os.remove(cache_path)
                    self.stats['misses'] += 1
                    return None
                
                # Move to memory cache
                self.memory_cache[key] = data['value']
                self.stats['hits'] += 1
                return data['value']
                
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error reading cache file {cache_path}: {e}")
                # Remove corrupted cache file
                try:
                    os.remove(cache_path)
                except OSError:
                    pass
        
        self.stats['misses'] += 1
        return None
    
    def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        """
        Set a value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
        """
        if ttl is None:
            ttl = 3600  # Default 1 hour
        
        expires = time.time() + ttl
        
        # Store in memory cache
        self.memory_cache[key] = value
        
        # Store in LRU cache for longer retention
        self.lru_cache[key] = value
        
        # Store on disk
        cache_path = self._get_cache_path(key)
        try:
            data = {
                'value': value,
                'expires': expires,
                'created': time.time()
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f)
                
        except IOError as e:
            logger.warning(f"Error writing cache file {cache_path}: {e}")
        
        self.stats['sets'] += 1
    
    def delete(self, key: str) -> None:
        """
        Delete a value from cache
        
        Args:
            key: Cache key to delete
        """
        # Remove from memory caches
        self.memory_cache.pop(key, None)
        self.lru_cache.pop(key, None)
        
        # Remove from disk
        cache_path = self._get_cache_path(key)
        try:
            if os.path.exists(cache_path):
                os.remove(cache_path)
        except OSError as e:
            logger.warning(f"Error deleting cache file {cache_path}: {e}")
        
        self.stats['deletes'] += 1
    
    def clear(self) -> None:
        """Clear all cache data"""
        # Clear memory caches
        self.memory_cache.clear()
        self.lru_cache.clear()
        
        # Clear disk cache
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.cache'):
                    filepath = os.path.join(self.cache_dir, filename)
                    try:
                        os.remove(filepath)
                    except OSError:
                        pass
        except OSError as e:
            logger.warning(f"Error clearing cache directory: {e}")
    
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'sets': self.stats['sets'],
            'deletes': self.stats['deletes'],
            'hit_rate': hit_rate,
            'memory_size': len(self.memory_cache),
            'lru_size': len(self.lru_cache)
        }
    
    def cleanup_expired(self) -> None:
        """Remove expired cache entries from disk"""
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.cache'):
                    filepath = os.path.join(self.cache_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        if 'expires' in data and data['expires'] < time.time():
                            os.remove(filepath)
                            
                    except (json.JSONDecodeError, IOError):
                        # Remove corrupted files
                        try:
                            os.remove(filepath)
                        except OSError:
                            pass
                            
        except OSError as e:
            logger.warning(f"Error cleaning up cache: {e}")

class ConversationCache:
    """Specialized cache for conversation history"""
    
    def __init__(self, max_conversations: int = 50):
        """
        Initialize conversation cache
        
        Args:
            max_conversations: Maximum number of conversations to cache
        """
        self.cache: LRUCache[str, Dict[str, Any]] = LRUCache(maxsize=max_conversations)
        self.conversation_ids: List[str] = []
    
    def store_conversation(self, conversation_id: str, messages: List[Dict[str, Any]]) -> None:
        """
        Store a conversation in cache
        
        Args:
            conversation_id: Unique conversation identifier
            messages: List of conversation messages
        """
        self.cache[conversation_id] = {
            'messages': messages,
            'timestamp': time.time(),
            'message_count': len(messages)
        }
        
        # Update conversation ID list
        if conversation_id not in self.conversation_ids:
            self.conversation_ids.append(conversation_id)
        
        # Remove oldest if we exceed max
        if len(self.conversation_ids) > self.cache.maxsize:
            oldest_id = self.conversation_ids.pop(0)
            self.cache.pop(oldest_id, None)
    
    def get_conversation(self, conversation_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve a conversation from cache
        
        Args:
            conversation_id: Unique conversation identifier
            
        Returns:
            List of conversation messages or None if not found
        """
        if conversation_id in self.cache:
            return self.cache[conversation_id]['messages']
        return None
    
    def add_message(self, conversation_id: str, message: Dict[str, Any]) -> None:
        """
        Add a message to an existing conversation
        
        Args:
            conversation_id: Unique conversation identifier
            message: Message to add
        """
        if conversation_id in self.cache:
            self.cache[conversation_id]['messages'].append(message)
            self.cache[conversation_id]['message_count'] += 1
            self.cache[conversation_id]['timestamp'] = time.time()
    
    def delete_conversation(self, conversation_id: str) -> None:
        """
        Delete a conversation from cache
        
        Args:
            conversation_id: Unique conversation identifier
        """
        self.cache.pop(conversation_id, None)
        if conversation_id in self.conversation_ids:
            self.conversation_ids.remove(conversation_id)
    
    def get_conversation_stats(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get statistics for a conversation
        
        Args:
            conversation_id: Unique conversation identifier
            
        Returns:
            Dictionary with conversation statistics or None if not found
        """
        if conversation_id in self.cache:
            return self.cache[conversation_id]
        return None
    
    def list_conversations(self) -> List[str]:
        """
        Get list of cached conversation IDs
        
        Returns:
            List of conversation IDs
        """
        return self.conversation_ids.copy()
    
    def clear(self) -> None:
        """Clear all cached conversations"""
        self.cache.clear()
        self.conversation_ids.clear()

# Global cache instances
cache_manager = CacheManager()
conversation_cache = ConversationCache()

def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance"""
    return cache_manager

def get_conversation_cache() -> ConversationCache:
    """Get the global conversation cache instance"""
    return conversation_cache 