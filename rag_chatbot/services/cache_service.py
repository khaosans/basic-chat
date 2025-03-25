from functools import lru_cache
from typing import Optional, Dict, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)

class CacheService:
    def __init__(self):
        self.embedding_cache = {}
        self.response_cache = {}
        self.document_cache = {}
        logger.info("✅ Cache service initialized")
    
    @lru_cache(maxsize=1000)
    def get_cached_embedding(self, text_hash: str) -> Optional[np.ndarray]:
        """Get cached embedding by text hash"""
        return self.embedding_cache.get(text_hash)
    
    def cache_embedding(self, text: str, embedding: np.ndarray) -> None:
        """Cache embedding for text"""
        text_hash = hash(text)
        self.embedding_cache[text_hash] = embedding
    
    def cache_response(self, query: str, response: str) -> None:
        """Cache response for query"""
        query_hash = hash(query)
        self.response_cache[query_hash] = response
    
    def get_cached_response(self, query: str) -> Optional[str]:
        """Get cached response for query"""
        query_hash = hash(query)
        return self.response_cache.get(query_hash)
    
    def cache_document(self, doc_id: str, metadata: Dict[str, Any]) -> None:
        """Cache document metadata"""
        self.document_cache[doc_id] = metadata
    
    def get_cached_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get cached document metadata"""
        return self.document_cache.get(doc_id)
    
    def clear_caches(self) -> None:
        """Clear all caches"""
        self.embedding_cache.clear()
        self.response_cache.clear()
        self.document_cache.clear()
        # Also clear the lru_cache
        self.get_cached_embedding.cache_clear()
        logger.info("✅ All caches cleared") 