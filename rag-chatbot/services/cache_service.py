from functools import lru_cache
from typing import Optional
import numpy as np

class CacheService:
    def __init__(self):
        self.embedding_cache = {}
        self.response_cache = {}
    
    @lru_cache(maxsize=1000)
    async def get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        return self.embedding_cache.get(hash(text))
    
    async def cache_embedding(self, text: str, embedding: np.ndarray):
        self.embedding_cache[hash(text)] = embedding 