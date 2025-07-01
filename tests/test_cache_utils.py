import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.caching import MemoryCache


def test_memory_cache_basic():
    cache = MemoryCache(ttl=1, maxsize=2)
    assert cache.set('a', '1')
    assert cache.get('a') == '1'
    assert cache.delete('a')
    assert cache.get('a') is None
