#!/usr/bin/env python3
"""
Enhanced health check for production deployments
"""
import asyncio
import aiohttp
import json
import sys
from typing import Dict
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class ProductionHealthCheck:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = None
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    async def check_endpoint(self, endpoint: str, expected_status: int = 200) -> Dict:
        url = f"{self.base_url}{endpoint}"
        try:
            async with self.session.get(url, timeout=10) as response:
                return {
                    'endpoint': endpoint,
                    'status': response.status,
                    'healthy': response.status == expected_status,
                    'response_time': response.headers.get('X-Response-Time', 'N/A')
                }
        except Exception as e:
            return {
                'endpoint': endpoint,
                'status': 'error',
                'healthy': False,
                'error': str(e)
            }
    async def run_health_checks(self) -> Dict:
        endpoints = [
            '/',
            '/_stcore/health',
            '/api/health',
        ]
        tasks = [self.check_endpoint(ep) for ep in endpoints]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        health_status = {
            'timestamp': asyncio.get_event_loop().time(),
            'base_url': self.base_url,
            'overall_healthy': True,
            'checks': results
        }
        for result in results:
            if isinstance(result, dict) and not result.get('healthy', True):
                health_status['overall_healthy'] = False
                break
        return health_status
async def main():
    if len(sys.argv) != 2:
        print("Usage: python production_health_check.py <base_url>")
        sys.exit(1)
    base_url = sys.argv[1]
    async with ProductionHealthCheck(base_url) as health_checker:
        results = await health_checker.run_health_checks()
        print(json.dumps(results, indent=2))
        if results['overall_healthy']:
            logger.info("✅ All health checks passed!")
            sys.exit(0)
        else:
            logger.error("❌ Some health checks failed!")
            sys.exit(1)
if __name__ == "__main__":
    asyncio.run(main()) 