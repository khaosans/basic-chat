"""
Async Ollama API client for non-blocking operations
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union
import aiohttp
from aiohttp import ClientSession, TCPConnector
from asyncio import Lock
import time

logger = logging.getLogger(__name__)

class AsyncOllamaClient:
    """Async client for Ollama API operations"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model_name: Optional[str] = None):
        """
        Initialize async Ollama client
        
        Args:
            base_url: Ollama server base URL
            model_name: Default model name to use
        """
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name or "mistral"
        self.session: Optional[ClientSession] = None
        self.connector: Optional[TCPConnector] = None
        self.throttler: Optional[asyncio.Semaphore] = None
        self.lock: Optional[Lock] = None
        self._initialized = False
    
    async def _ensure_initialized(self) -> None:
        """Ensure the client is properly initialized"""
        if not self._initialized:
            await self.initialize()
    
    async def initialize(self) -> None:
        """Initialize the async client"""
        if self._initialized:
            return
        
        # Create connector with connection pooling
        self.connector = TCPConnector(
            limit=100,
            limit_per_host=30,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        # Create throttler for rate limiting
        self.throttler = asyncio.Semaphore(10)  # Max 10 concurrent requests
        
        # Create lock for thread safety
        self.lock = Lock()
        
        # Create session
        self.session = ClientSession(
            connector=self.connector,
            timeout=aiohttp.ClientTimeout(total=60)
        )
        
        self._initialized = True
        logger.info("Async Ollama client initialized")
    
    async def _get_session(self) -> ClientSession:
        """Get the current session, initializing if necessary"""
        await self._ensure_initialized()
        if self.session is None:
            raise RuntimeError("Session not initialized")
        return self.session
    
    async def close(self) -> None:
        """Close the client and cleanup resources"""
        if self.session:
            async with self.lock or Lock():
                await self.session.close()
                self.session = None
        
        if self.connector:
            await self.connector.close()
            self.connector = None
        
        self._initialized = False
        logger.info("Async Ollama client closed")
    
    async def health_check(self) -> bool:
        """Check if Ollama server is healthy"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/api/tags") as response:
                return response.status == 200
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False
    
    async def generate(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Generate text using Ollama API
        
        Args:
            prompt: Input prompt
            model_name: Model to use (uses default if None)
            **kwargs: Additional generation parameters
            
        Returns:
            Generation response
        """
        model = model_name or self.model_name
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            **kwargs
        }
        
        try:
            session = await self._get_session()
            async with self.throttler or asyncio.Semaphore(1):
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                ) as response:
                    response.raise_for_status()
                    return await response.json()
                    
        except aiohttp.ClientError as e:
            logger.error(f"Request error: {e}")
            raise
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise
    
    async def generate_stream(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """
        Generate text with streaming response
        
        Args:
            prompt: Input prompt
            model_name: Model to use (uses default if None)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        model = model_name or self.model_name
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            **kwargs
        }
        
        try:
            session = await self._get_session()
            full_response = ""
            
            async with self.throttler or asyncio.Semaphore(1):
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                ) as response:
                    response.raise_for_status()
                    
                    async for line in response.content:
                        if line:
                            try:
                                data = json.loads(line.decode().strip())
                                if 'response' in data:
                                    full_response += data['response']
                                
                                # Check if done
                                if data.get('done', False):
                                    break
                                    
                            except json.JSONDecodeError:
                                continue
            
            return full_response
            
        except aiohttp.ClientError as e:
            logger.error(f"Stream request error: {e}")
            raise
        except Exception as e:
            logger.error(f"Stream generation error: {e}")
            raise
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model_name: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Chat with Ollama model
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model_name: Model to use (uses default if None)
            **kwargs: Additional chat parameters
            
        Returns:
            Chat response
        """
        model = model_name or self.model_name
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            **kwargs
        }
        
        try:
            session = await self._get_session()
            async with self.throttler or asyncio.Semaphore(1):
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json=payload
                ) as response:
                    response.raise_for_status()
                    return await response.json()
                    
        except aiohttp.ClientError as e:
            logger.error(f"Chat request error: {e}")
            raise
        except Exception as e:
            logger.error(f"Chat error: {e}")
            raise
    
    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model_name: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """
        Chat with streaming response
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model_name: Model to use (uses default if None)
            **kwargs: Additional chat parameters
            
        Returns:
            Generated response text
        """
        model = model_name or self.model_name
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            **kwargs
        }
        
        try:
            session = await self._get_session()
            full_response = ""
            
            async with self.throttler or asyncio.Semaphore(1):
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json=payload
                ) as response:
                    response.raise_for_status()
                    
                    async for line in response.content:
                        if line:
                            try:
                                data = json.loads(line.decode().strip())
                                if 'message' in data and 'content' in data['message']:
                                    full_response += data['message']['content']
                                
                                # Check if done
                                if data.get('done', False):
                                    break
                                    
                            except json.JSONDecodeError:
                                continue
            
            return full_response
            
        except aiohttp.ClientError as e:
            logger.error(f"Chat stream request error: {e}")
            raise
        except Exception as e:
            logger.error(f"Chat stream error: {e}")
            raise
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """Get list of available models"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/api/tags") as response:
                response.raise_for_status()
                data = await response.json()
                return data.get('models', [])
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    async def get_model_info(self, model_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get information about a specific model"""
        model = model_name or self.model_name
        
        try:
            session = await self._get_session()
            async with session.post(
                f"{self.base_url}/api/show",
                json={"name": model}
            ) as response:
                if response.status == 200:
                    return await response.json()
                return None
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return None
    
    async def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama library"""
        try:
            session = await self._get_session()
            async with session.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name}
            ) as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Error pulling model: {e}")
            return False

# Global client instance
_async_client: Optional[AsyncOllamaClient] = None

async def get_async_client() -> AsyncOllamaClient:
    """Get the global async Ollama client instance"""
    global _async_client
    if _async_client is None:
        _async_client = AsyncOllamaClient()
        await _async_client.initialize()
    return _async_client

async def close_async_client() -> None:
    """Close the global async Ollama client"""
    global _async_client
    if _async_client:
        await _async_client.close()
        _async_client = None 