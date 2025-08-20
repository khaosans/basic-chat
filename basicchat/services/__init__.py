"""
External service integrations for BasicChat.

This module contains integrations with external services like Ollama,
web search, and document processing.
"""

from .ollama_api import check_ollama_server, get_available_models
from .web_search import WebSearch
from .document_processor import DocumentProcessor

__all__ = ["check_ollama_server", "get_available_models", "WebSearch", "DocumentProcessor"]
