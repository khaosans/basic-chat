"""
BasicChat - Your Intelligent Local AI Assistant

A privacy-first, advanced reasoning AI assistant that runs entirely on your local machine.
"""

__version__ = "0.1.0"
__author__ = "Souriya Khaosanga"
__email__ = "sour@chainable.ai"

# Import main components for easy access
from .core.app import main
from .core.config import AppConfig
from .core.reasoning_engine import ReasoningEngine

__all__ = [
    "main",
    "AppConfig", 
    "ReasoningEngine",
    "__version__",
    "__author__",
    "__email__"
]
