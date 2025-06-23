"""
Configuration management for BasicChat application
"""

import os
from dataclasses import dataclass
from typing import Optional, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Core Ollama Configuration ---
OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://localhost:11434/api")

# --- Model Configuration ---
# Specifies the default model for general reasoning tasks.
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "mistral")

# Specifies the model to use for vision-related tasks (e.g., image analysis).
VISION_MODEL = os.environ.get("VISION_MODEL", "llava")

# Specifies the model used for generating text embeddings.
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")

# --- Reasoning Engine Configuration ---
# Defines the list of available reasoning modes in the UI.
REASONING_MODES: List[str] = [
    "Auto",
    "Standard",
    "Chain-of-Thought",
    "Multi-Step",
    "Agent-Based"
]

# Sets the default reasoning mode for the application.
# Must be one of the values from REASONING_MODES.
DEFAULT_REASONING_MODE = os.environ.get("DEFAULT_REASONING_MODE", "Auto")
if DEFAULT_REASONING_MODE not in REASONING_MODES:
    raise ValueError(f"Invalid DEFAULT_REASONING_MODE: '{DEFAULT_REASONING_MODE}'. "
                     f"Must be one of {REASONING_MODES}")

# --- Performance & Reliability ---
# Enables or disables caching for faster responses on repeated queries.
ENABLE_CACHING = os.environ.get("ENABLE_CACHING", "true").lower() == "true"
CACHE_TTL = int(os.environ.get("CACHE_TTL", "3600"))  # Time-to-live for cache in seconds
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "30")) # Request timeout in seconds

# --- Redis Configuration (Optional) ---
REDIS_ENABLED = os.environ.get("REDIS_ENABLED", "false").lower() == "true"
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")

# --- Task Queue Configuration ---
ENABLE_BACKGROUND_TASKS = os.environ.get("ENABLE_BACKGROUND_TASKS", "true").lower() == "true"
CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
MAX_TASK_DURATION = int(os.environ.get("MAX_TASK_DURATION", "300"))  # 5 minutes
TASK_CLEANUP_INTERVAL = int(os.environ.get("TASK_CLEANUP_INTERVAL", "3600"))  # 1 hour

# --- UI Configuration ---
# Defines the title of the web application.
APP_TITLE = "BasicChat"

# Defines the path to the favicon for the web application.
FAVICON_PATH = "assets/brand/favicon/favicon-32x32.png"

@dataclass
class AppConfig:
    """Application configuration with environment variable support"""
    
    # Ollama Configuration
    ollama_url: str = OLLAMA_API_URL
    ollama_model: str = DEFAULT_MODEL
    
    # LLM Parameters
    max_tokens: int = int(os.getenv("MAX_TOKENS", "2048"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
    
    # Caching Configuration
    cache_ttl: int = CACHE_TTL
    cache_maxsize: int = int(os.getenv("CACHE_MAXSIZE", "1000"))
    enable_caching: bool = ENABLE_CACHING
    
    # Performance Configuration
    enable_streaming: bool = os.getenv("ENABLE_STREAMING", "true").lower() == "true"
    request_timeout: int = REQUEST_TIMEOUT
    connect_timeout: int = int(os.getenv("CONNECT_TIMEOUT", "5"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
    rate_limit: int = int(os.getenv("RATE_LIMIT", "10"))
    rate_limit_period: int = int(os.getenv("RATE_LIMIT_PERIOD", "1"))
    
    # Redis Configuration (for distributed caching)
    redis_url: Optional[str] = REDIS_URL
    redis_enabled: bool = REDIS_ENABLED
    
    # Task Queue Configuration
    enable_background_tasks: bool = ENABLE_BACKGROUND_TASKS
    celery_broker_url: str = CELERY_BROKER_URL
    celery_result_backend: str = CELERY_RESULT_BACKEND
    max_task_duration: int = MAX_TASK_DURATION
    task_cleanup_interval: int = TASK_CLEANUP_INTERVAL
    
    # Logging Configuration
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    enable_structured_logging: bool = os.getenv("ENABLE_STRUCTURED_LOGGING", "true").lower() == "true"
    
    # Vector Store Configuration
    vectorstore_persist_directory: str = os.getenv("VECTORSTORE_DIR", "./chroma_db")
    embedding_model: str = EMBEDDING_MODEL
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create configuration from environment variables"""
        return cls()
    
    def get_ollama_base_url(self) -> str:
        """Get Ollama base URL without /api suffix"""
        return self.ollama_url.replace("/api", "")
    
    def validate(self) -> bool:
        """Validate configuration values"""
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")
        if self.max_tokens < 1:
            raise ValueError("Max tokens must be positive")
        if self.cache_ttl < 0:
            raise ValueError("Cache TTL must be non-negative")
        if self.max_task_duration < 60:
            raise ValueError("Max task duration must be at least 60 seconds")
        return True

# Global configuration instance
config = AppConfig.from_env() 