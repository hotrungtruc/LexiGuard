"""
Structured Logging Configuration (JSON format) and Langfuse Tracing Integration
"""

import sys
from loguru import logger
from src.core.config import settings

def setup_logging():
    logger.remove()
    
    # Standard format or structured JSON format based on environment
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    logger.add(
        sys.stdout,
        level=settings.LOG_LEVEL,
        format=log_format,
        colorize=True
    )
    
    logger.info(f"Logging initialized with level: {settings.LOG_LEVEL}")

def get_langfuse_handler():
    """Initializes and returns Langfuse callback handler if credentials exist."""
    if settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY:
        from langfuse.callback import CallbackHandler
        return CallbackHandler(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST
        )
    return None
