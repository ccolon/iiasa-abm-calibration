"""
Logging configuration and utilities.

This module provides centralized logging setup with support for file rotation,
structured logging, and different output formats.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

from ..config import LoggingConfig


def setup_logging(config: LoggingConfig) -> None:
    """
    Set up logging configuration based on provided settings.
    
    Args:
        config: Logging configuration object
    """
    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Set logging level
    level = getattr(logging, config.level.upper())
    root_logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(config.format)
    
    # Set up console handler with Rich for better formatting
    console_handler = RichHandler(
        console=Console(stderr=True),
        show_path=config.level == "DEBUG",
        show_time=True,
        rich_tracebacks=True
    )
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # Set up file handler if specified
    if config.file_path:
        file_path = Path(config.file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            filename=file_path,
            maxBytes=_parse_size(config.rotation_size),
            backupCount=config.retention_days
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set third-party loggers to WARNING to reduce noise
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the specified module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def _parse_size(size_str: str) -> int:
    """
    Parse size string (e.g., '10MB', '1GB') to bytes.
    
    Args:
        size_str: Size string with unit
        
    Returns:
        Size in bytes
    """
    size_str = size_str.upper()
    
    if size_str.endswith('KB'):
        return int(size_str[:-2]) * 1024
    elif size_str.endswith('MB'):
        return int(size_str[:-2]) * 1024 * 1024
    elif size_str.endswith('GB'):
        return int(size_str[:-2]) * 1024 * 1024 * 1024
    elif size_str.endswith('B'):
        return int(size_str[:-1])
    else:
        # Assume bytes if no unit specified
        return int(size_str)


class StructuredLogger:
    """
    Structured logger for consistent log formatting.
    
    This class provides methods for logging with structured data
    that can be easily parsed by log aggregation systems.
    """
    
    def __init__(self, name: str):
        """Initialize structured logger."""
        self.logger = get_logger(name)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message with structured data."""
        extra = {"structured_data": kwargs} if kwargs else {}
        self.logger.info(message, extra=extra)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with structured data."""
        extra = {"structured_data": kwargs} if kwargs else {}
        self.logger.warning(message, extra=extra)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message with structured data."""
        extra = {"structured_data": kwargs} if kwargs else {}
        self.logger.error(message, extra=extra)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with structured data."""
        extra = {"structured_data": kwargs} if kwargs else {}
        self.logger.debug(message, extra=extra)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message with structured data."""
        extra = {"structured_data": kwargs} if kwargs else {}
        self.logger.critical(message, extra=extra)