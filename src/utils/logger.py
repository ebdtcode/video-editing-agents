"""
Logging utilities for video processing pipeline
"""

import logging
import sys
from pathlib import Path
from typing import Optional


_loggers = {}


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: str = "INFO",
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with file and console handlers

    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string (optional)

    Returns:
        Configured logger instance
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.propagate = False

    # Clear any existing handlers
    logger.handlers.clear()

    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    formatter = logging.Formatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _loggers[name] = logger
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger or create a basic one

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    if name in _loggers:
        return _loggers[name]

    return setup_logger(name)
