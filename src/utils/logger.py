"""
Logging utilities for OtitisMediaAI.
"""

import os
import logging
import logging.handlers
import sys
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console_output: bool = True,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> logging.Logger:
    """
    Set up and configure a logger instance.
    
    Args:
        name: Logger name
        log_file: Path to log file (if None, no file logging is set up)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_output: Whether to output logs to the console
        log_format: Format string for log messages
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent adding handlers multiple times
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(log_format)
    
    # Add file handler if log_file is specified
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10485760, backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger by name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)
