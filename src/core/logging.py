"""
Centralized logging configuration for PyTorch Inference Framework.
"""

import os
import sys
import logging
import logging.handlers
from pathlib import Path
from typing import Optional
from datetime import datetime

from .config import get_config


def setup_logging() -> None:
    """Setup centralized logging configuration."""
    config = get_config()
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Get log level from configuration
    log_level = getattr(logging, config.server.log_level.upper(), logging.INFO)
    
    # Clear any existing handlers
    logging.getLogger().handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # File handlers
    if getattr(config, 'logging', {}).get('file_logging', True):
        # Main log file
        main_handler = logging.handlers.RotatingFileHandler(
            log_dir / "server.log",
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=5,
            encoding='utf-8'
        )
        main_handler.setLevel(log_level)
        main_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(main_handler)
        
        # Error log file
        error_handler = logging.handlers.RotatingFileHandler(
            log_dir / "server_errors.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=3,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(error_handler)
    
    # Create API logger
    api_logger = logging.getLogger("api_requests")
    api_logger.setLevel(logging.INFO)
    
    if getattr(config, 'logging', {}).get('file_logging', True):
        api_handler = logging.handlers.RotatingFileHandler(
            log_dir / "api_requests.log",
            maxBytes=20 * 1024 * 1024,  # 20MB
            backupCount=3,
            encoding='utf-8'
        )
        api_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        api_logger.addHandler(api_handler)
        api_logger.propagate = False  # Prevent double logging
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info("PYTORCH INFERENCE FRAMEWORK SERVER STARTUP")
    logger.info("="*80)
    logger.info(f"Startup time: {datetime.now().isoformat()}")
    logger.info(f"Environment: {config.environment}")
    logger.info(f"Log level: {config.server.log_level}")
    logger.info(f"Log files directory: {log_dir.absolute()}")


def get_api_logger() -> logging.Logger:
    """Get the API requests logger."""
    return logging.getLogger("api_requests")
