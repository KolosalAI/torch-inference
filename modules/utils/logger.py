import logging
import sys
from typing import Optional

def setup_logging(name: str = "root", level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Sets up and returns a logger with the specified name and level.
    
    Args:
        name (str): The name of the logger.
        level (str): Logging level (e.g. "DEBUG", "INFO").
        log_file (Optional[str]): If provided, logs will also be written to this file.
    
    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    
    # Avoid adding multiple handlers if they already exist.
    if logger.hasHandlers():
        return logger

    # Normalize level to uppercase
    logger.setLevel(level.upper())
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Stream (console) handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    # File handler (if log_file is provided)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logger.propagate = False  # Prevent messages from being propagated to the root logger.
    
    return logger
