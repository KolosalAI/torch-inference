import logging
import logging.config  # Added to use dictConfig
import sys
import traceback
from logging.handlers import QueueHandler, QueueListener
from typing import Optional, Callable, Any
from queue import Queue
import inspect
from threading import Thread
from datetime import datetime
import os
import torch  # Added to support torch.nn.Module usage

# Local imports
from .config import LOGGING_CONFIG

class StructuredLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._async_queue: Optional[Queue] = None
        self._async_listener: Optional[QueueListener] = None

    @classmethod
    def configure_logging(cls) -> None:
        """Configure logging using the centralized configuration"""
        try:
            logging.config.dictConfig(LOGGING_CONFIG)
            logging.captureWarnings(True)
        except Exception as e:
            print(f"Failed to configure logging: {str(e)}")
            # Fallback configuration if LOGGING_CONFIG fails
            logging.basicConfig(level=logging.INFO)

    @classmethod
    def get_logger(cls, name: Optional[str] = None) -> "StructuredLogger":
        """Get a logger instance with async support"""
        if not name:
            name = inspect.stack()[1][3]
        return cls(name)

    def _ensure_async(self) -> None:
        """Initialize async logging infrastructure if needed"""
        if self._async_queue is None:
            self._async_queue = Queue(-1)
            handlers = self.logger.handlers.copy()
            self._async_listener = QueueListener(self._async_queue, *handlers)
            self._async_listener.start()

            # Replace existing handlers with queue handler
            for handler in handlers:
                self.logger.removeHandler(handler)
            self.logger.addHandler(QueueHandler(self._async_queue))

    def async_log(self, func: Callable) -> Callable:
        """Decorator for async logging context"""
        def wrapper(*args, **kwargs):
            self._ensure_async()
            return func(*args, **kwargs)
        return wrapper

    def log_execution(self, func: Callable) -> Callable:
        """Decorator to log function entry/exit with timing"""
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            self.log(
                "DEBUG",
                f"Entering {func.__name__}",
                extra={"args": args, "kwargs": kwargs}
            )
            
            try:
                result = func(*args, **kwargs)
                duration = datetime.now() - start_time
                self.log(
                    "DEBUG",
                    f"Exiting {func.__name__}",
                    extra={"result": str(result), "duration": str(duration)}
                )
                return result
            except Exception as e:
                self.log(
                    "ERROR",
                    f"Error in {func.__name__}",
                    extra={"error": str(e), "traceback": traceback.format_exc()}
                )
                raise

        return wrapper

    def log(self, level: str, message: str, **kwargs) -> None:
        """Structured logging with additional context"""
        log_level = getattr(logging, level.upper(), logging.INFO)
        extra = kwargs.get('extra', {})
        exc_info = kwargs.get('exc_info', False)
        
        # Add code context information
        frame = inspect.currentframe().f_back.f_back
        extra.update({
            "file": frame.f_code.co_filename,
            "line": frame.f_lineno,
            "function": frame.f_code.co_name
        })

        self.logger.log(log_level, message, extra=extra, exc_info=exc_info)

    def log_model_summary(self, model: torch.nn.Module, input_shape: tuple) -> None:
        """Log model architecture summary"""
        from torchsummary import summary
        
        try:
            result = []
            def capture_summary(str_: str) -> None:
                result.append(str_)
            
            summary(model, input_shape, print_fn=capture_summary)
            self.log("INFO", "Model Architecture", extra={"summary": "\n".join(result)})
        except Exception as e:
            self.log("ERROR", "Failed to log model summary", extra={"error": str(e)})

    def shutdown(self) -> None:
        """Cleanup logging resources"""
        if self._async_listener:
            self._async_listener.stop()
    def debug(self, message: str, **kwargs) -> None:
        """Log debug level message"""
        self.log("DEBUG", message, **kwargs)
    def info(self, message: str, **kwargs) -> None:
        """Log info level message"""
        self.log("INFO", message, **kwargs)
    def warning(self, message: str, **kwargs) -> None:
        """Log warning level message"""
        self.log("WARNING", message, **kwargs)
    def error(self, message: str, **kwargs) -> None:
        """Log error level message"""
        self.log("ERROR", message, **kwargs)
    def critical(self, message: str, **kwargs) -> None:
        """Log critical level message"""
        self.log("CRITICAL", message, **kwargs)
    def exception(self, message: str, **kwargs) -> None:
        """Log exception with stack trace"""
        self.log("ERROR", message, exc_info=True, **kwargs)
# Initialize default logger configuration
StructuredLogger.configure_logging()

def get_logger(name: Optional[str] = None) -> StructuredLogger:
    """Get a configured StructuredLogger instance"""
    return StructuredLogger.get_logger(name)

if __name__ == "__main__":
    # Example usage
    logger = get_logger(__name__)

    @logger.log_execution
    def sample_function():
        logger.log("INFO", "Sample info message")
        logger.log("ERROR", "Sample error", extra={"debug_info": "additional context"})
        return "result"

    # Test async logging
    @logger.async_log
    def async_log_test():
        logger.log("DEBUG", "Async debug message")

    # Test model summary logging
    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 2)
        
        def forward(self, x):
            return self.linear(x)

    logger.log_model_summary(TestModel(), (10,))

    try:
        sample_function()
        async_log_test()
    finally:
        logger.shutdown()
