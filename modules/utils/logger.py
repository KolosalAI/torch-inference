import logging
import logging.config
import sys
import traceback
from logging.handlers import QueueHandler, QueueListener
from queue import Queue
from typing import Optional, Callable, Any, Tuple
import inspect
from threading import Thread
from datetime import datetime
import os
import torch
from functools import wraps

# Local imports - ensure that .config.LOGGING_CONFIG is available
from .config import LOGGING_CONFIG


class StructuredLogger:
    def __init__(self, name: str) -> None:
        """
        Initialize the StructuredLogger with a given logger name.
        """
        self.logger = logging.getLogger(name)
        self._async_queue: Optional[Queue] = None
        self._async_listener: Optional[QueueListener] = None

    @classmethod
    def configure_logging(cls) -> None:
        """
        Configure logging using the centralized configuration.
        """
        try:
            logging.config.dictConfig(LOGGING_CONFIG)
            logging.captureWarnings(True)
        except Exception as e:
            print(f"Failed to configure logging: {str(e)}")
            # Fallback configuration if LOGGING_CONFIG fails
            logging.basicConfig(level=logging.INFO)

    @classmethod
    def get_logger(cls, name: Optional[str] = None) -> "StructuredLogger":
        """
        Retrieve a StructuredLogger instance. If no name is provided,
        the caller's function name will be used.
        """
        if not name:
            # Grab the calling function's name
            name = inspect.stack()[1][3]
        return cls(name)

    def _ensure_async(self) -> None:
        """
        Initialize asynchronous logging infrastructure if not already done.
        """
        if self._async_queue is None:
            self._async_queue = Queue(-1)  # Unlimited size
            # Copy current handlers to pass to the listener
            handlers = self.logger.handlers.copy()
            self._async_listener = QueueListener(self._async_queue, *handlers)
            self._async_listener.start()

            # Replace existing handlers with a QueueHandler
            for handler in handlers:
                self.logger.removeHandler(handler)
            self.logger.addHandler(QueueHandler(self._async_queue))

    def async_log(self, func: Callable) -> Callable:
        """
        Decorator that ensures asynchronous logging is enabled before
        executing the decorated function.
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            self._ensure_async()
            return func(*args, **kwargs)
        return wrapper

    def log_execution(self, func: Callable) -> Callable:
        """
        Decorator that logs the entry, exit, and execution time of the decorated function.
        """
        @wraps(func)
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
                    extra={"error": str(e), "traceback": traceback.format_exc()},
                    exc_info=True
                )
                raise
        return wrapper

    def log(self, level: str, message: str, **kwargs: Any) -> None:
        """
        Log a message at the specified level with additional context.
        Automatically appends code context information (file, line, function).
        """
        log_level = getattr(logging, level.upper(), logging.INFO)
        extra = kwargs.get('extra', {})

        # Add code context information from the caller two frames up
        frame = inspect.currentframe()
        if frame is not None:
            caller_frame = frame.f_back.f_back
            if caller_frame is not None:
                extra.update({
                    "file": caller_frame.f_code.co_filename,
                    "line": caller_frame.f_lineno,
                    "function": caller_frame.f_code.co_name
                })

        exc_info = kwargs.get('exc_info', False)
        self.logger.log(log_level, message, extra=extra, exc_info=exc_info)

    def log_model_summary(self, model: torch.nn.Module, input_shape: Tuple[int, ...]) -> None:
        """
        Log a summary of the PyTorch model architecture using torchsummary.
        """
        try:
            from torchsummary import summary  # Local import to avoid dependency issues
            summary_lines = []

            def capture_summary(line: str) -> None:
                summary_lines.append(line)

            summary(model, input_shape, print_fn=capture_summary)
            self.log("INFO", "Model Architecture", extra={"summary": "\n".join(summary_lines)})
        except Exception as e:
            self.log("ERROR", "Failed to log model summary", extra={"error": str(e)}, exc_info=True)

    def shutdown(self) -> None:
        """
        Clean up and shut down asynchronous logging infrastructure.
        """
        if self._async_listener:
            self._async_listener.stop()

    # Convenience methods for different logging levels
    def debug(self, message: str, **kwargs: Any) -> None:
        self.log("DEBUG", message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        self.log("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        self.log("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        self.log("ERROR", message, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        self.log("CRITICAL", message, **kwargs)

    def exception(self, message: str, **kwargs: Any) -> None:
        """
        Log an exception message along with the stack trace.
        """
        self.log("ERROR", message, exc_info=True, **kwargs)
        
    def setLevel(self, level: int) -> None:
        """
        Set the logging level for the underlying logger.
        """
        self.logger.setLevel(level)

# Initialize default logger configuration at module import
StructuredLogger.configure_logging()


def get_logger(name: Optional[str] = None) -> StructuredLogger:
    """
    Get a configured StructuredLogger instance.
    """
    return StructuredLogger.get_logger(name)


if __name__ == "__main__":
    # Example usage of the StructuredLogger

    logger = get_logger(__name__)

    @logger.log_execution
    def sample_function() -> str:
        logger.info("Sample info message")
        logger.error("Sample error", extra={"debug_info": "additional context"})
        return "result"

    @logger.async_log
    def async_log_test() -> None:
        logger.debug("Async debug message")

    # Define a simple PyTorch model for demonstration purposes
    class TestModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(10, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x)

    # Log the model summary (input_shape provided as tuple of ints)
    logger.log_model_summary(TestModel(), (10,))

    try:
        sample_function()
        async_log_test()
    finally:
        logger.shutdown()
