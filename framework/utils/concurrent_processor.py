"""
Concurrent processing utilities for the PyTorch inference framework.

This module provides utilities for parallel and concurrent processing of
multiple tasks, batches, and model operations.
"""

import asyncio
import concurrent.futures
import threading
import time
import logging
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import queue
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import weakref
import gc

logger = logging.getLogger(__name__)


@dataclass
class ProcessingTask:
    """Represents a single processing task."""
    
    task_id: str
    function: Callable
    args: Tuple = ()
    kwargs: Dict[str, Any] = None
    priority: int = 0
    timeout: Optional[float] = None
    
    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}


class TaskResult:
    """Result of a processing task."""
    
    def __init__(self, task_id: str, result: Any = None, error: Optional[Exception] = None,
                 start_time: float = 0.0, end_time: float = 0.0):
        self.task_id = task_id
        self.result = result
        self.error = error
        self.start_time = start_time
        self.end_time = end_time
        self.processing_time = end_time - start_time if end_time > start_time else 0.0
    
    @property
    def success(self) -> bool:
        """Whether the task completed successfully."""
        return self.error is None
    
    def get_result(self):
        """Get the result or raise the error if task failed."""
        if self.error:
            raise self.error
        return self.result


class ConcurrentProcessor:
    """
    Concurrent processor for handling multiple tasks in parallel.
    
    Supports both thread-based and process-based parallelism with configurable
    worker pools and task queues.
    """
    
    def __init__(self, max_workers: Optional[int] = None, 
                 use_processes: bool = False,
                 task_timeout: Optional[float] = None,
                 queue_size: int = 1000):
        """
        Initialize concurrent processor.
        
        Args:
            max_workers: Maximum number of worker threads/processes
            use_processes: Whether to use processes instead of threads
            task_timeout: Default timeout for tasks
            queue_size: Maximum size of task queue
        """
        self.max_workers = max_workers or min(32, (mp.cpu_count() or 1) + 4)
        self.use_processes = use_processes
        self.task_timeout = task_timeout
        self.queue_size = queue_size
        
        self._executor: Optional[Union[ThreadPoolExecutor, ProcessPoolExecutor]] = None
        self._task_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._results: Dict[str, TaskResult] = {}
        self._active_tasks: Dict[str, concurrent.futures.Future] = {}
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        self.logger = logging.getLogger(f"{__name__}.ConcurrentProcessor")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
    
    def start(self):
        """Start the concurrent processor."""
        with self._lock:
            if self._running:
                return
            
            if self.use_processes:
                self._executor = ProcessPoolExecutor(max_workers=self.max_workers)
            else:
                self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
            
            self._running = True
            self._worker_thread = threading.Thread(target=self._process_tasks, daemon=True)
            self._worker_thread.start()
            
            self.logger.info(f"Started concurrent processor with {self.max_workers} workers "
                           f"({'processes' if self.use_processes else 'threads'})")
    
    def shutdown(self, wait: bool = True, timeout: Optional[float] = None):
        """
        Shutdown the concurrent processor.
        
        Args:
            wait: Whether to wait for pending tasks to complete
            timeout: Maximum time to wait for shutdown
        """
        with self._lock:
            if not self._running:
                return
            
            self._running = False
            
            # Stop accepting new tasks
            try:
                self._task_queue.put(None, timeout=1.0)  # Sentinel to stop worker thread
            except queue.Full:
                pass
            
            # Wait for worker thread to finish
            if self._worker_thread and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=timeout or 10.0)
            
            # Shutdown executor
            if self._executor:
                self._executor.shutdown(wait=wait, timeout=timeout)
                self._executor = None
            
            # Cancel remaining active tasks
            for future in self._active_tasks.values():
                future.cancel()
            
            self._active_tasks.clear()
            
            self.logger.info("Concurrent processor shutdown complete")
    
    def submit_task(self, task: ProcessingTask) -> bool:
        """
        Submit a task for processing.
        
        Args:
            task: Task to process
            
        Returns:
            True if task was submitted, False if queue is full
        """
        if not self._running:
            raise RuntimeError("Processor is not running")
        
        try:
            self._task_queue.put(task, block=False)
            self.logger.debug(f"Submitted task {task.task_id}")
            return True
        except queue.Full:
            self.logger.warning(f"Task queue full, cannot submit task {task.task_id}")
            return False
    
    def submit_function(self, task_id: str, function: Callable, *args, 
                       priority: int = 0, timeout: Optional[float] = None, **kwargs) -> bool:
        """
        Submit a function for processing.
        
        Args:
            task_id: Unique identifier for the task
            function: Function to execute
            *args: Positional arguments for function
            priority: Task priority (higher = more important)
            timeout: Task timeout
            **kwargs: Keyword arguments for function
            
        Returns:
            True if task was submitted, False if queue is full
        """
        task = ProcessingTask(
            task_id=task_id,
            function=function,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout or self.task_timeout
        )
        return self.submit_task(task)
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> TaskResult:
        """
        Get the result of a completed task.
        
        Args:
            task_id: ID of the task
            timeout: Maximum time to wait for result
            
        Returns:
            TaskResult containing the result or error
        """
        start_time = time.time()
        
        while timeout is None or (time.time() - start_time) < timeout:
            with self._lock:
                # Check if result is available
                if task_id in self._results:
                    return self._results.pop(task_id)
                
                # Check if task is still running
                if task_id in self._active_tasks:
                    time.sleep(0.01)  # Brief wait before checking again
                    continue
                
                # Task not found
                raise KeyError(f"Task {task_id} not found")
        
        raise TimeoutError(f"Timeout waiting for result of task {task_id}")
    
    def wait_for_completion(self, task_ids: List[str], 
                          timeout: Optional[float] = None) -> Dict[str, TaskResult]:
        """
        Wait for multiple tasks to complete.
        
        Args:
            task_ids: List of task IDs to wait for
            timeout: Maximum time to wait
            
        Returns:
            Dictionary mapping task IDs to results
        """
        results = {}
        start_time = time.time()
        
        remaining_tasks = set(task_ids)
        
        while remaining_tasks and (timeout is None or (time.time() - start_time) < timeout):
            completed_tasks = set()
            
            for task_id in remaining_tasks:
                try:
                    result = self.get_result(task_id, timeout=0.1)
                    results[task_id] = result
                    completed_tasks.add(task_id)
                except (KeyError, TimeoutError):
                    continue
            
            remaining_tasks -= completed_tasks
            
            if remaining_tasks:
                time.sleep(0.01)
        
        # Handle any remaining tasks that timed out
        for task_id in remaining_tasks:
            results[task_id] = TaskResult(
                task_id=task_id,
                error=TimeoutError(f"Task {task_id} timed out")
            )
        
        return results
    
    def get_queue_size(self) -> int:
        """Get current size of task queue."""
        return self._task_queue.qsize()
    
    def get_active_task_count(self) -> int:
        """Get number of currently active tasks."""
        with self._lock:
            return len(self._active_tasks)
    
    def is_running(self) -> bool:
        """Check if processor is running."""
        return self._running
    
    def _process_tasks(self):
        """Worker thread that processes tasks from the queue."""
        self.logger.debug("Task processor thread started")
        
        while self._running:
            try:
                # Get next task from queue
                task = self._task_queue.get(timeout=1.0)
                
                # Check for shutdown sentinel
                if task is None:
                    break
                
                # Submit task to executor
                self._submit_to_executor(task)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in task processor: {e}")
        
        self.logger.debug("Task processor thread stopped")
    
    def _submit_to_executor(self, task: ProcessingTask):
        """Submit a task to the executor."""
        if not self._executor:
            return
        
        try:
            start_time = time.time()
            
            # Submit to executor
            future = self._executor.submit(self._execute_task, task, start_time)
            
            # Store future
            with self._lock:
                self._active_tasks[task.task_id] = future
            
            # Add callback to handle completion
            future.add_done_callback(lambda f: self._handle_task_completion(task.task_id, f))
            
        except Exception as e:
            # Create error result
            result = TaskResult(
                task_id=task.task_id,
                error=e,
                start_time=time.time(),
                end_time=time.time()
            )
            
            with self._lock:
                self._results[task.task_id] = result
    
    def _execute_task(self, task: ProcessingTask, start_time: float) -> Any:
        """Execute a single task."""
        try:
            # Set timeout if specified
            if task.timeout:
                # Note: This is a simple timeout implementation
                # More sophisticated timeout handling could be added
                pass
            
            # Execute the function
            result = task.function(*task.args, **task.kwargs)
            return result
            
        except Exception as e:
            self.logger.error(f"Task {task.task_id} failed: {e}")
            raise
    
    def _handle_task_completion(self, task_id: str, future: concurrent.futures.Future):
        """Handle completion of a task."""
        end_time = time.time()
        
        try:
            # Get result or exception
            if future.exception():
                result = TaskResult(
                    task_id=task_id,
                    error=future.exception(),
                    end_time=end_time
                )
            else:
                result = TaskResult(
                    task_id=task_id,
                    result=future.result(),
                    end_time=end_time
                )
            
            # Store result
            with self._lock:
                self._results[task_id] = result
                self._active_tasks.pop(task_id, None)
            
            self.logger.debug(f"Task {task_id} completed in {result.processing_time:.3f}s")
            
        except Exception as e:
            self.logger.error(f"Error handling completion of task {task_id}: {e}")
            
            # Create error result
            result = TaskResult(
                task_id=task_id,
                error=e,
                end_time=end_time
            )
            
            with self._lock:
                self._results[task_id] = result
                self._active_tasks.pop(task_id, None)


class BatchProcessor:
    """
    Specialized processor for handling batched operations.
    """
    
    def __init__(self, batch_size: int = 32, max_wait_time: float = 1.0,
                 max_workers: Optional[int] = None):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Size of batches to process
            max_wait_time: Maximum time to wait before processing partial batch
            max_workers: Maximum number of worker threads
        """
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.processor = ConcurrentProcessor(max_workers=max_workers)
        
        self._pending_items: List[Tuple[str, Any]] = []
        self._batch_lock = threading.Lock()
        self._last_batch_time = time.time()
        
        self.logger = logging.getLogger(f"{__name__}.BatchProcessor")
    
    def __enter__(self):
        """Context manager entry."""
        self.processor.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.processor.shutdown()
    
    def submit_item(self, item_id: str, item: Any, 
                   batch_function: Callable[[List[Any]], List[Any]]) -> bool:
        """
        Submit an item for batch processing.
        
        Args:
            item_id: Unique identifier for the item
            item: Item to process
            batch_function: Function that processes a batch of items
            
        Returns:
            True if item was submitted successfully
        """
        with self._batch_lock:
            self._pending_items.append((item_id, item))
            
            # Check if we should process a batch
            should_process = (
                len(self._pending_items) >= self.batch_size or
                (time.time() - self._last_batch_time) > self.max_wait_time
            )
            
            if should_process:
                self._process_batch(batch_function)
                
        return True
    
    def _process_batch(self, batch_function: Callable[[List[Any]], List[Any]]):
        """Process the current batch of pending items."""
        if not self._pending_items:
            return
        
        # Extract items and IDs
        batch_items = [item for _, item in self._pending_items]
        batch_ids = [item_id for item_id, _ in self._pending_items]
        
        # Clear pending items
        self._pending_items.clear()
        self._last_batch_time = time.time()
        
        # Submit batch for processing
        batch_task_id = f"batch_{int(time.time() * 1000)}"
        
        def process_batch_wrapper():
            try:
                results = batch_function(batch_items)
                
                # Submit individual results
                for item_id, result in zip(batch_ids, results):
                    # Store result for the individual item
                    task_result = TaskResult(
                        task_id=item_id,
                        result=result,
                        start_time=time.time(),
                        end_time=time.time()
                    )
                    
                    with self.processor._lock:
                        self.processor._results[item_id] = task_result
                
                return results
                
            except Exception as e:
                # Handle batch error - create error results for all items
                for item_id in batch_ids:
                    task_result = TaskResult(
                        task_id=item_id,
                        error=e,
                        start_time=time.time(),
                        end_time=time.time()
                    )
                    
                    with self.processor._lock:
                        self.processor._results[item_id] = task_result
                
                raise
        
        self.processor.submit_function(
            batch_task_id,
            process_batch_wrapper
        )
    
    def get_result(self, item_id: str, timeout: Optional[float] = None) -> TaskResult:
        """Get result for a specific item."""
        return self.processor.get_result(item_id, timeout)


# Utility functions for common concurrent processing patterns

def parallel_map(function: Callable, items: List[Any], 
                max_workers: Optional[int] = None,
                use_processes: bool = False,
                timeout: Optional[float] = None) -> List[Any]:
    """
    Apply a function to a list of items in parallel.
    
    Args:
        function: Function to apply to each item
        items: List of items to process
        max_workers: Maximum number of workers
        use_processes: Whether to use processes instead of threads
        timeout: Timeout for the entire operation
        
    Returns:
        List of results in the same order as input items
    """
    if not items:
        return []
    
    with ConcurrentProcessor(max_workers=max_workers, use_processes=use_processes) as processor:
        # Submit all tasks
        task_ids = []
        for i, item in enumerate(items):
            task_id = f"map_task_{i}"
            task_ids.append(task_id)
            processor.submit_function(task_id, function, item, timeout=timeout)
        
        # Wait for results
        results = processor.wait_for_completion(task_ids, timeout=timeout)
        
        # Extract results in order
        ordered_results = []
        for task_id in task_ids:
            result = results[task_id]
            if result.success:
                ordered_results.append(result.result)
            else:
                raise result.error
        
        return ordered_results


async def async_parallel_map(function: Callable, items: List[Any],
                           max_workers: Optional[int] = None) -> List[Any]:
    """
    Async version of parallel_map using asyncio.
    
    Args:
        function: Function to apply to each item
        items: List of items to process
        max_workers: Maximum number of workers
        
    Returns:
        List of results in the same order as input items
    """
    if not items:
        return []
    
    loop = asyncio.get_event_loop()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [loop.run_in_executor(executor, function, item) for item in items]
        
        # Wait for completion
        results = await asyncio.gather(*futures)
        
        return results