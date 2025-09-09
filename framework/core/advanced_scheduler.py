"""
Advanced scheduler for multi-GPU workload management.
Provides intelligent task scheduling and resource allocation.
"""

import time
import heapq
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from queue import PriorityQueue, Queue, Empty
import torch

logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4

class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Task:
    """Task definition for the scheduler."""
    id: str
    priority: TaskPriority
    func: Callable
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    device_requirement: Optional[int] = None  # Specific device requirement
    memory_requirement: int = 0  # Memory requirement in bytes
    estimated_duration: float = 1.0  # Estimated duration in seconds
    dependencies: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    deadline: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    callback: Optional[Callable] = None
    
    def __lt__(self, other):
        """For priority queue ordering."""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        if self.deadline and other.deadline:
            return self.deadline < other.deadline
        return self.created_at < other.created_at

@dataclass
class DeviceState:
    """State of a GPU device."""
    device_id: int
    is_available: bool = True
    current_tasks: List[str] = field(default_factory=list)
    queue_length: int = 0
    memory_used: int = 0
    memory_total: int = 0
    utilization: float = 0.0
    last_update: float = field(default_factory=time.time)
    
    @property
    def memory_available(self) -> int:
        return self.memory_total - self.memory_used
    
    @property
    def load_score(self) -> float:
        """Calculate load score for scheduling decisions."""
        memory_ratio = self.memory_used / max(self.memory_total, 1)
        queue_weight = min(self.queue_length / 10.0, 1.0)  # Normalize queue length
        return (memory_ratio * 0.4 + self.utilization * 0.4 + queue_weight * 0.2)

class SchedulingStrategy(Enum):
    """Scheduling strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    MEMORY_AWARE = "memory_aware"
    DEADLINE_FIRST = "deadline_first"
    PRIORITY_FIRST = "priority_first"
    BALANCED = "balanced"

@dataclass
class SchedulerConfig:
    """Configuration for the advanced scheduler."""
    strategy: SchedulingStrategy = SchedulingStrategy.BALANCED
    max_tasks_per_device: int = 4
    task_timeout: float = 300.0  # 5 minutes default timeout
    heartbeat_interval: float = 1.0
    cleanup_interval: float = 60.0
    enable_preemption: bool = False
    enable_migration: bool = False
    load_balancing_threshold: float = 0.3

class AdvancedScheduler:
    """Advanced scheduler for multi-GPU workload management."""
    
    def __init__(self, devices: List[int], config: SchedulerConfig):
        self.devices = devices
        self.config = config
        
        # Task management
        self.pending_tasks: PriorityQueue = PriorityQueue()
        self.all_tasks: Dict[str, Task] = {}
        self.task_results: Dict[str, Any] = {}
        self.task_status: Dict[str, TaskStatus] = {}
        self.task_futures: Dict[str, Future] = {}
        
        # Device management
        self.device_states: Dict[int, DeviceState] = {}
        self.device_queues: Dict[int, Queue] = {}
        self.device_workers: Dict[int, threading.Thread] = {}
        
        # Dependency tracking
        self.dependency_graph: Dict[str, List[str]] = {}  # task_id -> dependents
        self.reverse_dependencies: Dict[str, List[str]] = {}  # task_id -> dependencies
        
        # Threading
        self.scheduler_active = False
        self.scheduler_thread: Optional[threading.Thread] = None
        self.heartbeat_thread: Optional[threading.Thread] = None
        self.cleanup_thread: Optional[threading.Thread] = None
        
        # Thread pool for task execution
        self.executor = ThreadPoolExecutor(max_workers=len(devices) * self.config.max_tasks_per_device)
        
        # Statistics
        self.stats = {
            'tasks_scheduled': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'average_wait_time': 0.0,
            'average_execution_time': 0.0
        }
        
        # Locks
        self.scheduler_lock = threading.Lock()
        self.stats_lock = threading.Lock()
        
        self._initialize_devices()
    
    def _initialize_devices(self):
        """Initialize device states and workers."""
        for device_id in self.devices:
            # Initialize device state
            try:
                with torch.cuda.device(device_id):
                    props = torch.cuda.get_device_properties(device_id)
                    total_memory = props.total_memory
            except:
                total_memory = 0
            
            self.device_states[device_id] = DeviceState(
                device_id=device_id,
                memory_total=total_memory
            )
            
            # Create device queue and worker
            self.device_queues[device_id] = Queue()
            worker = threading.Thread(
                target=self._device_worker,
                args=(device_id,),
                daemon=True
            )
            self.device_workers[device_id] = worker
            
        logger.info(f"Initialized scheduler for {len(self.devices)} devices")
    
    def _device_worker(self, device_id: int):
        """Worker thread for a specific device."""
        while self.scheduler_active:
            try:
                # Get task from device queue
                task_id = self.device_queues[device_id].get(timeout=1.0)
                
                if task_id not in self.all_tasks:
                    continue
                
                task = self.all_tasks[task_id]
                
                # Update task status
                self.task_status[task_id] = TaskStatus.RUNNING
                
                # Update device state
                device_state = self.device_states[device_id]
                device_state.current_tasks.append(task_id)
                device_state.queue_length = self.device_queues[device_id].qsize()
                
                # Execute task
                start_time = time.time()
                try:
                    result = self._execute_task(task, device_id)
                    self.task_results[task_id] = result
                    self.task_status[task_id] = TaskStatus.COMPLETED
                    
                    # Update statistics
                    execution_time = time.time() - start_time
                    self._update_stats(task, execution_time, success=True)
                    
                    # Execute callback if provided
                    if task.callback:
                        task.callback(result)
                    
                    # Process dependencies
                    self._process_dependencies(task_id)
                    
                except Exception as e:
                    logger.error(f"Task {task_id} failed on device {device_id}: {e}")
                    self.task_status[task_id] = TaskStatus.FAILED
                    self._handle_task_failure(task, str(e))
                    self._update_stats(task, time.time() - start_time, success=False)
                
                finally:
                    # Clean up device state
                    if task_id in device_state.current_tasks:
                        device_state.current_tasks.remove(task_id)
                    device_state.queue_length = self.device_queues[device_id].qsize()
                    
                    # Mark queue task as done
                    self.device_queues[device_id].task_done()
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Device worker {device_id} error: {e}")
    
    def _execute_task(self, task: Task, device_id: int) -> Any:
        """Execute a task on a specific device."""
        # Set CUDA device if needed
        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)
        
        # Add device_id to kwargs for the task function
        kwargs = task.kwargs.copy()
        kwargs['device_id'] = device_id
        
        # Execute the task
        return task.func(*task.args, **kwargs)
    
    def _handle_task_failure(self, task: Task, error_msg: str):
        """Handle task failure with retry logic."""
        task.retry_count += 1
        
        if task.retry_count <= task.max_retries:
            logger.info(f"Retrying task {task.id} (attempt {task.retry_count}/{task.max_retries})")
            # Reschedule task
            self.task_status[task.id] = TaskStatus.PENDING
            self.pending_tasks.put(task)
        else:
            logger.error(f"Task {task.id} failed permanently after {task.max_retries} retries: {error_msg}")
            self.task_results[task.id] = Exception(error_msg)
    
    def _process_dependencies(self, completed_task_id: str):
        """Process dependencies when a task completes."""
        if completed_task_id not in self.dependency_graph:
            return
        
        # Check each dependent task
        for dependent_id in self.dependency_graph[completed_task_id]:
            if dependent_id not in self.reverse_dependencies:
                continue
            
            # Remove completed dependency
            deps = self.reverse_dependencies[dependent_id]
            if completed_task_id in deps:
                deps.remove(completed_task_id)
            
            # If all dependencies are satisfied, make task eligible for scheduling
            if not deps and dependent_id in self.all_tasks:
                task = self.all_tasks[dependent_id]
                if self.task_status[dependent_id] == TaskStatus.PENDING:
                    self.pending_tasks.put(task)
    
    def _update_stats(self, task: Task, execution_time: float, success: bool):
        """Update scheduler statistics."""
        with self.stats_lock:
            if success:
                self.stats['tasks_completed'] += 1
            else:
                self.stats['tasks_failed'] += 1
            
            # Update average execution time
            completed = self.stats['tasks_completed']
            if completed > 0:
                current_avg = self.stats['average_execution_time']
                self.stats['average_execution_time'] = (
                    (current_avg * (completed - 1) + execution_time) / completed
                )
            
            # Update wait time
            wait_time = time.time() - task.created_at - execution_time
            if completed > 0:
                current_avg = self.stats['average_wait_time']
                self.stats['average_wait_time'] = (
                    (current_avg * (completed - 1) + wait_time) / completed
                )
    
    def _select_device(self, task: Task) -> int:
        """Select optimal device for task execution."""
        if task.device_requirement is not None:
            # Task requires specific device
            if task.device_requirement in self.devices:
                return task.device_requirement
            else:
                raise ValueError(f"Required device {task.device_requirement} not available")
        
        available_devices = [
            device_id for device_id in self.devices
            if len(self.device_states[device_id].current_tasks) < self.config.max_tasks_per_device
        ]
        
        if not available_devices:
            # All devices are at capacity, select least loaded
            available_devices = self.devices
        
        if self.config.strategy == SchedulingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(available_devices)
        elif self.config.strategy == SchedulingStrategy.LEAST_LOADED:
            return self._least_loaded_selection(available_devices)
        elif self.config.strategy == SchedulingStrategy.MEMORY_AWARE:
            return self._memory_aware_selection(available_devices, task)
        elif self.config.strategy == SchedulingStrategy.BALANCED:
            return self._balanced_selection(available_devices, task)
        else:
            return available_devices[0]
    
    def _round_robin_selection(self, available_devices: List[int]) -> int:
        """Round-robin device selection."""
        if not hasattr(self, '_round_robin_index'):
            self._round_robin_index = 0
        
        device = available_devices[self._round_robin_index % len(available_devices)]
        self._round_robin_index += 1
        return device
    
    def _least_loaded_selection(self, available_devices: List[int]) -> int:
        """Select device with lowest load."""
        return min(available_devices, key=lambda d: self.device_states[d].load_score)
    
    def _memory_aware_selection(self, available_devices: List[int], task: Task) -> int:
        """Select device based on memory requirements."""
        suitable_devices = [
            device_id for device_id in available_devices
            if self.device_states[device_id].memory_available >= task.memory_requirement
        ]
        
        if suitable_devices:
            return min(suitable_devices, key=lambda d: self.device_states[d].load_score)
        else:
            # Fallback to least loaded if no device has enough memory
            return self._least_loaded_selection(available_devices)
    
    def _balanced_selection(self, available_devices: List[int], task: Task) -> int:
        """Balanced selection considering multiple factors."""
        scores = {}
        
        for device_id in available_devices:
            state = self.device_states[device_id]
            
            # Memory score (0-1, lower is better)
            memory_score = state.memory_used / max(state.memory_total, 1)
            
            # Queue score (0-1, lower is better)
            queue_score = min(state.queue_length / 10.0, 1.0)
            
            # Utilization score (0-1, lower is better)
            util_score = state.utilization
            
            # Task requirement compatibility
            memory_fit = 1.0 if state.memory_available >= task.memory_requirement else 0.5
            
            # Combined score
            scores[device_id] = (memory_score * 0.3 + queue_score * 0.3 + 
                               util_score * 0.3 + (1 - memory_fit) * 0.1)
        
        return min(scores.items(), key=lambda x: x[1])[0]
    
    def schedule_task(self, func: Callable, args: Tuple = (), kwargs: Dict = None,
                     priority: TaskPriority = TaskPriority.NORMAL,
                     device_requirement: Optional[int] = None,
                     memory_requirement: int = 0,
                     estimated_duration: float = 1.0,
                     dependencies: List[str] = None,
                     deadline: Optional[float] = None,
                     callback: Optional[Callable] = None,
                     task_id: Optional[str] = None) -> str:
        """Schedule a task for execution."""
        
        if kwargs is None:
            kwargs = {}
        if dependencies is None:
            dependencies = []
        
        # Generate task ID
        if task_id is None:
            task_id = str(uuid.uuid4())
        
        # Create task
        task = Task(
            id=task_id,
            priority=priority,
            func=func,
            args=args,
            kwargs=kwargs,
            device_requirement=device_requirement,
            memory_requirement=memory_requirement,
            estimated_duration=estimated_duration,
            dependencies=dependencies.copy(),
            deadline=deadline,
            callback=callback
        )
        
        with self.scheduler_lock:
            # Store task
            self.all_tasks[task_id] = task
            self.task_status[task_id] = TaskStatus.PENDING
            
            # Set up dependencies
            if dependencies:
                self.reverse_dependencies[task_id] = dependencies.copy()
                for dep_id in dependencies:
                    if dep_id not in self.dependency_graph:
                        self.dependency_graph[dep_id] = []
                    self.dependency_graph[dep_id].append(task_id)
            
            # Add to pending queue if no dependencies or all are satisfied
            if not dependencies or all(
                self.task_status.get(dep_id) == TaskStatus.COMPLETED 
                for dep_id in dependencies
            ):
                self.pending_tasks.put(task)
            
            self.stats['tasks_scheduled'] += 1
        
        logger.debug(f"Scheduled task {task_id} with priority {priority.name}")
        return task_id
    
    def start(self):
        """Start the scheduler."""
        if self.scheduler_active:
            return
        
        self.scheduler_active = True
        
        # Start device workers
        for worker in self.device_workers.values():
            worker.start()
        
        # Start scheduler thread
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        # Start heartbeat thread
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        
        logger.info("Advanced scheduler started")
    
    def stop(self):
        """Stop the scheduler."""
        self.scheduler_active = False
        
        # Wait for threads to finish
        for thread in [self.scheduler_thread, self.heartbeat_thread, self.cleanup_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=1.0)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Advanced scheduler stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.scheduler_active:
            try:
                # Get next task
                task = self.pending_tasks.get(timeout=1.0)
                
                # Check if task is still valid
                if task.id not in self.all_tasks or self.task_status[task.id] != TaskStatus.PENDING:
                    continue
                
                # Check deadline
                if task.deadline and time.time() > task.deadline:
                    logger.warning(f"Task {task.id} missed deadline")
                    self.task_status[task.id] = TaskStatus.FAILED
                    continue
                
                # Select device
                try:
                    device_id = self._select_device(task)
                    
                    # Assign task to device
                    self.task_status[task.id] = TaskStatus.ASSIGNED
                    self.device_queues[device_id].put(task.id)
                    
                    logger.debug(f"Assigned task {task.id} to device {device_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to assign task {task.id}: {e}")
                    self.task_status[task.id] = TaskStatus.FAILED
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
    
    def _heartbeat_loop(self):
        """Heartbeat loop for monitoring."""
        while self.scheduler_active:
            try:
                # Update device states
                for device_id in self.devices:
                    self._update_device_state(device_id)
                
                time.sleep(self.config.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")
                time.sleep(self.config.heartbeat_interval)
    
    def _cleanup_loop(self):
        """Cleanup loop for old tasks and results."""
        while self.scheduler_active:
            try:
                current_time = time.time()
                cleanup_threshold = current_time - self.config.cleanup_interval * 10
                
                # Clean up old completed tasks
                tasks_to_remove = []
                for task_id, task in self.all_tasks.items():
                    if (self.task_status[task_id] in [TaskStatus.COMPLETED, TaskStatus.FAILED] and
                        task.created_at < cleanup_threshold):
                        tasks_to_remove.append(task_id)
                
                for task_id in tasks_to_remove:
                    self._cleanup_task(task_id)
                
                time.sleep(self.config.cleanup_interval)
                
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                time.sleep(self.config.cleanup_interval)
    
    def _update_device_state(self, device_id: int):
        """Update device state with current metrics."""
        try:
            if torch.cuda.is_available() and device_id < torch.cuda.device_count():
                with torch.cuda.device(device_id):
                    memory_used = torch.cuda.memory_allocated(device_id)
                    # Approximate utilization based on memory and task count
                    task_count = len(self.device_states[device_id].current_tasks)
                    utilization = min(task_count / self.config.max_tasks_per_device, 1.0)
                    
                    self.device_states[device_id].memory_used = memory_used
                    self.device_states[device_id].utilization = utilization
                    self.device_states[device_id].last_update = time.time()
        except Exception as e:
            logger.warning(f"Failed to update device state for GPU {device_id}: {e}")
    
    def _cleanup_task(self, task_id: str):
        """Clean up a completed task."""
        with self.scheduler_lock:
            self.all_tasks.pop(task_id, None)
            self.task_status.pop(task_id, None)
            self.task_results.pop(task_id, None)
            self.task_futures.pop(task_id, None)
            self.reverse_dependencies.pop(task_id, None)
            
            # Clean up from dependency graph
            if task_id in self.dependency_graph:
                for dependent in self.dependency_graph[task_id]:
                    if dependent in self.reverse_dependencies:
                        deps = self.reverse_dependencies[dependent]
                        if task_id in deps:
                            deps.remove(task_id)
                del self.dependency_graph[task_id]
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get status of a task."""
        return self.task_status.get(task_id)
    
    def get_task_result(self, task_id: str) -> Any:
        """Get result of a completed task."""
        return self.task_results.get(task_id)
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or assigned task."""
        with self.scheduler_lock:
            if task_id in self.task_status:
                status = self.task_status[task_id]
                if status in [TaskStatus.PENDING, TaskStatus.ASSIGNED]:
                    self.task_status[task_id] = TaskStatus.CANCELLED
                    return True
        return False
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        with self.stats_lock:
            device_stats = {}
            for device_id, state in self.device_states.items():
                device_stats[device_id] = {
                    'current_tasks': len(state.current_tasks),
                    'queue_length': state.queue_length,
                    'memory_used_mb': state.memory_used / (1024 * 1024),
                    'memory_total_mb': state.memory_total / (1024 * 1024),
                    'utilization': state.utilization,
                    'load_score': state.load_score
                }
            
            return {
                'global_stats': self.stats.copy(),
                'device_stats': device_stats,
                'pending_tasks': self.pending_tasks.qsize(),
                'total_tasks': len(self.all_tasks),
                'active_devices': len(self.devices)
            }
    
    def cleanup(self):
        """Clean up scheduler resources."""
        self.stop()
        
        # Clear all data structures
        self.all_tasks.clear()
        self.task_status.clear()
        self.task_results.clear()
        self.dependency_graph.clear()
        self.reverse_dependencies.clear()
        
        logger.info("Advanced scheduler cleanup completed")
