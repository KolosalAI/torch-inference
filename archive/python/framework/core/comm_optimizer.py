"""
Communication optimizer for multi-GPU setups.
Handles efficient data transfer, synchronization, and communication patterns.
"""

import os
import torch
import torch.distributed as dist
import threading
import time
import queue
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, Future

logger = logging.getLogger(__name__)

class CommPattern(Enum):
    """Communication patterns for multi-GPU operations."""
    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    BROADCAST = "broadcast"
    REDUCE_SCATTER = "reduce_scatter"
    POINT_TO_POINT = "point_to_point"

@dataclass
class CommOp:
    """Communication operation descriptor."""
    pattern: CommPattern
    tensor: torch.Tensor
    src_device: int
    dst_devices: List[int]
    callback: Optional[Callable] = None
    priority: int = 0

@dataclass
class CommStats:
    """Communication statistics."""
    total_ops: int = 0
    total_bytes: int = 0
    avg_latency: float = 0.0
    bandwidth_utilization: float = 0.0
    error_count: int = 0

class CommunicationOptimizer:
    """Advanced communication optimizer for multi-GPU inference."""
    
    def __init__(self, devices: List[int], enable_nccl: bool = True):
        self.devices = devices
        self.enable_nccl = enable_nccl
        self.device_map = {i: devices[i] for i in range(len(devices))}
        self.reverse_map = {dev: i for i, dev in enumerate(devices)}
        
        # Communication queues
        self.comm_queues: Dict[int, queue.PriorityQueue] = {}
        self.comm_workers: Dict[int, threading.Thread] = {}
        self.active = False
        
        # Statistics
        self.stats = CommStats()
        self.device_stats: Dict[int, CommStats] = {}
        
        # Optimization parameters
        self.chunk_size = 4 * 1024 * 1024  # 4MB chunks
        self.overlap_threshold = 1024 * 1024  # 1MB threshold for overlap
        self.bandwidth_limit = 0.8  # Use 80% of available bandwidth
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=len(devices))
        self.lock = threading.Lock()
        
        self._initialize_communication()
    
    def _initialize_communication(self):
        """Initialize communication infrastructure."""
        # Initialize per-device statistics
        for device_id in self.devices:
            self.device_stats[device_id] = CommStats()
            self.comm_queues[device_id] = queue.PriorityQueue()
        
        # Initialize NCCL if available and enabled (skip in testing)
        if (self.enable_nccl and torch.cuda.is_available() and 
            len(self.devices) > 1 and 
            not os.environ.get('TESTING', False)):
            try:
                if not dist.is_initialized():
                    # Initialize for single-node multi-GPU
                    dist.init_process_group(
                        backend='nccl',
                        init_method='env://',
                        world_size=len(self.devices),
                        rank=0
                    )
                logger.info("NCCL communication initialized")
            except Exception as e:
                logger.debug(f"Failed to initialize NCCL: {e}")
                self.enable_nccl = False
        
        self.active = True
        self._start_comm_workers()
    
    def _start_comm_workers(self):
        """Start communication worker threads."""
        for device_id in self.devices:
            worker = threading.Thread(
                target=self._comm_worker,
                args=(device_id,),
                daemon=True
            )
            worker.start()
            self.comm_workers[device_id] = worker
        
        logger.info(f"Started communication workers for {len(self.devices)} devices")
    
    def _comm_worker(self, device_id: int):
        """Communication worker for a specific device."""
        while self.active:
            try:
                # Get operation with timeout
                priority, timestamp, op = self.comm_queues[device_id].get(timeout=1.0)
                
                # Execute communication operation
                start_time = time.time()
                self._execute_comm_op(op)
                latency = time.time() - start_time
                
                # Update statistics
                self._update_stats(device_id, op, latency)
                
                # Mark task as done
                self.comm_queues[device_id].task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Communication worker error on device {device_id}: {e}")
                self.device_stats[device_id].error_count += 1
    
    def _execute_comm_op(self, op: CommOp):
        """Execute a communication operation."""
        try:
            if op.pattern == CommPattern.BROADCAST:
                self._broadcast(op.tensor, op.src_device, op.dst_devices)
            elif op.pattern == CommPattern.ALL_REDUCE:
                self._all_reduce(op.tensor, op.dst_devices)
            elif op.pattern == CommPattern.ALL_GATHER:
                self._all_gather(op.tensor, op.dst_devices)
            elif op.pattern == CommPattern.POINT_TO_POINT:
                self._point_to_point(op.tensor, op.src_device, op.dst_devices[0])
            elif op.pattern == CommPattern.REDUCE_SCATTER:
                self._reduce_scatter(op.tensor, op.dst_devices)
            
            # Execute callback if provided
            if op.callback:
                op.callback(op)
                
        except Exception as e:
            logger.error(f"Failed to execute communication operation {op.pattern}: {e}")
            raise
    
    def _broadcast(self, tensor: torch.Tensor, src_device: int, dst_devices: List[int]):
        """Broadcast tensor from source to destination devices."""
        if self.enable_nccl and len(dst_devices) > 1:
            # Use NCCL for efficient broadcast
            with torch.cuda.device(src_device):
                dist.broadcast(tensor, src=src_device)
        else:
            # Manual broadcast
            src_tensor = tensor.to(f'cuda:{src_device}')
            for dst_device in dst_devices:
                if dst_device != src_device:
                    dst_tensor = src_tensor.to(f'cuda:{dst_device}', non_blocking=True)
    
    def _all_reduce(self, tensor: torch.Tensor, devices: List[int]):
        """All-reduce operation across devices."""
        if self.enable_nccl:
            with torch.cuda.device(tensor.device):
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        else:
            # Manual all-reduce
            tensors = []
            for device_id in devices:
                device_tensor = tensor.to(f'cuda:{device_id}')
                tensors.append(device_tensor)
            
            # Sum all tensors
            result = torch.stack(tensors).sum(dim=0)
            
            # Copy result back to all devices
            for i, device_id in enumerate(devices):
                tensors[i].copy_(result)
    
    def _all_gather(self, tensor: torch.Tensor, devices: List[int]):
        """All-gather operation across devices."""
        if self.enable_nccl:
            tensor_list = [torch.empty_like(tensor) for _ in devices]
            with torch.cuda.device(tensor.device):
                dist.all_gather(tensor_list, tensor)
            return tensor_list
        else:
            # Manual all-gather
            tensors = []
            for device_id in devices:
                device_tensor = tensor.to(f'cuda:{device_id}')
                tensors.append(device_tensor)
            return tensors
    
    def _point_to_point(self, tensor: torch.Tensor, src_device: int, dst_device: int):
        """Point-to-point transfer between devices."""
        if src_device == dst_device:
            return tensor
        
        # Optimized device-to-device transfer
        with torch.cuda.device(src_device):
            dst_tensor = tensor.to(f'cuda:{dst_device}', non_blocking=True)
            torch.cuda.synchronize()
        
        return dst_tensor
    
    def _reduce_scatter(self, tensor: torch.Tensor, devices: List[int]):
        """Reduce-scatter operation across devices."""
        if self.enable_nccl:
            output = torch.empty_like(tensor)
            with torch.cuda.device(tensor.device):
                dist.reduce_scatter(output, [tensor], op=dist.ReduceOp.SUM)
            return output
        else:
            # Manual reduce-scatter
            chunk_size = tensor.size(0) // len(devices)
            chunks = torch.chunk(tensor, len(devices), dim=0)
            
            results = []
            for i, device_id in enumerate(devices):
                device_chunk = chunks[i].to(f'cuda:{device_id}')
                results.append(device_chunk)
            
            return results
    
    def _update_stats(self, device_id: int, op: CommOp, latency: float):
        """Update communication statistics."""
        with self.lock:
            # Device-specific stats
            device_stats = self.device_stats[device_id]
            device_stats.total_ops += 1
            device_stats.total_bytes += op.tensor.numel() * op.tensor.element_size()
            device_stats.avg_latency = (
                (device_stats.avg_latency * (device_stats.total_ops - 1) + latency) /
                device_stats.total_ops
            )
            
            # Global stats
            self.stats.total_ops += 1
            self.stats.total_bytes += op.tensor.numel() * op.tensor.element_size()
            self.stats.avg_latency = (
                (self.stats.avg_latency * (self.stats.total_ops - 1) + latency) /
                self.stats.total_ops
            )
    
    def async_broadcast(self, tensor: torch.Tensor, src_device: int, 
                       dst_devices: List[int], priority: int = 0) -> Future:
        """Asynchronous broadcast operation."""
        op = CommOp(
            pattern=CommPattern.BROADCAST,
            tensor=tensor,
            src_device=src_device,
            dst_devices=dst_devices,
            priority=priority
        )
        
        return self.executor.submit(self._schedule_operation, op)
    
    def async_all_reduce(self, tensor: torch.Tensor, devices: List[int], 
                        priority: int = 0) -> Future:
        """Asynchronous all-reduce operation."""
        op = CommOp(
            pattern=CommPattern.ALL_REDUCE,
            tensor=tensor,
            src_device=tensor.device.index,
            dst_devices=devices,
            priority=priority
        )
        
        return self.executor.submit(self._schedule_operation, op)
    
    def async_transfer(self, tensor: torch.Tensor, src_device: int, 
                      dst_device: int, priority: int = 0) -> Future:
        """Asynchronous point-to-point transfer."""
        op = CommOp(
            pattern=CommPattern.POINT_TO_POINT,
            tensor=tensor,
            src_device=src_device,
            dst_devices=[dst_device],
            priority=priority
        )
        
        return self.executor.submit(self._schedule_operation, op)
    
    def _schedule_operation(self, op: CommOp):
        """Schedule a communication operation."""
        timestamp = time.time()
        priority_item = (-op.priority, timestamp, op)  # Negative for max priority
        
        # Add to appropriate device queue
        device_id = op.src_device
        self.comm_queues[device_id].put(priority_item)
    
    def optimize_transfer_pattern(self, tensors: List[torch.Tensor], 
                                 src_devices: List[int], dst_devices: List[int]) -> List[Future]:
        """Optimize transfer pattern for multiple tensors."""
        futures = []
        
        # Group transfers by bandwidth requirements
        large_transfers = []
        small_transfers = []
        
        for tensor, src, dst in zip(tensors, src_devices, dst_devices):
            transfer_size = tensor.numel() * tensor.element_size()
            if transfer_size > self.overlap_threshold:
                large_transfers.append((tensor, src, dst))
            else:
                small_transfers.append((tensor, src, dst))
        
        # Schedule large transfers with higher priority
        for tensor, src, dst in large_transfers:
            future = self.async_transfer(tensor, src, dst, priority=10)
            futures.append(future)
        
        # Schedule small transfers with lower priority
        for tensor, src, dst in small_transfers:
            future = self.async_transfer(tensor, src, dst, priority=1)
            futures.append(future)
        
        return futures
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics."""
        with self.lock:
            return {
                'global_stats': {
                    'total_operations': self.stats.total_ops,
                    'total_bytes_transferred': self.stats.total_bytes,
                    'average_latency_ms': self.stats.avg_latency * 1000,
                    'error_count': self.stats.error_count
                },
                'device_stats': {
                    device_id: {
                        'operations': stats.total_ops,
                        'bytes_transferred': stats.total_bytes,
                        'avg_latency_ms': stats.avg_latency * 1000,
                        'errors': stats.error_count
                    }
                    for device_id, stats in self.device_stats.items()
                }
            }
    
    def wait_for_completion(self, timeout: Optional[float] = None):
        """Wait for all pending communication operations to complete."""
        start_time = time.time()
        
        for device_id in self.devices:
            queue_obj = self.comm_queues[device_id]
            
            while not queue_obj.empty():
                if timeout and (time.time() - start_time) > timeout:
                    logger.warning(f"Timeout waiting for communication completion")
                    return False
                
                time.sleep(0.01)
        
        return True
    
    def cleanup(self):
        """Clean up communication optimizer resources."""
        self.active = False
        
        # Wait for workers to finish
        for worker in self.comm_workers.values():
            if worker.is_alive():
                worker.join(timeout=1.0)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Cleanup NCCL if initialized
        if self.enable_nccl and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception as e:
                logger.warning(f"Error cleaning up NCCL: {e}")
        
        logger.info("Communication optimizer cleanup completed")
