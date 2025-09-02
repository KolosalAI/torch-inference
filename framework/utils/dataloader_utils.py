"""
Optimized DataLoader utilities for PyTorch inference framework.

ROI Optimization 0: DataLoader configurations for hiding H2D latency and reducing worker startup stalls.
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Union, Any, Dict
import logging


logger = logging.getLogger(__name__)


def create_optimized_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    num_workers: Optional[int] = None,
    prefetch_factor: Optional[int] = None,
    persistent_workers: Optional[bool] = None,
    pin_memory: Optional[bool] = None,
    **kwargs
) -> DataLoader:
    """
    Create an optimized DataLoader with ROI performance enhancements.
    
    ROI Optimizations applied:
    - pin_memory=True for faster H2D transfers
    - Optimized num_workers based on CPU count and workload
    - persistent_workers=True to reduce worker startup stalls
    - Tuned prefetch_factor to hide latency without excessive RAM usage
    
    Args:
        dataset: Dataset to load from
        batch_size: Batch size for loading
        device: Target device (used to determine if pin_memory should be enabled)
        num_workers: Number of worker processes (auto-calculated if None)
        prefetch_factor: Samples per worker to prefetch (auto-calculated if None)
        persistent_workers: Keep workers alive between epochs (auto-determined if None)
        pin_memory: Enable pinned memory for faster GPU transfers (auto-determined if None)
        **kwargs: Additional DataLoader arguments
        
    Returns:
        Optimized DataLoader instance
    """
    
    # Auto-detect optimal settings if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ROI Optimization 0: pin_memory=True for GPU targets
    if pin_memory is None:
        pin_memory = device.type == 'cuda'
    
    # ROI Optimization 0: Optimal num_workers based on CPU count and workload
    if num_workers is None:
        cpu_count = os.cpu_count() or 4
        if device.type == 'cuda':
            # For GPU: Use more workers to hide H2D transfer latency
            num_workers = min(8, max(2, cpu_count // 2))
        else:
            # For CPU: Use fewer workers to avoid oversubscription
            num_workers = min(4, max(1, cpu_count // 4))
    
    # ROI Optimization 0: persistent_workers=True to reduce startup stalls
    if persistent_workers is None:
        persistent_workers = num_workers > 0
    
    # ROI Optimization 0: prefetch_factor to hide latency without excessive RAM
    if prefetch_factor is None and num_workers > 0:
        # Conservative prefetch: 2-4 batches per worker
        if batch_size * num_workers < 64:
            prefetch_factor = 4  # Small batches can prefetch more
        elif batch_size * num_workers < 256:
            prefetch_factor = 3  # Medium batches
        else:
            prefetch_factor = 2  # Large batches use less prefetch
    
    # Merge with user-provided kwargs
    dataloader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'persistent_workers': persistent_workers,
        **kwargs
    }
    
    # Only add prefetch_factor if num_workers > 0 (not needed for single-threaded)
    if num_workers > 0 and prefetch_factor is not None:
        dataloader_kwargs['prefetch_factor'] = prefetch_factor
    
    logger.info(f"Creating optimized DataLoader: batch_size={batch_size}, "
                f"num_workers={num_workers}, pin_memory={pin_memory}, "
                f"persistent_workers={persistent_workers}, prefetch_factor={prefetch_factor}")
    
    return DataLoader(dataset, **dataloader_kwargs)


def optimize_existing_dataloader(dataloader: DataLoader, device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Analyze an existing DataLoader and provide optimization recommendations.
    
    Args:
        dataloader: Existing DataLoader to analyze
        device: Target device for optimization recommendations
        
    Returns:
        Dictionary with optimization recommendations and warnings
    """
    recommendations = {
        'warnings': [],
        'optimizations': [],
        'current_config': {
            'batch_size': dataloader.batch_size,
            'num_workers': dataloader.num_workers,
            'pin_memory': dataloader.pin_memory,
            'persistent_workers': getattr(dataloader, 'persistent_workers', False),
            'prefetch_factor': getattr(dataloader, 'prefetch_factor', None),
        }
    }
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check pin_memory optimization
    if device.type == 'cuda' and not dataloader.pin_memory:
        recommendations['optimizations'].append(
            "Enable pin_memory=True for faster GPU transfers"
        )
    
    # Check num_workers optimization
    if dataloader.num_workers == 0:
        recommendations['warnings'].append(
            "Single-threaded DataLoader may bottleneck GPU training/inference"
        )
        cpu_count = os.cpu_count() or 4
        optimal_workers = min(8, max(2, cpu_count // 2)) if device.type == 'cuda' else min(4, max(1, cpu_count // 4))
        recommendations['optimizations'].append(
            f"Consider using num_workers={optimal_workers} for better performance"
        )
    
    # Check persistent_workers
    if dataloader.num_workers > 0 and not getattr(dataloader, 'persistent_workers', False):
        recommendations['optimizations'].append(
            "Enable persistent_workers=True to reduce worker startup overhead"
        )
    
    # Check prefetch_factor
    if (dataloader.num_workers > 0 and 
        getattr(dataloader, 'prefetch_factor', None) is None):
        recommendations['optimizations'].append(
            "Set prefetch_factor=2-4 to hide data loading latency"
        )
    
    return recommendations


class OptimizedDataset(Dataset):
    """
    Base dataset class with built-in optimizations for inference workloads.
    """
    
    def __init__(self, data, device: Optional[torch.device] = None):
        self.data = data
        self.device = device or torch.device('cpu')  # Keep data on CPU for proper pinning
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Keep data on CPU for DataLoader with pin_memory=True
        if isinstance(item, torch.Tensor):
            return item.cpu()  # Ensure on CPU for pinning
        elif isinstance(item, (list, tuple)):
            # Convert to CPU tensor
            return torch.tensor(item, dtype=torch.float32)
        else:
            return item


def benchmark_dataloader_performance(
    dataloader: DataLoader, 
    num_batches: int = 10,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Benchmark DataLoader performance to validate optimizations.
    
    Args:
        dataloader: DataLoader to benchmark
        num_batches: Number of batches to time
        device: Device to transfer data to (if different from DataLoader)
        
    Returns:
        Performance metrics dictionary
    """
    import time
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    times = []
    transfer_times = []
    
    # Warmup
    for i, batch in enumerate(dataloader):
        if i >= 3:  # 3 warmup iterations
            break
        if isinstance(batch, torch.Tensor):
            _ = batch.to(device, non_blocking=True)
    
    # Benchmark
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
            
        start_time = time.perf_counter()
        
        # Time data loading
        load_time = time.perf_counter()
        
        # Time device transfer if needed
        if isinstance(batch, torch.Tensor) and batch.device != device:
            batch = batch.to(device, non_blocking=True)
            if device.type == 'cuda':
                torch.cuda.synchronize()  # Ensure transfer is complete
        
        transfer_time = time.perf_counter()
        
        total_time = time.perf_counter() - start_time
        times.append(total_time)
        transfer_times.append(transfer_time - load_time)
    
    if not times:
        return {'error': 'No batches processed'}
    
    return {
        'avg_batch_time': sum(times) / len(times),
        'min_batch_time': min(times),
        'max_batch_time': max(times),
        'avg_transfer_time': sum(transfer_times) / len(transfer_times),
        'total_throughput': len(times) / sum(times),  # batches per second
    }
