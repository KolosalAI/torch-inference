import time
import torch
import psutil
from typing import Dict

class SystemMonitor:
    def __init__(self):
        self.start_time = None
        self.start_mem = None
        
    def start_recording(self):
        self.start_time = time.time()
        self.start_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
    def stop_recording(self) -> Dict[str, float]:
        latency = (time.time() - self.start_time) * 1000  # ms
        
        stats = {
            "latency": latency,
            "cpu_util": psutil.cpu_percent(),
            "ram_util": psutil.virtual_memory().percent
        }
        
        if torch.cuda.is_available():
            stats.update({
                "gpu_mem": (torch.cuda.memory_allocated() - self.start_mem) / 1e9,
                "gpu_util": torch.cuda.utilization()
            })
            
        return stats