import psutil
import torch
from typing import Dict, Any

class DeviceMonitor:
    """Monitor device resource utilization"""
    
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
    
    def get_device_stats(self, device_id: int = 0) -> Dict[str, Any]:
        """Get current device statistics"""
        stats = {
            'device_id': device_id,
            'timestamp': time.time()
        }
        
        # CPU stats
        stats.update({
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'available_memory_gb': psutil.virtual_memory().available / (1024**3)
        })
        
        # GPU stats if available
        if self.gpu_available and device_id == 0:
            try:
                gpu_memory = torch.cuda.memory_stats()
                stats.update({
                    'gpu_memory_allocated_gb': gpu_memory.get('allocated_bytes.all.current', 0) / (1024**3),
                    'gpu_memory_reserved_gb': gpu_memory.get('reserved_bytes.all.current', 0) / (1024**3),
                    'gpu_utilization_percent': 50.0  # Placeholder - would need nvidia-ml-py for real stats
                })
            except:
                stats['gpu_available'] = False
        
        return stats
    
    def estimate_device_capability(self, device_stats: Dict[str, Any]) -> float:
        """Estimate device capability score (0-1)"""
        cpu_factor = max(0, 1.0 - device_stats.get('cpu_percent', 50) / 100.0)
        memory_factor = max(0, 1.0 - device_stats.get('memory_percent', 50) / 100.0)
        
        # Weight CPU and memory
        capability = cpu_factor * 0.6 + memory_factor * 0.4
        
        # Boost for GPU availability
        if device_stats.get('gpu_available', False):
            capability *= 1.5
        
        return min(1.0, capability)

import time
