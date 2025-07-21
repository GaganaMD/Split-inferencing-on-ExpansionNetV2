import time
import random
from typing import Tuple, List

class NetworkMonitor:
    """Monitor and simulate network conditions"""
    
    def __init__(self):
        self.current_bandwidths = [100.0, 50.0, 200.0]  # Default bandwidths in Mbps
        self.current_latencies = [20.0, 30.0, 15.0]     # Default latencies in ms
        self.last_update = time.time()
    
    def update_conditions(self, bandwidths: List[float], latencies: List[float]):
        """Update network conditions manually"""
        self.current_bandwidths = bandwidths
        self.current_latencies = latencies
        self.last_update = time.time()
    
    def simulate_network_variation(self):
        """Simulate realistic network variations"""
        current_time = time.time()
        
        # Update every 5 seconds
        if current_time - self.last_update > 5.0:
            # Add random variations
            for i in range(len(self.current_bandwidths)):
                variation = random.uniform(-10, 10)  # ±10% variation
                self.current_bandwidths[i] = max(10, self.current_bandwidths[i] + variation)
            
            for i in range(len(self.current_latencies)):
                variation = random.uniform(-5, 5)    # ±5ms variation
                self.current_latencies[i] = max(1, self.current_latencies[i] + variation)
            
            self.last_update = current_time
    
    def get_current_conditions(self) -> Tuple[List[float], List[float]]:
        """Get current network conditions"""
        self.simulate_network_variation()
        return self.current_bandwidths.copy(), self.current_latencies.copy()
    
    def estimate_transfer_time(self, data_size_mb: float, link_index: int = 0) -> float:
        """Estimate data transfer time for specific link"""
        if link_index >= len(self.current_bandwidths):
            link_index = 0
        
        bandwidth = self.current_bandwidths[link_index]
        latency = self.current_latencies[link_index]
        
        # Transfer time + latency
        transfer_time = (data_size_mb * 8) / bandwidth  # Convert to seconds
        total_time = transfer_time + (latency / 1000.0)
        
        return total_time
