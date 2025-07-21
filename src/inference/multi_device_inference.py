import torch
import time
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import asdict

from ..models.expansionnet_inference import InferenceExpansionNet, SplitConfiguration
from .static_splitting import StaticSplittingStrategy
from .coordination_manager import InferenceCoordinator
from ..utils.device_utils import DeviceMonitor
from ..utils.network_utils import NetworkMonitor

class MultiDeviceInferenceSystem:
    """
    Complete multi-device inference system with multiple splitting strategies
    """
    
    def __init__(self, model_path: str, device: str = 'cuda', 
                 splitting_strategy: str = 'even_split'):
        self.device = device
        self.splitting_strategy = splitting_strategy
        
        # Load model
        self.model = InferenceExpansionNet().to(device)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        
        # Initialize components
        self.static_splitter = StaticSplittingStrategy(self.model)
        self.coordinator = InferenceCoordinator()
        self.device_monitor = DeviceMonitor()
        self.network_monitor = NetworkMonitor()
        
        # Performance tracking
        self.performance_history = []
        
        # Available devices
        self.available_devices = []
        self.device_capabilities = {}
    
    def register_device(self, device_id: int, device_type: str, 
                       compute_power: float = 0.5, **kwargs):
        """Register an available device"""
        device_info = {
            'device_id': device_id,
            'device_type': device_type,
            'compute_power': compute_power,
            **kwargs
        }
        
        if device_id not in self.available_devices:
            self.available_devices.append(device_id)
        
        self.device_capabilities[device_id] = device_info
        print(f"Registered device {device_id} ({device_type}) with compute power {compute_power}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.performance_history:
            return {'message': 'No inference history available'}
        
        recent_history = self.performance_history[-50:]  # Last 50 inferences
        
        times = [record['total_time'] for record in recent_history]
        device_counts = [record['split_config']['total_devices'] for record in recent_history]
        
        stats = {
            'total_inferences': len(self.performance_history),
            'recent_avg_time': np.mean(times) if times else 0,
            'recent_std_time': np.std(times) if times else 0,
            'avg_devices_used': np.mean(device_counts) if device_counts else 0,
            'fastest_time': min(times) if times else 0,
            'slowest_time': max(times) if times else 0,
            'registered_devices': len(self.available_devices),
            'strategies_used': list(set(record['split_config']['strategy'] for record in recent_history))
        }
        
        return stats

    
    def set_network_conditions(self, bandwidths: List[float], latencies: List[float]):
        """Set network conditions between devices"""
        self.network_monitor.update_conditions(bandwidths, latencies)
    
    def get_split_configuration(self, num_devices: int = None, 
                               strategy: str = None) -> SplitConfiguration:
        """Get split configuration based on current conditions"""
        if strategy is None:
            strategy = self.splitting_strategy
        
        if num_devices is None:
            num_devices = min(len(self.available_devices), 4)  # Default to 4 devices
        
        # Ensure we don't exceed available devices
        num_devices = min(num_devices, len(self.available_devices))
        
        if strategy == 'even_split':
            device_types = [self.device_capabilities.get(i, {}).get('device_type', 'edge') 
                          for i in range(num_devices)]
            return self.static_splitter.even_split(num_devices, device_types)
        
        elif strategy == 'capability_weighted':
            capabilities = [self.device_capabilities.get(i, {}).get('compute_power', 0.5) 
                          for i in range(num_devices)]
            return self.static_splitter.capability_weighted_split(capabilities)
        
        elif strategy == 'bandwidth_aware':
            bandwidths, latencies = self.network_monitor.get_current_conditions()
            return self.static_splitter.bandwidth_aware_split(bandwidths[:num_devices-1], 
                                                            latencies[:num_devices-1])
        
        elif strategy == 'encoder_decoder':
            return self.static_splitter.encoder_decoder_split(num_devices)
        
        else:
            # Default to even split
            return self.static_splitter.even_split(num_devices)
    
    def inference(self, images: torch.Tensor, captions: torch.Tensor = None,
                 num_devices: int = None, strategy: str = None) -> Dict[str, Any]:
        """
        Perform multi-device inference
        """
        start_time = time.time()
        
        # Get split configuration
        split_config = self.get_split_configuration(num_devices, strategy)
        
        print(f"Using {split_config.total_devices} devices with {split_config.strategy} strategy:")
        for assignment in split_config.device_assignments:
            print(f"  Device {assignment.device_id}: layers {assignment.layer_start}-{assignment.layer_end} ({assignment.device_type})")
        
        # Execute coordinated inference
        result = self.coordinator.execute_pipeline(
            model=self.model,
            split_config=split_config,
            images=images,
            captions=captions,
            device_capabilities=self.device_capabilities,
            network_conditions=self.network_monitor.get_current_conditions()
        )
        
        total_time = time.time() - start_time
        result['total_inference_time'] = total_time
        
        # Update performance history
        performance_record = {
            'timestamp': time.time(),
            'split_config': asdict(split_config),
            'metrics': result.get('metrics', {}),
            'total_time': total_time
        }
        self.performance_history.append(performance_record)
        
        return result
    
    def generate_caption(self, image: torch.Tensor, max_length: int = 20,
                    temperature: float = 1.0, num_devices: int = None) -> List[int]:
        """Generate caption using multi-device inference with proper tensor handling"""
        batch_size = 1
        device = image.device
        
        # Ensure correct image dimensions - FIX for 5D tensor issue
        if image.dim() == 3:
            image = image.unsqueeze(0)  # Add batch dimension: (3,224,224) -> (1,3,224,224)
        elif image.dim() == 5:
            image = image.squeeze(1).squeeze(1)  # Remove extra dimensions
        
        # Start with BOS token
        generated = torch.zeros(batch_size, max_length, dtype=torch.long, device=device)
        generated[:, 0] = 1  # BOS token
        
        for i in range(1, max_length):
            current_seq = generated[:, :i]
            
            # Perform inference
            result = self.inference(
                images=image,
                captions=current_seq,
                num_devices=num_devices
            )
            
            logits = result.get('final_logits')
            if logits is None:
                print("Warning: No logits generated, stopping generation")
                break
            
            # Sample next token
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated[:, i] = next_token.squeeze(1)
            
            # Stop if EOS token
            if next_token.item() == 2:  # EOS token
                break
        
        return generated[0].tolist()

