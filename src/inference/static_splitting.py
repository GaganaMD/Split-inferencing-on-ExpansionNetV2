import torch
import numpy as np
from typing import List, Dict, Any
from ..models.expansionnet_inference import InferenceExpansionNet, SplitConfiguration, DeviceAssignment

class StaticSplittingStrategy:
    """
    Static splitting strategies that don't require RL training
    """
    
    def __init__(self, model: InferenceExpansionNet):
        self.model = model
        self.total_layers = model.get_total_layers()
        self.layer_complexities = model.estimate_layer_complexity()
    
    def even_split(self, num_devices: int, device_types: List[str] = None) -> SplitConfiguration:
        """Split layers evenly across devices"""
        if device_types is None:
            device_types = ['edge'] * num_devices
        
        layers_per_device = self.total_layers // num_devices
        remaining_layers = self.total_layers % num_devices
        
        assignments = []
        current_layer = 0
        
        for i in range(num_devices):
            # Distribute remaining layers to first devices
            device_layers = layers_per_device + (1 if i < remaining_layers else 0)
            
            assignments.append(DeviceAssignment(
                device_id=i,
                layer_start=current_layer,
                layer_end=current_layer + device_layers - 1,
                device_type=device_types[i]
            ))
            
            current_layer += device_layers
        
        return SplitConfiguration(
            device_assignments=assignments,
            total_devices=num_devices,
            strategy='even_split'
        )
    
    def capability_weighted_split(self, device_capabilities: List[float]) -> SplitConfiguration:
        """Split based on device computational capabilities"""
        num_devices = len(device_capabilities)
        total_capability = sum(device_capabilities)
        
        assignments = []
        current_layer = 0
        
        for i, capability in enumerate(device_capabilities):
            # Allocate layers proportional to device capability
            layer_fraction = capability / total_capability
            device_layers = max(1, int(self.total_layers * layer_fraction))
            
            # Ensure we don't exceed total layers
            if i == num_devices - 1:  # Last device gets remaining layers
                device_layers = self.total_layers - current_layer
            
            assignments.append(DeviceAssignment(
                device_id=i,
                layer_start=current_layer,
                layer_end=current_layer + device_layers - 1,
                device_type='auto',
                estimated_time=self._estimate_processing_time(current_layer, device_layers, capability)
            ))
            
            current_layer += device_layers
            
            if current_layer >= self.total_layers:
                break
        
        return SplitConfiguration(
            device_assignments=assignments,
            total_devices=len(assignments),
            strategy='capability_weighted'
        )
    
    def bandwidth_aware_split(self, bandwidths: List[float], latencies: List[float]) -> SplitConfiguration:
        """Split considering network bandwidth between devices"""
        num_devices = len(bandwidths) + 1  # N bandwidths for N+1 devices
        
        # Calculate communication costs
        comm_costs = []
        for i in range(len(bandwidths)):
            # Estimate data transfer cost (MB/s conversion and latency)
            transfer_cost = 1.0 / (bandwidths[i] / 8.0) + latencies[i] / 1000.0
            comm_costs.append(transfer_cost)
        
        # Allocate more layers to devices with cheaper communication
        assignments = []
        current_layer = 0
        
        # First device (no incoming communication cost)
        first_device_layers = max(2, self.total_layers // (num_devices * 2))
        assignments.append(DeviceAssignment(
            device_id=0,
            layer_start=0,
            layer_end=first_device_layers - 1,
            device_type='mobile'
        ))
        current_layer = first_device_layers
        
        # Intermediate devices
        remaining_layers = self.total_layers - current_layer
        for i in range(1, num_devices):
            if i < len(comm_costs):
                # More layers for better connections (inverse of cost)
                cost_factor = 1.0 / (comm_costs[i-1] + 0.1)  # Avoid division by zero
                layer_fraction = cost_factor / sum(1.0 / (c + 0.1) for c in comm_costs[i-1:])
                device_layers = max(1, int(remaining_layers * layer_fraction))
            else:
                # Last device gets remaining layers
                device_layers = self.total_layers - current_layer
            
            assignments.append(DeviceAssignment(
                device_id=i,
                layer_start=current_layer,
                layer_end=current_layer + device_layers - 1,
                device_type='server' if i == num_devices - 1 else 'edge'
            ))
            
            current_layer += device_layers
            
            if current_layer >= self.total_layers:
                break
        
        return SplitConfiguration(
            device_assignments=assignments,
            total_devices=len(assignments),
            strategy='bandwidth_aware'
        )
    
    def encoder_decoder_split(self, num_devices: int) -> SplitConfiguration:
        """Split at the encoder-decoder boundary"""
        if num_devices < 2:
            return self.even_split(1)
        
        assignments = []
        
        if num_devices == 2:
            # Simple encoder-decoder split
            assignments = [
                DeviceAssignment(
                    device_id=0,
                    layer_start=0,
                    layer_end=self.model.N_enc - 1,
                    device_type='mobile'
                ),
                DeviceAssignment(
                    device_id=1,
                    layer_start=self.model.N_enc,
                    layer_end=self.total_layers - 1,
                    device_type='server'
                )
            ]
        else:
            # Distribute encoder layers among first devices, decoder to last
            encoder_devices = num_devices - 1
            encoder_layers_per_device = self.model.N_enc // encoder_devices
            
            current_layer = 0
            # Encoder distribution
            for i in range(encoder_devices):
                if i == encoder_devices - 1:  # Last encoder device
                    end_layer = self.model.N_enc - 1
                else:
                    end_layer = current_layer + encoder_layers_per_device - 1
                
                assignments.append(DeviceAssignment(
                    device_id=i,
                    layer_start=current_layer,
                    layer_end=end_layer,
                    device_type='edge'
                ))
                current_layer = end_layer + 1
            
            # Decoder on last device
            assignments.append(DeviceAssignment(
                device_id=num_devices - 1,
                layer_start=self.model.N_enc,
                layer_end=self.total_layers - 1,
                device_type='server'
            ))
        
        return SplitConfiguration(
            device_assignments=assignments,
            total_devices=len(assignments),
            strategy='encoder_decoder_split'
        )
    
    def _estimate_processing_time(self, start_layer: int, num_layers: int, 
                                 device_capability: float) -> float:
        """Estimate processing time for a device segment"""
        total_complexity = sum(self.layer_complexities[start_layer:start_layer + num_layers])
        
        # Base processing time per GFLOP (in seconds)
        base_time_per_gflop = 0.001  # 1ms per GFLOP for reference device
        
        # Adjust by device capability
        processing_time = total_complexity * base_time_per_gflop / device_capability
        
        return processing_time
