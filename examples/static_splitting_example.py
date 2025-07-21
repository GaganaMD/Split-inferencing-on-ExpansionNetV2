#!/usr/bin/env python3
"""
Static splitting strategies example
"""

import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.expansionnet_inference import InferenceExpansionNet
from src.inference.static_splitting import StaticSplittingStrategy

def main():
    print("Static Splitting Strategies Demo")
    print("=" * 40)
    
    # Create model
    model = InferenceExpansionNet()
    splitter = StaticSplittingStrategy(model)
    
    print(f"Model has {model.get_total_layers()} layers ({model.N_enc} encoder + {model.N_dec} decoder)")
    
    # Test different static strategies
    print(f"\n1. Even Split (4 devices):")
    config = splitter.even_split(4)
    print_split_config(config)
    
    print(f"\n2. Capability Weighted Split:")
    device_capabilities = [0.3, 0.5, 0.8, 1.0]  # Different device powers
    config = splitter.capability_weighted_split(device_capabilities)
    print_split_config(config)
    
    print(f"\n3. Bandwidth Aware Split:")
    bandwidths = [50, 100, 200]  # Mbps between 4 devices
    latencies = [30, 20, 10]     # ms between 4 devices
    config = splitter.bandwidth_aware_split(bandwidths, latencies)
    print_split_config(config)
    
    print(f"\n4. Encoder-Decoder Split:")
    config = splitter.encoder_decoder_split(3)
    print_split_config(config)
    
    # Layer complexity analysis
    print(f"\n📈 Layer Complexity Analysis:")
    complexities = model.estimate_layer_complexity()
    print(f"Encoder layers: {[f'{c:.2f}' for c in complexities[:model.N_enc]]} GFLOPs")
    print(f"Decoder layers: {[f'{c:.2f}' for c in complexities[model.N_enc:]]} GFLOPs")
    print(f"Total complexity: {sum(complexities):.2f} GFLOPs")

def print_split_config(config):
    """Helper function to print split configuration"""
    print(f"  Strategy: {config.strategy}")
    print(f"  Total devices: {config.total_devices}")
    for assignment in config.device_assignments:
        layer_count = assignment.layer_end - assignment.layer_start + 1
        print(f"  Device {assignment.device_id} ({assignment.device_type}): "
              f"layers {assignment.layer_start}-{assignment.layer_end} ({layer_count} layers)")
        if assignment.estimated_time > 0:
            print(f"    Estimated time: {assignment.estimated_time:.3f}s")

if __name__ == "__main__":
    main()
