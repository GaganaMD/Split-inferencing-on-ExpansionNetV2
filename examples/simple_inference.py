#!/usr/bin/env python3
"""
Simple multi-device inference example - Windows compatible
"""

import torch
import sys
import os

# Windows-compatible path addition
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.inference.multi_device_inference import MultiDeviceInferenceSystem
from config.inference_config import InferenceConfig

def main():
    print("Multi-Device ExpansionNet Inference Example")
    print("=" * 50)
    
    # Initialize inference system
    # Note: Pass None for model_path to use random weights for demo
    try:
        inference_system = MultiDeviceInferenceSystem(
    model_path=None,
    device='cpu',  # Add this line
    splitting_strategy='even_split'
)
        print("✅ Inference system initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize inference system: {e}")
        return
    
    # Register devices
    print("\nRegistering devices...")
    try:
        inference_system.register_device(0, 'mobile', compute_power=0.3)
        inference_system.register_device(1, 'edge', compute_power=0.5)
        inference_system.register_device(2, 'edge', compute_power=0.6)
        inference_system.register_device(3, 'server', compute_power=1.0)
        print("✅ All devices registered successfully")
    except Exception as e:
        print(f"❌ Failed to register devices: {e}")
        return
    
    # Set network conditions (bandwidths and latencies between devices)
    print("Setting network conditions...")
    bandwidths = [80.0, 120.0, 500.0]  # Mbps between device pairs
    latencies = [25.0, 15.0, 10.0]     # ms between device pairs
    inference_system.set_network_conditions(bandwidths, latencies)
    
    # Create dummy input data
    print("\nCreating test data...")
    try:
        dummy_image = torch.randn(1, 3, 224, 224)  # Single image
        dummy_captions = torch.randint(0, 10201, (1, 10))  # Short caption
        print(f"Input image shape: {dummy_image.shape}")
        print(f"Input caption shape: {dummy_captions.shape}")
        print("✅ Test data created successfully")
    except Exception as e:
        print(f"❌ Failed to create test data: {e}")
        return
    
    # Test different splitting strategies
    strategies = ['even_split', 'capability_weighted', 'bandwidth_aware', 'encoder_decoder']
    
    print(f"\nTesting {len(strategies)} splitting strategies:")
    print("-" * 40)
    
    successful_strategies = 0
    
    for strategy in strategies:
        print(f"\n🔄 Testing {strategy}...")
        
        try:
            result = inference_system.inference(
                images=dummy_image,
                captions=dummy_captions,
                num_devices=3,
                strategy=strategy
            )
            
            metrics = result.get('metrics', {})
            print(f"✅ Success! Total time: {result.get('total_inference_time', 0):.3f}s")
            print(f"   Devices used: {metrics.get('device_count', 0)}")
            print(f"   Processing time: {metrics.get('total_processing_time', 0):.3f}s")
            print(f"   Communication time: {metrics.get('total_communication_time', 0):.3f}s")
            print(f"   Data transferred: {metrics.get('total_data_transferred_mb', 0):.2f} MB")
            print(f"   Load balance: {metrics.get('load_balance_score', 0):.2f}")
            successful_strategies += 1
            
        except Exception as e:
            print(f"❌ Failed: {str(e)}")
    
    print(f"\n Strategy Test Results: {successful_strategies}/{len(strategies)} successful")
    
    # Generate caption example
    print(f"\n Caption Generation Example:")
    print("-" * 30)
    
    try:
        caption_tokens = inference_system.generate_caption(
            dummy_image, 
            max_length=15, 
            num_devices=2
        )
        print(f"Generated caption tokens: {caption_tokens}")
        print(f"Caption length: {len([t for t in caption_tokens if t != 0])} tokens")
        print("✅ Caption generation successful")
        
    except Exception as e:
        print(f"❌ Caption generation failed: {e}")
    
    # Performance statistics
    print(f"\n Performance Statistics:")
    print("-" * 25)
    
    try:
        stats = inference_system.get_performance_stats()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.3f}")
            else:
                print(f"{key}: {value}")
        print("✅ Performance stats retrieved successfully")
    except Exception as e:
        print(f"❌ Failed to get performance stats: {e}")
    
    print(f"\n✅ Demo completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPress Enter to exit...")