#!/usr/bin/env python3
"""
Fixed multi-device inference example with proper tensor handling
"""

import torch
import sys
import os

# Windows-compatible path addition
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

def main():
    print("Fixed Multi-Device ExpansionNet Inference")
    print("=" * 45)
    
    # Test basic PyTorch operations first
    print("🔧 Testing PyTorch setup...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Test tensor operations
    test_tensor = torch.randn(1, 3, 224, 224, device=device)
    print(f"Test tensor shape: {test_tensor.shape}")
    print("✅ PyTorch operations working")
    
    # Import and test the fixed model
    try:
        from src.models.expansionnet_inference import InferenceExpansionNet
        model = InferenceExpansionNet().to(device)
        print("✅ Model created successfully")
        
        # Test single-device forward pass first
        with torch.no_grad():
            dummy_captions = torch.randint(0, 1000, (1, 8), device=device)
            output = model(test_tensor, dummy_captions)
            print(f"✅ Single-device forward pass successful: {output.shape}")
            
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test multi-device system
    try:
        from src.inference.multi_device_inference import MultiDeviceInferenceSystem
        
        inference_system = MultiDeviceInferenceSystem(
            model_path=None,
            device=device,
            splitting_strategy='encoder_decoder'
        )
        print("✅ Multi-device system initialized")
        
        # Register minimal devices
        inference_system.register_device(0, 'mobile', compute_power=0.5)
        inference_system.register_device(1, 'server', compute_power=1.0)
        
        # Test simple 2-device split
        result = inference_system.inference(
            images=test_tensor,
            captions=dummy_captions,
            num_devices=2,
            strategy='encoder_decoder'
        )
        
        print("✅ Multi-device inference successful!")
        print(f"Final output shape: {result.get('final_logits', torch.tensor([])).shape}")
        
    except Exception as e:
        print(f"❌ Multi-device test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎯 All tests completed!")

if __name__ == "__main__":
    main()
    input("Press Enter to exit...")
