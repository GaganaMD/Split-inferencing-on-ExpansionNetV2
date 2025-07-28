#!/usr/bin/env python3
import torch
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.inference.multi_device_inference import MultiDeviceInferenceSystem

def main():
    print(" Simple Multi-Device Test")
    print("=" * 30)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize system
    system = MultiDeviceInferenceSystem(
        model_path=None,
        device=device,
        splitting_strategy='encoder_decoder'  # Use working strategy
    )
    
    # Register devices
    system.register_device(0, 'mobile', compute_power=0.5)
    system.register_device(1, 'server', compute_power=1.0)
    
    # Simple test data
    test_image = torch.randn(1, 3, 224, 224, device=device)
    test_caption = torch.randint(0, 100, (1, 5), device=device)
    
    # Test working strategy first
    print("\n✅ Testing encoder_decoder (should work):")
    try:
        result = system.inference(
            images=test_image,
            captions=test_caption,
            num_devices=2,
            strategy='encoder_decoder'
        )
        final_logits = result.get('final_logits')
        print(f"Success: Final logits = {final_logits is not None}")
        if final_logits is not None:
            print(f"Logits shape: {final_logits.shape}")
    except Exception as e:
        print(f"Failed: {e}")
    
    # Test caption generation
    print("\n Testing caption generation:")
    try:
        caption_tokens = system.generate_caption(test_image, max_length=10, num_devices=2)
        non_zero_tokens = [t for t in caption_tokens if t != 0]
        print(f"Success: Generated {len(non_zero_tokens)} tokens: {non_zero_tokens}")
    except Exception as e:
        print(f"Failed: {e}")
    
    print("\n✅ Simple test completed!")

if __name__ == "__main__":
    main()
    input("Press Enter to exit...")
