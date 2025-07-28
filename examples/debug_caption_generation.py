#!/usr/bin/env python3
"""
Debug caption generation tensor flow
"""

import torch
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.inference.multi_device_inference import MultiDeviceInferenceSystem

def main():
    print(" Caption Generation Debug Mode")
    print("=" * 40)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize system
    system = MultiDeviceInferenceSystem(
        model_path=None,
        device=device,
        splitting_strategy='encoder_decoder'
    )
    
    # Register minimal devices for easier debugging
    system.register_device(0, 'mobile', compute_power=0.5)
    system.register_device(1, 'server', compute_power=1.0)
    
    # Test data
    test_image = torch.randn(1, 3, 224, 224, device=device)
    
    print(f"\n Testing Encoder-Decoder Split (2 devices)")
    print("-" * 50)
    
    # Test with debug mode enabled
    result = system.inference(
        images=test_image,
        captions=torch.randint(0, 100, (1, 5), device=device),
        num_devices=2,
        strategy='encoder_decoder'
    )
    
    # Analyze results
    print(f"\n Debug Analysis:")
    print("-" * 20)
    
    debug_info = result.get('debug_info', {})
    tensor_flow = debug_info.get('tensor_flow_log', [])
    
    for entry in tensor_flow:
        print(f"Device {entry['device_id']}:")
        print(f"  Input keys: {entry['input_keys']}")
        print(f"  Output keys: {entry['output_keys']}")
        print(f"  Encoder complete: {entry['encoder_complete_present']}")
    
    encoder_complete_devices = debug_info.get('encoder_complete_devices', [])
    print(f"\nDevices with encoder_complete: {encoder_complete_devices}")
    
    final_logits = result.get('final_logits')
    print(f"Final logits generated: {final_logits is not None}")
    
    if final_logits is not None:
        print(f"Logits shape: {final_logits.shape}")
        # Check if logits have meaningful values
        logits_range = (final_logits.min().item(), final_logits.max().item())
        print(f"Logits range: {logits_range}")
        
        # Test token generation
        probs = torch.softmax(final_logits[0, 0], dim=-1)
        top_tokens = torch.topk(probs, 5)
        print(f"Top 5 token probabilities: {top_tokens.values.tolist()}")
        print(f"Top 5 token indices: {top_tokens.indices.tolist()}")
    
    print(f"\n{'✅' if final_logits is not None else '❌'} Caption generation pipeline test complete")

if __name__ == "__main__":
    main()
    input("Press Enter to exit...")
