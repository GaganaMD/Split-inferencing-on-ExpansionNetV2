#!/usr/bin/env python3
import torch
import sys
import os
from src.models.expansionnet_inference import InferenceExpansionNet

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

def main():
    print("Testing Fixed Visual Encoder")
    print("=" * 30)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Test the fixed model
    try:
        from src.models.expansionnet_inference_fixed import InferenceExpansionNet
        model = InferenceExpansionNet().to(device)
        print("✅ Fixed model created successfully")
        
        # Test forward pass
        test_image = torch.randn(1, 3, 224, 224, device=device)
        test_captions = torch.randint(0, 1000, (1, 8), device=device)
        
        with torch.no_grad():
            output = model(test_image, test_captions)
            print(f"✅ Forward pass successful!")
            print(f"Output shape: {output.shape}")
            print(f"Expected shape: (1, 8, vocab_size)")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
    input("Press Enter to exit...")
