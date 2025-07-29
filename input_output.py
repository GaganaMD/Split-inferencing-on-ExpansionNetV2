import torch
import sys
import os

# Ensure local imports work (adjust relative to your project root if needed)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.inference.multi_device_inference import MultiDeviceInferenceSystem

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize system with encoder-decoder split for reliability
    system = MultiDeviceInferenceSystem(
        model_path=None,               # Use None for random demo weights; for real, set path to weights
        device=device,
        splitting_strategy='encoder_decoder'
    )

    # Register simulated devices (change compute_power/types as appropriate)
    system.register_device(0, 'mobile', compute_power=0.4)
    system.register_device(1, 'server', compute_power=1.0)

    # Prepare a dummy image (replace with your image tensor if you have one)
    image = torch.randn(1, 3, 224, 224, device=device)

    print("\n--- Inference (Forward Pass) ---")
    # Run full inference (get logits for every token position in a dummy caption)
    dummy_caption = torch.randint(0, 1000, (1, 8), device=device)  # Dummy 8-token caption
    result = system.inference(
        images=image,
        captions=dummy_caption,
        num_devices=2,
        strategy='encoder_decoder'
    )
    logits = result.get('final_logits')
    if logits is not None:
        print(f"Logits shape: {logits.shape}")
    else:
        print("No logits generated.")

    print("\n--- Caption Generation (Token-by-Token) ---")
    # Generate an auto-regressive caption given the image
    caption_tokens = system.generate_caption(
        image, max_length=10, num_devices=2
    )
    print(f"Generated tokens: {caption_tokens}")

if __name__ == "__main__":
    main()