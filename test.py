import torch
import time

print("CUDA available:", torch.cuda.is_available())
print("Using device:", torch.cuda.get_device_name(0))

# Send large tensors to GPU to simulate load
print("Running GPU stress test...")
start = time.time()
for i in range(10):
    a = torch.randn(10000, 10000, device='cuda')
    b = torch.randn(10000, 10000, device='cuda')
    c = torch.matmul(a, b)  # Matrix multiply on GPU
    torch.cuda.synchronize()
    print(f"Iteration {i+1}/10 complete")
end = time.time()

print("Completed in", round(end - start, 2), "seconds")
