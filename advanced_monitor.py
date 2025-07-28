import time
import threading
import psutil
import torch

def monitor_system(interval=1):
    """
    Monitors CPU, memory, and GPU stats every `interval` seconds.
    """
    try:
        while True:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_freq = psutil.cpu_freq()
            mem = psutil.virtual_memory()

            # GPU (if available)
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
                gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # GB
                gpu_util = get_gpu_utilization()
                gpu_name = torch.cuda.get_device_name(0)
            else:
                gpu_memory_allocated = gpu_memory_reserved = gpu_util = gpu_name = None

            # Print stats
            print(f"CPU: {cpu_percent}% | CPU freq: {cpu_freq.current:.2f} MHz | RAM used: {mem.percent}% ({mem.used / (1024 ** 3):.2f} GB / {mem.total / (1024 ** 3):.2f} GB)")
            if gpu_available:
                print(f"GPU {gpu_name}: Memory Allocated {gpu_memory_allocated:.2f} GB, Reserved {gpu_memory_reserved:.2f} GB, Utilization {gpu_util}%")
            else:
                print("GPU not available.")
            print("-" * 60)

            time.sleep(interval)
    except KeyboardInterrupt:
        print("Monitoring stopped.")

def get_gpu_utilization():
    """
    Returns GPU utilization percentage (requires pynvml).
    Falls back to None if not available.
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return util.gpu  # % utilization
    except ImportError:
        return None

# Example usage: monitor in background while running your inference
if __name__ == "__main__":
    import threading

    # Start monitoring thread
    monitor_thread = threading.Thread(target=monitor_system, args=(1,), daemon=True)
    monitor_thread.start()

    # Simulate workload for demonstration (replace this with your inference code)
    import torch.nn as nn

    print("Starting dummy workload...")

    model = nn.Linear(1024, 1024).cuda() if torch.cuda.is_available() else nn.Linear(1024, 1024)
    data = torch.randn(64, 1024).cuda() if torch.cuda.is_available() else torch.randn(64, 1024)

    for i in range(100):
        output = model(data)
        # Simulate some CPU load
        time.sleep(0.1)

    print("Dummy workload complete.")
    input("Press Enter to stop monitoring and exit...")
