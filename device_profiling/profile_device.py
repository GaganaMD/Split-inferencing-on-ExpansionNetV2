import psutil
import torch
import platform

def profile_cpu_ram():
    print("===== CPU & RAM =====")
    print(f"CPU Count: {psutil.cpu_count(logical=True)}")
    print(f"CPU Usage: {psutil.cpu_percent(interval=1)}%")
    vm = psutil.virtual_memory()
    print(f"RAM Total: {vm.total / 1e9:.2f} GB")
    print(f"RAM Available: {vm.available / 1e9:.2f} GB")
    print(f"RAM Used: {vm.used / 1e9:.2f} GB ({vm.percent}%)")

def profile_python():
    print("===== Python & OS =====")
    print(f"Python version: {platform.python_version()}")
    print(f"OS: {platform.system()} {platform.release()}")

def profile_gpu():
    print("===== GPU (CUDA) =====")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory Allocated: {torch.cuda.memory_allocated(i)/1e9:.2f} GB")
            print(f"  Memory Reserved: {torch.cuda.memory_reserved(i)/1e9:.2f} GB")
    else:
        print("No CUDA GPU detected")

if __name__ == "__main__":
    profile_python()
    profile_cpu_ram()
    profile_gpu()
