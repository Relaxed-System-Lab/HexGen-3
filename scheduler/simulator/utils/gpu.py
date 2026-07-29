from pynvml import *

NVML_INITIALIZED=False

def get_gpu_spec():
    global NVML_INITIALIZED
    if not NVML_INITIALIZED:
        try:
            nvmlInit()
            NVML_INITIALIZED = True
        except NVMLError as e:
            print(f"Failed to initialize NVML: {e}")
            return None

    gpu_count = nvmlDeviceGetCount()
    gpu_specs = []

    for i in range(gpu_count):
        handle = nvmlDeviceGetHandleByIndex(i)
        name = nvmlDeviceGetName(handle)
        memory_info = nvmlDeviceGetMemoryInfo(handle)
        gpu_specs.append({
            "index": i,
            "name": name,
            "memory_total": memory_info.total,
            "memory_free": memory_info.free,
            "memory_used": memory_info.used
        })

    return gpu_specs

if __name__ == "__main__":
    specs = get_gpu_spec()
    print(specs)