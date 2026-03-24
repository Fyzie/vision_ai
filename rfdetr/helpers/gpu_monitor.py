import torch
import weakref
import gc

def check_gpu_vram(required_gb=8):
    if not torch.cuda.is_available():
        print("[WARNING] CUDA not available. Training will run on CPU.")
        return False

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    total = props.total_memory / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    free = total - (reserved + allocated)

    print(f"[GPU] Model: {props.name}")
    print(f"[GPU] Total VRAM : {total:.2f} GB")
    print(f"[GPU] Free VRAM  : {free:.2f} GB")

    if free < required_gb:
        print(f"[WARNING] Expected at least {required_gb} GB free. You may OOM.")
        return False

    print("[GPU] VRAM check passed.")
    return True

def cleanup_gpu_memory(obj=None, verbose=False):
    if not torch.cuda.is_available():
        return

    def stats():
        return (
            torch.cuda.memory_allocated() / 1024**2,
            torch.cuda.memory_reserved() / 1024**2,
        )

    torch.cuda.synchronize()

    if verbose:
        a, r = stats()
        print(f"[Cleanup Before] Alloc: {a:.2f} MB | Reserved: {r:.2f} MB")

    if obj is not None:
        ref = weakref.ref(obj)
        del obj
        if ref() is not None and verbose:
            print("[WARNING] Object not fully deleted.")

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.synchronize()

    if verbose:
        a, r = stats()
        print(f"[Cleanup After]  Alloc: {a:.2f} MB | Reserved: {r:.2f} MB")