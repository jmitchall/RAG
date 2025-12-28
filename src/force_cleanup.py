"""
Emergency cleanup script for stuck vLLM processes.
Run this if you get NCCL or process group warnings.
"""
import torch
import subprocess

def force_cleanup():
    """
    Emergency cleanup script for stuck vLLM processes.
    Run this if you get NCCL or process group warnings.
    # If you get stuck with NCCL warnings:

    uv run python -m vllm_srv.force_cleanup

    # Or manually:
    pkill -9 -f vllm
    python -c "import torch; torch.distributed.destroy_process_group() if torch.distributed.is_initialized() else None"

    """
    print("🧹 Emergency vLLM cleanup...")
    
    # 1. Destroy PyTorch distributed if initialized
    try:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            print("✅ Destroyed PyTorch process group")
    except Exception as e:
        print(f"⚠️  Could not destroy process group: {e}")
    
    # 2. Clear CUDA cache
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("✅ Cleared CUDA cache")
    except Exception as e:
        print(f"⚠️  Could not clear CUDA: {e}")
    
    # 3. Kill vLLM processes
    try:
        subprocess.run(["pkill", "-9", "-f", "vllm"], check=False)
        print("✅ Killed vLLM processes")
    except Exception as e:
        print(f"⚠️  Could not kill processes: {e}")
    
    # 4. Kill Python processes using CUDA
    try:
        subprocess.run(["pkill", "-9", "-f", "python.*cuda"], check=False)
        print("✅ Killed CUDA Python processes")
    except Exception as e:
        print(f"⚠️  Could not kill CUDA processes: {e}")
    
    print("✅ Cleanup complete! Reboot recommended for cleanest state.")

if __name__ == "__main__":
    force_cleanup()