#!/usr/bin/env python3
"""
GPU Health Check Script
Run this before starting the reflection agent to ensure GPU is ready
"""
import subprocess
import sys

def check_gpu_health():
    """Check if GPU has sufficient memory and is functioning"""
    print("🔍 Checking GPU health...")
    
    try:
        # Get GPU info
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu', 
             '--format=csv,noheader,nounits'],
            stderr=subprocess.STDOUT
        ).decode().strip()
        
        if not result:
            print("❌ No GPU detected")
            return False
            
        parts = result.split(', ')
        if len(parts) < 4:
            print(f"⚠️  Unexpected nvidia-smi output: {result}")
            return False
            
        gpu_name = parts[0]
        total_mem = float(parts[1])
        used_mem = float(parts[2])
        free_mem = float(parts[3])
        temp = parts[4] if len(parts) > 4 else "N/A"
        
        usage_percent = (used_mem / total_mem) * 100
        
        print(f"✅ GPU: {gpu_name}")
        print(f"   Total Memory: {total_mem:.0f} MB")
        print(f"   Used Memory:  {used_mem:.0f} MB ({usage_percent:.1f}%)")
        print(f"   Free Memory:  {free_mem:.0f} MB")
        print(f"   Temperature:  {temp}°C")
        
        # Check if we have enough free memory (recommend at least 4GB free)
        min_free_mb = 4096
        if free_mem < min_free_mb:
            print(f"⚠️  WARNING: Low GPU memory! Free: {free_mem:.0f} MB, Recommended: >{min_free_mb} MB")
            print("   Consider:")
            print("   - Closing other GPU processes")
            print("   - Reducing model size or batch size")
            print("   - Using a smaller max_model_len")
            return False
            
        # Check CUDA availability with PyTorch
        try:
            import torch
            if not torch.cuda.is_available():
                print("❌ PyTorch cannot access CUDA")
                return False
            
            cuda_version = torch.version.cuda
            print(f"✅ CUDA Runtime: {cuda_version}")
            print(f"✅ PyTorch CUDA available: {torch.cuda.is_available()}")
            
        except ImportError:
            print("⚠️  PyTorch not installed, skipping CUDA runtime check")
            
        return True
        
    except FileNotFoundError:
        print("❌ nvidia-smi not found. Is NVIDIA driver installed?")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ nvidia-smi failed: {e.output.decode()}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    if check_gpu_health():
        print("\n✅ GPU is ready for vLLM engine")
        sys.exit(0)
    else:
        print("\n❌ GPU health check failed")
        sys.exit(1)
