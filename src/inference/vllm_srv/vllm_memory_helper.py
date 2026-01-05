#!/usr/bin/env python3
"""
VLLM Memory Management Helper

A standalone utility to diagnose and resolve VLLM GPU memory issues.
"""

import subprocess
import logging
import gc
import time
import os
import sys

# Add the project root to Python path
sys.path.insert(0, '/home/jmitchall/quickagents')

def check_gpu_memory():
    """Check current GPU memory usage."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            memory_info = result.stdout.strip().split(', ')
            total, used, free = map(int, memory_info)
            return {
                'total_mb': total,
                'used_mb': used,
                'free_mb': free,
                'free_ratio': free / total,
                'used_ratio': used / total
            }
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError) as e:
        print(f"Could not retrieve GPU memory status: {e}")
    
    return {'total_mb': 0, 'used_mb': 0, 'free_mb': 0, 'free_ratio': 0.0, 'used_ratio': 0.0}

def force_memory_cleanup():
    """Force comprehensive memory cleanup."""
    print("🧹 Performing memory cleanup...")
    
    # Clear Python garbage collector
    gc.collect()
    
    # Clear PyTorch CUDA cache if available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("✅ Cleared PyTorch CUDA cache")
    except ImportError:
        print("PyTorch not available, skipping CUDA cache clear")
    
    # Force system sync
    try:
        subprocess.run(['sync'], check=False, timeout=5)
    except:
        pass
    
    print("✅ Memory cleanup completed")

def test_vllm_isolated():
    """Test VLLM in an isolated process."""
    print("🔄 Testing VLLM in isolated process...")
    
    try:
        from inference.vllm_srv.vllm_process_manager import VLLMProcessManager
        
        manager = VLLMProcessManager(
            download_dir="./models",
            gpu_memory_utilization=0.5,
            max_model_len=4096
        )
        
        success = manager.start_vllm_process()
        manager.cleanup()
        
        return success
        
    except Exception as e:
        print(f"Process isolation test failed: {e}")
        return False

def kill_gpu_processes():
    """Help user identify and kill GPU processes."""
    print("🔍 Checking for GPU processes...")
    
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            process_lines = [line for line in lines if 'python' in line.lower() or 'vllm' in line.lower()]
            
            if process_lines:
                print("Found GPU processes:")
                for line in process_lines:
                    print(f"  {line.strip()}")
                
                print("\n💡 To kill Python processes using GPU:")
                print("   pkill -f python")
                print("   # OR kill specific PIDs shown above")
            else:
                print("No obvious GPU processes found")
        
        print(f"\nFull nvidia-smi output:\n{result.stdout}")
        
    except Exception as e:
        print(f"Could not check GPU processes: {e}")

def main():
    """Main diagnostic and resolution function."""
    print("🔍 VLLM Memory Issue Resolver")
    print("=" * 50)
    
    # Check GPU memory
    memory_status = check_gpu_memory()
    if memory_status['total_mb'] == 0:
        print("❌ Could not detect GPU. Check nvidia-smi installation.")
        return
    
    print(f"📊 GPU Memory Status:")
    print(f"   - Total: {memory_status['total_mb']} MB")
    print(f"   - Used: {memory_status['used_mb']} MB ({memory_status['used_ratio']:.1%})")
    print(f"   - Free: {memory_status['free_mb']} MB ({memory_status['free_ratio']:.1%})")
    
    # Analyze memory situation
    if memory_status['free_ratio'] < 0.3:
        print("\n⚠️  Low GPU memory detected!")
        
        # Show current GPU processes
        kill_gpu_processes()
        
        # Attempt cleanup
        print("\n🧹 Attempting memory cleanup...")
        force_memory_cleanup()
        time.sleep(2)
        
        # Check again
        memory_status = check_gpu_memory()
        print(f"After cleanup: {memory_status['free_mb']} MB free")
    
    # Test VLLM with process isolation
    print("\n🔄 Testing VLLM with process isolation...")
    success = test_vllm_isolated()
    
    if success:
        print("✅ VLLM works correctly in isolation!")
        print("\n💡 PRIMARY SOLUTION:")
        print("   Restart your main Python process to clear multiprocessing conflicts:")
        print("   pkill -f python")
        print("   python src/lang_graph_langchain_tools_reflection_agent.py")
        
        print("\n💡 ALTERNATIVE SOLUTIONS:")
        print("1. Use process isolation in your code:")
        print("   use_process_isolation=True")
        print("\n2. Reduce memory usage:")
        print("   gpu_memory_utilization = 0.5")
        print("   max_model_len = 4096")
        
    else:
        print("❌ VLLM has deeper configuration issues")
        print("\n💡 TROUBLESHOOTING STEPS:")
        print("1. Kill all Python processes:")
        print("   pkill -f python")
        
        print("\n2. Restart your system (if needed)")
        
        print("\n3. Reduce memory parameters:")
        print("   gpu_memory_utilization = 0.4")
        print("   max_model_len = 4096")
        
        print("\n4. Check VLLM installation:")
        print("   pip install --upgrade vllm")

if __name__ == "__main__":
    main()