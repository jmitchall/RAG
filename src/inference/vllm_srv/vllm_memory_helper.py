#!/usr/bin/env python3
"""
VLLM Memory Management Helper

A standalone utility to diagnose and resolve VLLM GPU memory issues.
"""

import gc
from refection_logger import logger
import subprocess
import sys
import time

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
        logger.info(f"Could not retrieve GPU memory status: {e}")

    return {'total_mb': 0, 'used_mb': 0, 'free_mb': 0, 'free_ratio': 0.0, 'used_ratio': 0.0}


def force_memory_cleanup():
    """Force comprehensive memory cleanup."""
    logger.info("🧹 Performing memory cleanup...")

    # Clear Python garbage collector
    gc.collect()

    # Clear PyTorch CUDA cache if available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("✅ Cleared PyTorch CUDA cache")
    except ImportError:
        logger.info("PyTorch not available, skipping CUDA cache clear")

    # Force system sync
    try:
        subprocess.run(['sync'], check=False, timeout=5)
    except:
        pass

    logger.info("✅ Memory cleanup completed")


def test_vllm_isolated():
    """Test VLLM in an isolated process."""
    logger.info("🔄 Testing VLLM in isolated process...")

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
        logger.info(f"Process isolation test failed: {e}")
        return False


def kill_gpu_processes():
    """Help user identify and kill GPU processes."""
    logger.info("🔍 Checking for GPU processes...")

    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            process_lines = [line for line in lines if 'python' in line.lower() or 'vllm' in line.lower()]

            if process_lines:
                logger.info("Found GPU processes:")
                for line in process_lines:
                    logger.info(f"  {line.strip()}")

                logger.info("\n💡 To kill Python processes using GPU:")
                logger.info("   pkill -f python")
                logger.info("   # OR kill specific PIDs shown above")
            else:
                logger.info("No obvious GPU processes found")

        logger.info(f"\nFull nvidia-smi output:\n{result.stdout}")

    except Exception as e:
        logger.info(f"Could not check GPU processes: {e}")


def main():
    """Main diagnostic and resolution function."""
    logger.info("🔍 VLLM Memory Issue Resolver")
    logger.info("=" * 50)

    # Check GPU memory
    memory_status = check_gpu_memory()
    if memory_status['total_mb'] == 0:
        logger.info("❌ Could not detect GPU. Check nvidia-smi installation.")
        return

    logger.info(f"📊 GPU Memory Status:")
    logger.info(f"   - Total: {memory_status['total_mb']} MB")
    logger.info(f"   - Used: {memory_status['used_mb']} MB ({memory_status['used_ratio']:.1%})")
    logger.info(f"   - Free: {memory_status['free_mb']} MB ({memory_status['free_ratio']:.1%})")

    # Analyze memory situation
    if memory_status['free_ratio'] < 0.3:
        logger.info("\n⚠️  Low GPU memory detected!")

        # Show current GPU processes
        kill_gpu_processes()

        # Attempt cleanup
        logger.info("\n🧹 Attempting memory cleanup...")
        force_memory_cleanup()
        time.sleep(2)

        # Check again
        memory_status = check_gpu_memory()
        logger.info(f"After cleanup: {memory_status['free_mb']} MB free")

    # Test VLLM with process isolation
    logger.info("\n🔄 Testing VLLM with process isolation...")
    success = test_vllm_isolated()

    if success:
        logger.info("✅ VLLM works correctly in isolation!")
        logger.info("\n💡 PRIMARY SOLUTION:")
        logger.info("   Restart your main Python process to clear multiprocessing conflicts:")
        logger.info("   pkill -f python")
        logger.info("   python src/lang_graph_langchain_tools_reflection_agent.py")

        logger.info("\n💡 ALTERNATIVE SOLUTIONS:")
        logger.info("1. Use process isolation in your code:")
        logger.info("   use_process_isolation=True")
        logger.info("\n2. Reduce memory usage:")
        logger.info("   gpu_memory_utilization = 0.5")
        logger.info("   max_model_len = 4096")

    else:
        logger.info("❌ VLLM has deeper configuration issues")
        logger.info("\n💡 TROUBLESHOOTING STEPS:")
        logger.info("1. Kill all Python processes:")
        logger.info("   pkill -f python")

        logger.info("\n2. Restart your system (if needed)")

        logger.info("\n3. Reduce memory parameters:")
        logger.info("   gpu_memory_utilization = 0.4")
        logger.info("   max_model_len = 4096")

        logger.info("\n4. Check VLLM installation:")
        logger.info("   pip install --upgrade vllm")


if __name__ == "__main__":
    main()
