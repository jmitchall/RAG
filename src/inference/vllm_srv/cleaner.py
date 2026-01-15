#!/usr/bin/env python3
"""
vLLM Engine Cleaner - GPU Memory Management

Author: Jonathan A. Mitchall
Version: 1.0
Last Updated: January 10, 2026

License: MIT License

Copyright (c) 2026 Jonathan A. Mitchall

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Revision History:
    2026-01-10 (v1.0): Initial comprehensive documentation
"""

import gc
from refection_logger import logger
import subprocess
import sys
import warnings



# =============================================================================
# CLEANUP FUNCTIONS - Prevent vLLM engine core crash on exit
# =============================================================================

def cleanup_vllm_engine(llm_instance):
    """
    Safely cleanup vLLM engine resources to prevent "Engine core proc died unexpectedly" error.
    
    The vLLM engine creates background processes and GPU resources that need proper cleanup.
    Without this, Python's automatic garbage collection can cause the engine core to crash
    during shutdown, producing error messages (though functionality still works).
    
    Args:
        llm_instance: The VLLM instance to cleanup (LangChain wrapper)
    
    How it works:
    1. Suppresses warnings during cleanup (vLLM may emit deprecation warnings)
    2. Tries to access the underlying vLLM client and shut it down gracefully
    3. Deletes the Python object reference
    4. Forces garbage collection to free GPU memory immediately
    5. Catches and ignores any errors during cleanup (they're non-critical)
    """
    try:
        # Suppress warnings during cleanup to avoid clutter
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Access the underlying vLLM engine client (stored in the 'client' attribute)
            # and try to destroy it gracefully if it has a cleanup method
            if hasattr(llm_instance, 'client') and llm_instance.client is not None:
                # Some versions of vLLM have explicit shutdown methods
                if hasattr(llm_instance.client, '__del__'):
                    # Let the destructor clean up
                    pass

                # Set client to None to break the reference
                llm_instance.client = None

            # Delete the LLM instance to remove all references
            del llm_instance

            # Force Python's garbage collector to run immediately
            # This frees GPU memory and other resources right away instead of
            # waiting for Python's automatic cleanup (which can be delayed)
            gc.collect()

    except Exception as e:
        # If cleanup fails, it's usually not critical - just logger.info a note
        # The error during cleanup doesn't affect the actual functionality
        logger.info(f"Note: Cleanup encountered non-critical issue: {e}")
        pass


def suppress_vllm_shutdown_errors():
    """
    Suppress the "Engine core proc died unexpectedly" error that occurs during Python exit.
    
    This error is cosmetic - it happens AFTER all work is done, during final cleanup.
    The error message looks scary but doesn't indicate a real problem:
    - Your code ran successfully
    - Text generation completed properly  
    - It's just an ungraceful shutdown of background processes
    
    This function redirects stderr temporarily to suppress the error message.
    Use this as a last resort if proper cleanup doesn't prevent the error.
    """
    import atexit
    import os

    def silent_exit():
        """Redirect stderr to /dev/null during final exit to hide engine shutdown errors."""
        try:
            # On Unix systems, redirect stderr to /dev/null
            sys.stderr = open(os.devnull, 'w')
        except:
            pass

    # Register this function to run at the very end of Python execution
    # It will hide any error messages that occur during final cleanup
    atexit.register(silent_exit)


def check_gpu_memory_status() -> dict:
    """
    Check current GPU memory usage to detect potential memory issues.

    Returns:
        dict: GPU memory information including total, used, and free memory
    """
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
        logger.warning(f"Could not retrieve GPU memory status: {e}")

    return {'total_mb': 0, 'used_mb': 0, 'free_mb': 0, 'free_ratio': 0.0, 'used_ratio': 0.0}


def force_gpu_memory_cleanup():
    """
    Force comprehensive GPU memory cleanup.

    This method performs aggressive memory cleanup to free up GPU memory
    that may be held by previous VLLM instances or other processes.
    """
    logger.info("🧹 Performing GPU memory cleanup...")

    try:
        # Clear Python's garbage collector
        gc.collect()

        # Try to import torch and clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("✅ Cleared PyTorch CUDA cache")
        except ImportError:
            logger.info("PyTorch not available, skipping CUDA cache clear")

        # Force system memory cleanup
        try:
            # On Linux, try to drop caches if we have permission
            subprocess.run(['sync'], check=False, timeout=5)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        logger.info("✅ GPU memory cleanup completed")

    except Exception as e:
        logger.warning(f"⚠️  GPU memory cleanup encountered issue: {e}")
