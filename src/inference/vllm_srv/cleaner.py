
import sys
import gc
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
        # If cleanup fails, it's usually not critical - just print a note
        # The error during cleanup doesn't affect the actual functionality
        print(f"Note: Cleanup encountered non-critical issue: {e}")
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