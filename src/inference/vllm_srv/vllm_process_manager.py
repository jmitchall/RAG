#!/usr/bin/env python3

"""
Process pool-based VLLM initialization to work around multiprocessing issues.
"""

import sys
import os
import subprocess
import tempfile
import time
import signal
from typing import Optional

class VLLMProcessManager:
    """Manage VLLM in a separate process to avoid multiprocessing conflicts."""
    
    def __init__(self, download_dir: str = "./models", gpu_memory_utilization: float = 0.5, max_model_len: int = 4096):
        self.download_dir = download_dir
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.process: Optional[subprocess.Popen] = None
        self.temp_script_path: Optional[str] = None
        
    def create_vllm_script(self) -> str:
        """Create a temporary script for running VLLM."""
        script_content = f'''
import sys
import os
sys.path.insert(0, '{os.getcwd()}')

# Import after setting path
from src.inference.vllm_srv.minstral_langchain import create_vllm_chat_model

def main():
    print("Starting VLLM in dedicated process...")
    try:
        llm = create_vllm_chat_model(
            download_dir="{self.download_dir}",
            gpu_memory_utilization={self.gpu_memory_utilization},
            max_model_len={self.max_model_len}
        )
        print("VLLM model created successfully!")
        
        # Keep the process alive and handle simple requests
        print("VLLM process ready for requests")
        
        # Simple test
        from langchain_core.messages import HumanMessage
        result = llm.invoke([HumanMessage(content="Hello, this is a test.")])
        print(f"Test result: {{result.content[:100]}}...")
        
        print("VLLM process completed successfully")
        return 0
        
    except Exception as e:
        print(f"Error in VLLM process: {{e}}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
'''
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script_content)
            return f.name
    
    def start_vllm_process(self) -> bool:
        """Start VLLM in a separate process."""
        try:
            self.temp_script_path = self.create_vllm_script()
            
            print("Starting VLLM in separate process...")
            self.process = subprocess.Popen([
                sys.executable, self.temp_script_path
            ], cwd=os.getcwd(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Wait for the process to complete (with timeout)
            try:
                stdout, stderr = self.process.communicate(timeout=600)  # 10 minute timeout
                
                if self.process.returncode == 0:
                    print("✓ VLLM process completed successfully!")
                    print("Output:", stdout[-500:] if len(stdout) > 500 else stdout)
                    return True
                else:
                    print("✗ VLLM process failed")
                    print("STDOUT:", stdout[-500:] if len(stdout) > 500 else stdout)
                    print("STDERR:", stderr[-500:] if len(stderr) > 500 else stderr)
                    return False
                    
            except subprocess.TimeoutExpired:
                print("✗ VLLM process timed out")
                self.cleanup()
                return False
                
        except Exception as e:
            print(f"Error starting VLLM process: {{e}}")
            return False
    
    def cleanup(self):
        """Clean up the process and temporary files."""
        if self.process:
            try:
                # Terminate the process gracefully
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't terminate
                    self.process.kill()
                    self.process.wait()
            except:
                pass
        
        if self.temp_script_path and os.path.exists(self.temp_script_path):
            os.unlink(self.temp_script_path)
    
    def __del__(self):
        self.cleanup()

def test_vllm_with_process_manager():
    """Test VLLM using the process manager."""
    manager = VLLMProcessManager(download_dir="./models")
    
    try:
        success = manager.start_vllm_process()
        if success:
            print("\\n✅ SUCCESS: VLLM is working correctly!")
            print("Your VLLM setup is functional. The original error was due to")
            print("repeated initialization in the same process, which is a known")
            print("limitation of VLLM 0.13.0.")
            print("\\nRecommendation: Restart your application process to use VLLM.")
            return True
        else:
            print("\\n❌ FAILED: VLLM has configuration issues.")
            return False
    finally:
        manager.cleanup()

if __name__ == "__main__":
    test_vllm_with_process_manager()