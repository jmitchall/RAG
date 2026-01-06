# Import the main libraries we need from vLLM
from vllm import LLM  # LLM = Large Language Model, SamplingParams = AI response settings


def load_mistral_quantized(gpu_memory_utilization: float = 0.85,
                           max_model_len: int = 16384) -> LLM:
    """
    Load Mistral 7B with 4-bit quantization (GPTQ) to fit RTX 5080 16GB.

    Demonstrates the difference between unquantized (13.5GB) and quantized (~4GB) models.

    Returns:
        LLM instance if successful, None otherwise
    """
    print("🔧 Loading Mistral 7B Instruct v0.2 (GPTQ 4-bit Quantized)...")
    print("   Model size: ~4GB VRAM (vs 13.5GB unquantized)")
    print("   Quantization: GPTQ 4-bit for RTX 5080 16GB compatibility")

    try:
        llm = LLM(
            model="TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,  # Reduced from 0.95 to fit available memory
            max_model_len=max_model_len,  # Full context window
            quantization="gptq",
            dtype="half",
            trust_remote_code=True
        )
        print("✅ SUCCESS! Mistral 7B GPTQ loaded on RTX 5080 16GB!")
        print("   Available KV cache: ~11GB")
        print("   Context window: 16,384 tokens")
        return llm

    except Exception as e:
        print(f"\n❌ UNEXPECTED FAILURE: {str(e)}")
        print("\n📋 Fallback options:")
        print("   • Try AWQ quantization: TheBloke/Mistral-7B-Instruct-v0.2-AWQ")
        print("   • Reduce max_model_len to 8192")
        print("   • Lower gpu_memory_utilization to 0.80")
        return None
