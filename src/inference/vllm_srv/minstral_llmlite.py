# Import the main libraries we need from vLLM
from vllm import LLM  # LLM = Large Language Model, SamplingParams = AI response settings
from refection_logger import logger

def load_mistral_quantized(gpu_memory_utilization: float = 0.85,
                           max_model_len: int = 16384) -> LLM:
    """
    Load Mistral 7B with 4-bit quantization (GPTQ) to fit RTX 5080 16GB.

    Demonstrates the difference between unquantized (13.5GB) and quantized (~4GB) models.

    Returns:
        LLM instance if successful, None otherwise
    """
    logger.info("🔧 Loading Mistral 7B Instruct v0.2 (GPTQ 4-bit Quantized)...")
    logger.info("   Model size: ~4GB VRAM (vs 13.5GB unquantized)")
    logger.info("   Quantization: GPTQ 4-bit for RTX 5080 16GB compatibility")

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
        logger.info("✅ SUCCESS! Mistral 7B GPTQ loaded on RTX 5080 16GB!")
        logger.info("   Available KV cache: ~11GB")
        logger.info("   Context window: 16,384 tokens")
        return llm

    except Exception as e:
        logger.info(f"\n❌ UNEXPECTED FAILURE: {str(e)}")
        logger.info("\n📋 Fallback options:")
        logger.info("   • Try AWQ quantization: TheBloke/Mistral-7B-Instruct-v0.2-AWQ")
        logger.info("   • Reduce max_model_len to 8192")
        logger.info("   • Lower gpu_memory_utilization to 0.80")
        return None
