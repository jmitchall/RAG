#!/usr/bin/env python3
"""
RTX 5080 16GB GPU Reality Check: Mistral 7B Analysis
===================================================

⚠️ IMPORTANT FINDINGS FROM TESTING:
- Mistral 7B model weights: 13.5GB 
- Available KV cache memory: -4.2GB (NEGATIVE!)
- RTX 5080 16GB: INSUFFICIENT for Mistral 7B

This script demonstrates:
1. Why Mistral 7B fails on 16GB GPUs
2. Working alternatives (Phi-3 Mini)
3. Hardware requirements for Mistral 7B
4. Practical solutions for 16GB GPU users
"""

import os
from vllm import LLM, SamplingParams


def demonstrate_mistral_quantized():
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
            gpu_memory_utilization=0.85,  # Reduced from 0.95 to fit available memory
            max_model_len=16384,  # Full context window
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


def load_llama_3_2_1b():
    """
    Load Llama 3.2 1B Instruct - extremely efficient for RTX 5080.
    
    ⚠️ NOTE: Llama models are GATED and require HuggingFace authentication.
    You need to:
    1. Request access at https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
    2. Login: uv run huggingface-cli login
    
    Returns:
        LLM instance or None
    """
    print("\n🦙 Loading Llama 3.2 1B Instruct...")
    print("   ⚠️ This is a GATED model - requires HuggingFace access")
    print("   Visit: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct")
    
    try:
        llm = LLM(
            model="meta-llama/Llama-3.2-1B-Instruct",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.85,
            max_model_len=32768,
            dtype="bfloat16",
            trust_remote_code=True
        )
        print("✅ Llama 3.2 1B loaded successfully!")
        return llm
    except Exception as e:
        if "gated repo" in str(e) or "403" in str(e):
            print("❌ Access denied - model is gated")
            print("💡 To use this model:")
            print("   1. Visit https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct")
            print("   2. Click 'Request Access' and accept terms")
            print("   3. Run: huggingface-cli login")
            print("   4. Paste your HF token")
            return None
        else:
            raise e


def load_llama_3_2_3b():
    """
    Load Llama 3.2 3B Instruct - good balance of size and quality.
    
    ⚠️ NOTE: Llama models are GATED and require HuggingFace authentication.
    uv run huggingface-cli login
    Returns:
        LLM instance or None
    """
    print("\n🦙 Loading Llama 3.2 3B Instruct...")
    print("   ⚠️ This is a GATED model - requires HuggingFace access")
    
    try:
        llm = LLM(
            model="meta-llama/Llama-3.2-3B-Instruct",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.85,
            max_model_len=16384,
            dtype="bfloat16",
            trust_remote_code=True
        )
        print("✅ Llama 3.2 3B loaded successfully!")
        return llm
    except Exception as e:
        if "gated repo" in str(e) or "403" in str(e):
            print("❌ Access denied - model is gated")
            print("💡 Use ungated alternatives instead (see below)")
            return None
        else:
            raise e


def load_mistral_7b_gptq():
    """
    Load Mistral 7B GPTQ quantized (4-bit) - fits on RTX 5080!
    
    Returns:
        LLM instance
    """
    print("\n🔧 Loading Mistral 7B GPTQ (4-bit quantized)...")
    print("   Model size: ~4GB VRAM (vs 13.5GB unquantized)")
    print("   Expected available cache: ~11GB")
    
    llm = LLM(
        model="TheBloke/Mistral-7B-Instruct-v0.1-GPTQ",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=16384,
        quantization="gptq",
        dtype="half",
        trust_remote_code=True
    )
    print("✅ Mistral 7B GPTQ loaded successfully!")
    return llm


def load_mistral_7b_awq():
    """
    Load Mistral 7B AWQ quantized (4-bit) - alternative quantization.
    
    Returns:
        LLM instance
    """
    print("\n🔧 Loading Mistral 7B AWQ (4-bit quantized)...")
    print("   Model size: ~4GB VRAM")
    print("   Expected available cache: ~11GB")
    
    llm = LLM(
        model="TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=16384,
        quantization="awq",
        dtype="half",
        trust_remote_code=True
    )
    print("✅ Mistral 7B AWQ loaded successfully!")
    return llm


def load_llava_mistral_7b():
    """
    Load LLaVA v1.6 Mistral 7B - multimodal vision model.
    Note: Requires aggressive memory settings for RTX 5080.
    
    Returns:
        LLM instance
    """
    print("\n🖼️ Loading LLaVA v1.6 Mistral 7B...")
    print("   ⚠️ Vision models require more VRAM")
    print("   Attempting with aggressive memory settings...")
    
    try:
        llm = LLM(
            model="llava-hf/llava-v1.6-mistral-7b-hf",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.75,  # Reduced from 0.85
            max_model_len=2048,  # Reduced from 4096 to minimize KV cache
            dtype="half",  # Changed from bfloat16 to half for memory savings
            trust_remote_code=True,
            max_num_seqs=1,  # Process one at a time
            enforce_eager=True  # Disable CUDA graphs to save memory
        )
        print("✅ LLaVA Mistral 7B loaded!")
        return llm
    except Exception as e:
        print(f"❌ Failed to load: {str(e)}")
        print("💡 Recommendations:")
        print("   • Vision models are challenging on 16GB GPUs")
        print("   • Try smaller vision models like llava-hf/llava-1.5-7b-hf")
        print("   • Consider using API-based vision models instead")
        return None


def load_openhermes_mistral_7b():
    """
    Load OpenHermes 2.5 Mistral 7B fine-tune (requires quantization).
    
    Returns:
        LLM instance
    """
    print("\n🧙 Loading OpenHermes 2.5 Mistral 7B (GPTQ)...")
    print("   Fine-tuned for instruction following")
    
    llm = LLM(
        model="TheBloke/OpenHermes-2.5-Mistral-7B-GPTQ",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=16384,
        quantization="gptq",
        dtype="half",
        trust_remote_code=True
    )
    print("✅ OpenHermes 2.5 loaded successfully!")
    return llm


def load_zephyr_7b_beta():
    """
    Load Zephyr 7B Beta - Mistral 7B fine-tune (requires quantization).
    
    Returns:
        LLM instance
    """
    print("\n💨 Loading Zephyr 7B Beta (GPTQ)...")
    print("   Aligned Mistral 7B variant")
    
    llm = LLM(
        model="TheBloke/zephyr-7B-beta-GPTQ",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=16384,
        quantization="gptq",
        dtype="half",
        trust_remote_code=True
    )
    print("✅ Zephyr 7B Beta loaded successfully!")
    return llm


def load_neural_chat_7b():
    """
    Load Neural Chat 7B - Intel's Mistral 7B fine-tune (requires quantization).
    
    Returns:
        LLM instance
    """
    print("\n💬 Loading Neural Chat 7B (GPTQ)...")
    print("   Optimized for conversational tasks")
    
    llm = LLM(
        model="TheBloke/neural-chat-7B-v3-1-GPTQ",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=16384,
        quantization="gptq",
        dtype="half",
        trust_remote_code=True
    )
    print("✅ Neural Chat 7B loaded successfully!")
    return llm


def load_starling_7b():
    """
    Load Starling LM 7B Alpha - RLAIF trained Mistral variant (requires quantization).
    
    Returns:
        LLM instance
    """
    print("\n⭐ Loading Starling LM 7B Alpha (AWQ)...")
    print("   RLAIF-trained for helpfulness")
    
    llm = LLM(
        model="TheBloke/Starling-LM-7B-alpha-AWQ",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=8192,
        quantization="awq",
        dtype="half",
        trust_remote_code=True
    )
    print("✅ Starling LM 7B loaded successfully!")
    return llm

def load_opt_125m():
    """
    Load OPT-125M - Facebook's tiny model for testing and development.
    
    Perfect for:
    - Testing vLLM setup
    - Rapid prototyping
    - Low-resource environments
    - Learning/experimentation
    
    Returns:
        LLM instance
    """
    print("\n🧪 Loading OPT-125M (Facebook)...")
    print("   Model size: ~500MB VRAM")
    print("   Use case: Testing, prototyping, learning")
    print("   Expected available cache: ~15GB")
    
    llm = LLM(
        model="facebook/opt-125m",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        dtype="float16",
        trust_remote_code=True
    )
    print("✅ OPT-125M loaded successfully!")
    print("   💡 This tiny model loads in <10 seconds - perfect for testing!")
    return llm


def load_phi_3_mini():
    """
    Load Phi-3 Mini 3.8B - Microsoft's efficient instruction model.
    
    One of the best small models available:
    - Competitive with 7B models in many tasks
    - Fits comfortably on 16GB VRAM
    - Strong reasoning capabilities
    - 4K context window
    
    Returns:
        LLM instance
    """
    print("\n🔷 Loading Phi-3 Mini 3.8B Instruct...")
    print("   Model size: ~7GB VRAM")
    print("   Microsoft's efficient model (UNGATED!)")
    print("   Expected available cache: ~8GB")  
    llm = LLM(
        model="microsoft/Phi-3-mini-4k-instruct",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        dtype="bfloat16",
        trust_remote_code=True
        # model_path="./models/phi-3-mini-4k-instruct"  # Optional: specify local path to save/load model
    )
    print("✅ Phi-3 Mini loaded successfully!")
    print("   💡 Excellent performance-to-size ratio!")
    return llm


def load_phi_3_mini_quantized():
    """
    Load Phi-3 Mini with GPTQ quantization for extra memory headroom.
    
    Returns:
        LLM instance
    """
    print("\n🔷 Loading Phi-3 Mini 3.8B (GPTQ 4-bit)...")
    print("   Model size: ~3GB VRAM (quantized)")
    print("   Expected available cache: ~12GB")
    
    try:
        llm = LLM(
            model="TheBloke/Phi-3-Mini-4K-Instruct-GPTQ",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.85,
            max_model_len=4096,
            quantization="gptq",
            dtype="half",
            trust_remote_code=True
        )
        print("✅ Phi-3 Mini GPTQ loaded successfully!")
        return llm
    except Exception as e:
        print(f"⚠️ GPTQ version not available: {str(e)}")
        print("💡 Using unquantized version instead...")
        return load_phi_3_mini()


def demonstrate_all_models():
    """
    Demonstrate all RTX 5080 compatible models.
    """
    print("\n" + "=" * 70)
    print("🎯 RTX 5080 16GB: COMPATIBLE MODELS SHOWCASE")
    print("=" * 70)
    
    models_to_test = [
       # 🧪 TINY MODELS (For testing)
       ("OPT-125M (Testing)", load_opt_125m),
       
       # 🔷 SMALL EFFICIENT MODELS (3-4B)
       ("Phi-3 Mini 3.8B", load_phi_3_mini),
        ("Phi-3 Mini 3.8B GPTQ", load_phi_3_mini_quantized),
       
       # ⚠️ GATED MODELS (Require HF auth)
       # Permission Pending ("Llama 3.2 1B", load_llama_3_2_1b),
       # Permission Pending ("Llama 3.2 3B", load_llama_3_2_3b),
       # ✅ QUANTIZED 7B MODELS (All work!)
       ("Mistral 7B", demonstrate_mistral_quantized),
       ("Mistral 7B GPTQ", load_mistral_7b_gptq),
       ("Mistral 7B AWQ", load_mistral_7b_awq),
       ("OpenHermes 2.5", load_openhermes_mistral_7b),
       ("Zephyr 7B Beta", load_zephyr_7b_beta),
       ("Neural Chat 7B", load_neural_chat_7b),
       ("Starling LM 7B", load_starling_7b),


    ]
    import torch; 
    print("\n📊 Testing models sequentially (each loads and unloads)...\n")
    
    for model_name, loader_func in models_to_test:
        print(f"\n{'=' * 70}")
        torch.cuda.empty_cache()
        try:
            llm = loader_func()
            if llm:
                # Quick test
                sampling_params = SamplingParams(temperature=0.7, max_tokens=50)
                output = llm.generate(["Hello, how are you?"], sampling_params)
                print(f"📝 Test output: {output[0].outputs[0].text[:100]}...")
                del llm  # Free memory for next model
                print(f"✅ {model_name} tested successfully!")
                answer = input(" Start Next ?")
                if answer.lower().startswith("y"):
                    continue
                else:
                    break
        except Exception as e:
            print(f"❌ {model_name} failed: {str(e)}")
        print()


def main():
    """Main function to demonstrate RTX 5080 limitations and solutions."""
    print("\n🔍 Exploring alternatives...")
    # Load and demonstrate compatible models
    demonstrate_all_models()
    
def show_model_comparison():
    """
    Display comparison table of all available models.
    """
    print("\n" + "=" * 70)
    print("📊 RTX 5080 16GB: MODEL COMPARISON")
    print("=" * 70)
    print()
    print("| Model               | Size | VRAM  | Speed    | Quality | Context |")
    print("|---------------------|------|-------|----------|---------|---------|")
    print("| OPT-125M            | 125M | 0.5GB | ⚡⚡⚡⚡ | ⭐      | 2K      |")
    print("| Phi-3 Mini          | 3.8B | 7GB   | ⚡⚡⚡   | ⭐⭐⭐  | 4K      |")
    print("| Llama 3.2 1B        | 1B   | 3GB   | ⚡⚡⚡⚡ | ⭐⭐    | 32K     |")
    print("| Llama 3.2 3B        | 3B   | 6GB   | ⚡⚡⚡   | ⭐⭐⭐  | 16K     |")
    print("| Mistral 7B GPTQ     | 7B   | 4GB   | ⚡⚡     | ⭐⭐⭐⭐ | 16K     |")
    print("| OpenHermes 2.5      | 7B   | 4GB   | ⚡⚡     | ⭐⭐⭐⭐ | 16K     |")
    print("| Zephyr 7B           | 7B   | 4GB   | ⚡⚡     | ⭐⭐⭐⭐ | 16K     |")
    print("| Starling LM 7B      | 7B   | 4GB   | ⚡⚡     | ⭐⭐⭐⭐ | 8K      |")
    print()
    print("🎯 Recommendations:")
    print("   • Testing/Learning:  OPT-125M (fastest to load)")
    print("   • Small & Fast:      Phi-3 Mini (best 3B model)")
    print("   • Balanced:          Llama 3.2 3B (good quality, ungated soon)")
    print("   • Best Quality:      Mistral 7B GPTQ (industry standard)")
    print("   • Instruction:       OpenHermes 2.5 (tuned for tasks)")
    print("   • Chat:              Zephyr 7B (conversational)")
    print("   • Helpful:           Starling LM 7B (RLAIF trained)")
    print("=" * 70)


if __name__ == "__main__":
    show_model_comparison()
    main()
