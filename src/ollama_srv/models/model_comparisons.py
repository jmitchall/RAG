#!/usr/bin/env python3
"""
LiteLLM Model Comparison: Local and Cloud Models
================================================

This script demonstrates different AI models you can use with LiteLLM:
1. Local models via Ollama (free, private, runs on your GPU)
2. Cloud models via various providers (OpenAI, Anthropic, etc.)
3. HuggingFace models via API

Key differences from vLLM approach:
- No need to manage GPU memory directly
- No quantization configuration (handled by provider)
- Unified interface across all model types
- Easier to switch between local and cloud
"""

import os
from litellm import completion
import litellm

# Optional: Enable debug mode for troubleshooting
# litellm.set_verbose = True


def test_model(model_name, description, provider_info="", example_prompt="Hello, how are you?"):
    """
    Test a specific AI model configuration.

    Args:
        model_name: Name of the model (with provider prefix for LiteLLM)
        description: Human-readable description
        provider_info: Additional provider information
        example_prompt: Test prompt to send

    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n{'=' * 70}")
    print(f"🔍 Testing: {model_name}")
    print(f"📝 Description: {description}")
    if provider_info:
        print(f"🔌 Provider: {provider_info}")
    print(f"{'=' * 70}")

    try:
        # Standard generation parameters
        messages = [
            {"role": "user", "content": example_prompt}
        ]

        # Generate response
        response = completion(
            model=model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=50,
            top_p=0.9
        )

        # Extract response text
        response_text = response.choices[0].message.content

        print(f"✅ SUCCESS!")
        print(f"🤖 AI Response: {response_text[:100]}...")
        return True

    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        if "gated" in str(e).lower() or "403" in str(e):
            print("💡 This model requires authentication")
        return False


def demonstrate_ollama_models():
    """
    Demonstrate local models via Ollama.

    Ollama must be running locally. Install from: https://ollama.ai
    """
    print("\n" + "=" * 70)
    print("🏠 LOCAL MODELS VIA OLLAMA")
    print("=" * 70)
    print("\n💡 These models run on your computer - no API costs, fully private")
    print("⚠️  Requires Ollama to be running: ollama serve")

    ollama_models = [
        ("ollama/phi3", "Phi-3 Mini 3.8B - Excellent small model", "Microsoft (local)"),
        ("ollama/mistral", "Mistral 7B - Industry standard quality", "Mistral AI (local)"),
        ("ollama/llama3.2:1b", "Llama 3.2 1B - Very fast, lightweight", "Meta (local)"),
        ("ollama/llama3.2:3b", "Llama 3.2 3B - Balanced performance", "Meta (local)"),
        ("ollama/tinyllama", "TinyLlama 1.1B - Fastest for testing", "TinyLlama (local)"),
    ]

    successful = []
    for model_name, description, provider in ollama_models:
        if test_model(model_name, description, provider):
            successful.append((model_name, description))

    return successful


def demonstrate_openai_models():
    """
    Demonstrate OpenAI models.

    Requires: OPENAI_API_KEY environment variable
    """
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  OPENAI_API_KEY not set - skipping OpenAI models")
        return []

    print("\n" + "=" * 70)
    print("☁️  OPENAI MODELS")
    print("=" * 70)

    openai_models = [
        ("gpt-3.5-turbo", "GPT-3.5 Turbo - Fast and cost-effective", "OpenAI (cloud)"),
        ("gpt-4o-mini", "GPT-4o Mini - Balanced quality and speed", "OpenAI (cloud)"),
        ("gpt-4", "GPT-4 - Highest quality", "OpenAI (cloud)"),
    ]

    successful = []
    for model_name, description, provider in openai_models:
        if test_model(model_name, description, provider):
            successful.append((model_name, description))

    return successful


def demonstrate_anthropic_models():
    """
    Demonstrate Anthropic Claude models.

    Requires: ANTHROPIC_API_KEY environment variable
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\n⚠️  ANTHROPIC_API_KEY not set - skipping Anthropic models")
        return []

    print("\n" + "=" * 70)
    print("☁️  ANTHROPIC CLAUDE MODELS")
    print("=" * 70)

    claude_models = [
        ("claude-3-haiku-20240307", "Claude 3 Haiku - Fast and efficient", "Anthropic (cloud)"),
        ("claude-3-sonnet-20240229", "Claude 3 Sonnet - Balanced performance", "Anthropic (cloud)"),
        ("claude-3-opus-20240229", "Claude 3 Opus - Highest quality", "Anthropic (cloud)"),
    ]

    successful = []
    for model_name, description, provider in claude_models:
        if test_model(model_name, description, provider):
            successful.append((model_name, description))

    return successful


def demonstrate_huggingface_models():
    """
    Demonstrate HuggingFace models via API.

    Requires: HUGGINGFACE_API_KEY environment variable
    Note: Some models are gated and require access approval
    """
    if not os.getenv("HUGGINGFACE_API_KEY"):
        print("\n⚠️  HUGGINGFACE_API_KEY not set - skipping HuggingFace models")
        return []

    print("\n" + "=" * 70)
    print("🤗 HUGGINGFACE MODELS (API)")
    print("=" * 70)

    hf_models = [
        ("huggingface/microsoft/Phi-3-mini-4k-instruct", "Phi-3 Mini - Microsoft's efficient model", "HuggingFace (cloud)"),
        ("huggingface/mistralai/Mistral-7B-Instruct-v0.2", "Mistral 7B Instruct", "HuggingFace (cloud)"),
    ]

    successful = []
    for model_name, description, provider in hf_models:
        if test_model(model_name, description, provider):
            successful.append((model_name, description))

    return successful


def demonstrate_all_models():
    """
    Demonstrate all available models across providers.
    """
    print("\n" + "=" * 70)
    print("🎯 LITELLM MODEL COMPARISON")
    print("=" * 70)
    print("\nTesting models across multiple providers...")

    all_successful = {}

    # Test local models (Ollama)
    try:
        ollama_successful = demonstrate_ollama_models()
        if ollama_successful:
            all_successful["Ollama (Local)"] = ollama_successful
    except Exception as e:
        print(f"\n⚠️  Ollama testing failed: {str(e)}")
        print("💡 Make sure Ollama is running: ollama serve")

    # Test OpenAI models
    try:
        openai_successful = demonstrate_openai_models()
        if openai_successful:
            all_successful["OpenAI (Cloud)"] = openai_successful
    except Exception as e:
        print(f"\n⚠️  OpenAI testing failed: {str(e)}")

    # Test Anthropic models
    try:
        anthropic_successful = demonstrate_anthropic_models()
        if anthropic_successful:
            all_successful["Anthropic (Cloud)"] = anthropic_successful
    except Exception as e:
        print(f"\n⚠️  Anthropic testing failed: {str(e)}")

    # Test HuggingFace models
    try:
        hf_successful = demonstrate_huggingface_models()
        if hf_successful:
            all_successful["HuggingFace (Cloud)"] = hf_successful
    except Exception as e:
        print(f"\n⚠️  HuggingFace testing failed: {str(e)}")

    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY OF WORKING MODELS")
    print("=" * 70)

    if all_successful:
        for provider, models in all_successful.items():
            print(f"\n✅ {provider}:")
            for model_name, description in models:
                print(f"   • {model_name}")
                print(f"     {description}")
    else:
        print("\n❌ No models tested successfully")
        print("\n💡 Setup instructions:")
        print("   • Local models: Install and run Ollama (https://ollama.ai)")
        print("   • OpenAI: Set OPENAI_API_KEY environment variable")
        print("   • Anthropic: Set ANTHROPIC_API_KEY environment variable")
        print("   • HuggingFace: Set HUGGINGFACE_API_KEY environment variable")

    return all_successful


def show_model_comparison():
    """
    Display comparison table of models across providers.
    """
    print("\n" + "=" * 70)
    print("📊 LITELLM MODEL COMPARISON TABLE")
    print("=" * 70)
    print()
    print("🏠 LOCAL MODELS (via Ollama):")
    print("| Model           | Size | Speed    | Quality | Context | Cost    |")
    print("|-----------------|------|----------|---------|---------|---------|")
    print("| TinyLlama       | 1.1B | ⚡⚡⚡⚡ | ⭐⭐    | 2K      | Free    |")
    print("| Llama 3.2 1B    | 1B   | ⚡⚡⚡⚡ | ⭐⭐    | 128K    | Free    |")
    print("| Llama 3.2 3B    | 3B   | ⚡⚡⚡   | ⭐⭐⭐  | 128K    | Free    |")
    print("| Phi-3 Mini      | 3.8B | ⚡⚡⚡   | ⭐⭐⭐  | 4K      | Free    |")
    print("| Mistral 7B      | 7B   | ⚡⚡     | ⭐⭐⭐⭐ | 32K     | Free    |")
    print()
    print("☁️  CLOUD MODELS:")
    print("| Model              | Provider   | Speed    | Quality | Context | Cost/1M |")
    print("|--------------------|------------|----------|---------|---------|---------|")
    print("| GPT-3.5 Turbo      | OpenAI     | ⚡⚡⚡   | ⭐⭐⭐  | 16K     | $0.50   |")
    print("| GPT-4o Mini        | OpenAI     | ⚡⚡⚡   | ⭐⭐⭐⭐ | 128K    | $0.15   |")
    print("| GPT-4              | OpenAI     | ⚡⚡     | ⭐⭐⭐⭐ | 8K      | $30.00  |")
    print("| Claude 3 Haiku     | Anthropic  | ⚡⚡⚡⚡ | ⭐⭐⭐  | 200K    | $0.25   |")
    print("| Claude 3 Sonnet    | Anthropic  | ⚡⚡⚡   | ⭐⭐⭐⭐ | 200K    | $3.00   |")
    print("| Claude 3 Opus      | Anthropic  | ⚡⚡     | ⭐⭐⭐⭐ | 200K    | $15.00  |")
    print()
    print("🎯 RECOMMENDATIONS:")
    print("   • Testing/Learning:     TinyLlama (local, fastest)")
    print("   • Local Development:    Phi-3 Mini (local, best quality/size)")
    print("   • Local Production:     Mistral 7B (local, industry standard)")
    print("   • Cloud Fast & Cheap:   GPT-3.5 Turbo or Claude Haiku")
    print("   • Cloud Best Quality:   GPT-4 or Claude Opus")
    print("   • Long Context:         Claude 3 (200K tokens)")
    print()
    print("💡 LITELLM ADVANTAGES:")
    print("   • Unified API across all providers")
    print("   • Easy switching between local and cloud")
    print("   • No GPU memory management needed")
    print("   • Automatic fallback support")
    print("   • Built-in cost tracking")
    print("=" * 70)


def main():
    """Main function to demonstrate LiteLLM capabilities."""
    show_model_comparison()

    print("\n🚀 Starting model tests...")
    print("⏳ This will test each available model sequentially...")

    input("\nPress Enter to continue with testing...")

    demonstrate_all_models()

    print("\n✨ Testing complete!")
    print("\n💡 Next steps:")
    print("   • For local models: Pull models with 'ollama pull <model>'")
    print("   • For cloud models: Set up API keys in environment variables")
    print("   • Try different models by changing the model string")
    print("   • See LiteLLM docs: https://docs.litellm.ai")


if __name__ == "__main__":
    main()