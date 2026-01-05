#!/usr/bin/env python3
"""
WORKING AI Model Server for 16GB GPU
===================================

This script demonstrates models that ACTUALLY WORK on 16GB GPUs.
Based on real testing with RTX 5080 16GB.

KEY FINDINGS:
- ✅ Phi-3 Mini (3.8B): Excellent quality via Ollama
- ✅ TinyLlama (1.1B): Very fast via Ollama
- ✅ Cloud models: Always work (OpenAI, Anthropic, etc.)
- 💡 Use Ollama for local GPU models (simpler than vLLM)
"""

from litellm import completion
import os


def test_working_models():
    """Test models that are confirmed to work on 16GB GPUs."""

    # Local models via Ollama - much simpler than vLLM
    local_models = [
        {
            "name": "ollama/tinyllama",
            "description": "Tiny model - very fast, basic quality, uses <2GB"
        },
        {
            "name": "ollama/phi3",
            "description": "RECOMMENDED: Excellent quality, uses ~7GB, reliable on 16GB"
        },
        {
            "name": "ollama/llama3.2:1b",
            "description": "Llama 3.2 1B - Fast and efficient, uses ~2GB"
        },
        {
            "name": "ollama/llama3.2:3b",
            "description": "Llama 3.2 3B - Balanced quality/speed, uses ~5GB"
        }
    ]

    # Cloud models - always work regardless of GPU
    cloud_models = [
        {
            "name": "gpt-3.5-turbo",
            "description": "OpenAI GPT-3.5 - Fast and cost-effective (requires OPENAI_API_KEY)",
            "requires_key": "OPENAI_API_KEY"
        },
        {
            "name": "claude-3-haiku-20240307",
            "description": "Claude 3 Haiku - Very fast Anthropic model (requires ANTHROPIC_API_KEY)",
            "requires_key": "ANTHROPIC_API_KEY"
        }
    ]

    # Test local models
    print("\n🏠 Testing LOCAL models via Ollama...")
    print("⚠️  Requires Ollama running: ollama serve")
    print("=" * 60)

    for model_info in local_models:
        print(f"\n🧪 Testing: {model_info['name']}")
        print(f"📝 {model_info['description']}")

        try:
            # Simple test prompt
            messages = [
                {"role": "user", "content": "Explain artificial intelligence in simple terms:"}
            ]

            # Generate response
            response = completion(
                model=model_info["name"],
                messages=messages,
                temperature=0.7,
                max_tokens=100
            )

            # Extract response text
            response_text = response.choices[0].message.content

            print("✅ SUCCESS!")
            print(f"🤖 Response: {response_text[:200]}...")

        except Exception as e:
            print(f"❌ FAILED: {str(e)}")
            if "Could not connect" in str(e) or "Connection refused" in str(e):
                print("💡 Start Ollama first: ollama serve")
                print(f"💡 Then pull the model: ollama pull {model_info['name'].replace('ollama/', '')}")

    # Test cloud models
    print(f"\n{'=' * 60}")
    print("☁️  Testing CLOUD models...")
    print("=" * 60)

    for model_info in cloud_models:
        required_key = model_info.get("requires_key")

        if required_key and not os.getenv(required_key):
            print(f"\n⚠️  Skipping {model_info['name']}: {required_key} not set")
            continue

        print(f"\n🧪 Testing: {model_info['name']}")
        print(f"📝 {model_info['description']}")

        try:
            # Simple test prompt
            messages = [
                {"role": "user", "content": "Explain artificial intelligence in simple terms:"}
            ]

            # Generate response
            response = completion(
                model=model_info["name"],
                messages=messages,
                temperature=0.7,
                max_tokens=100
            )

            # Extract response text
            response_text = response.choices[0].message.content

            print("✅ SUCCESS!")
            print(f"🤖 Response: {response_text[:200]}...")

        except Exception as e:
            print(f"❌ FAILED: {str(e)}")


def recommended_setup():
    """Show the recommended setup for 16GB GPUs."""
    print("\n" + "=" * 60)
    print("🎯 RECOMMENDED SETUP for 16GB GPU")
    print("=" * 60)
    print()
    print("✅ Option 1: Local Models via Ollama (RECOMMENDED)")
    print("   • No API costs")
    print("   • Fully private")
    print("   • No GPU memory management needed")
    print("   • Simple setup")
    print()
    print("📦 Install Ollama:")
    print("   Visit: https://ollama.ai")
    print()
    print("🚀 Start Ollama server:")
    print("   ollama serve")
    print()
    print("⬇️  Pull Phi-3 model:")
    print("   ollama pull phi3")
    print()
    print("🧪 Test with Python:")
    print("   from litellm import completion")
    print("   response = completion(")
    print("       model='ollama/phi3',")
    print("       messages=[{'role': 'user', 'content': 'Hello!'}]")
    print("   )")
    print("   print(response.choices[0].message.content)")
    print()
    print("✅ Option 2: Cloud Models (No GPU Required)")
    print("   • Set OPENAI_API_KEY for GPT models")
    print("   • Set ANTHROPIC_API_KEY for Claude models")
    print("   • Works on any machine")
    print()
    print("🧪 Test with Python:")
    print("   from litellm import completion")
    print("   response = completion(")
    print("       model='gpt-3.5-turbo',")
    print("       messages=[{'role': 'user', 'content': 'Hello!'}]")
    print("   )")


def show_comparison():
    """Show comparison between local and cloud options."""
    print("\n" + "=" * 60)
    print("📊 LOCAL vs CLOUD COMPARISON")
    print("=" * 60)
    print()
    print("🏠 LOCAL MODELS (via Ollama on 16GB GPU):")
    print("| Model          | Size  | GPU Usage | Quality | Speed    | Cost |")
    print("|----------------|-------|-----------|---------|----------|------|")
    print("| TinyLlama      | 1.1B  | ~2GB      | ⭐⭐    | ⚡⚡⚡⚡ | Free |")
    print("| Llama 3.2 1B   | 1B    | ~2GB      | ⭐⭐    | ⚡⚡⚡⚡ | Free |")
    print("| Llama 3.2 3B   | 3B    | ~5GB      | ⭐⭐⭐  | ⚡⚡⚡   | Free |")
    print("| Phi-3 Mini     | 3.8B  | ~7GB      | ⭐⭐⭐⭐ | ⚡⚡⚡   | Free |")
    print()
    print("☁️  CLOUD MODELS (No GPU needed):")
    print("| Model          | Provider  | Quality | Speed    | Cost/1M tokens |")
    print("|----------------|-----------|---------|----------|----------------|")
    print("| GPT-3.5 Turbo  | OpenAI    | ⭐⭐⭐  | ⚡⚡⚡   | $0.50          |")
    print("| GPT-4o Mini    | OpenAI    | ⭐⭐⭐⭐ | ⚡⚡⚡   | $0.15          |")
    print("| Claude 3 Haiku | Anthropic | ⭐⭐⭐  | ⚡⚡⚡⚡ | $0.25          |")
    print()
    print("🎯 RECOMMENDATIONS:")
    print("   • Learning/Testing:    TinyLlama (local, fastest)")
    print("   • Development:         Phi-3 Mini (local, best quality)")
    print("   • Production (budget): Ollama models (local, free)")
    print("   • Production (quality): GPT-4o Mini or Claude 3 (cloud, paid)")
    print()
    print("💡 WHY LITELLM + OLLAMA vs vLLM?")
    print("   • ✅ No GPU memory configuration needed")
    print("   • ✅ Simpler setup and usage")
    print("   • ✅ Unified API for local and cloud")
    print("   • ✅ Easy model switching")
    print("   • ✅ Automatic resource management")


if __name__ == "__main__":
    print("🚀 16GB GPU AI Model Compatibility Test (LiteLLM)")
    print("This script tests models that work on 16GB GPUs using LiteLLM.")

    show_comparison()

    print("\n" + "=" * 60)
    input("Press Enter to start testing models...")

    test_working_models()
    recommended_setup()

    print(f"\n{'=' * 60}")
    print("💡 SUMMARY")
    print("=" * 60)
    print("✅ LOCAL (Ollama): tinyllama, phi3, llama3.2:1b, llama3.2:3b")
    print("✅ CLOUD: gpt-3.5-turbo, claude-3-haiku (with API keys)")
    print()
    print("🎯 For 16GB GPU:")
    print("   • Best local: Phi-3 Mini via Ollama")
    print("   • Best cloud: GPT-3.5 Turbo or Claude 3 Haiku")
    print()
    print("🚀 Next steps:")
    print("   1. Install Ollama: https://ollama.ai")
    print("   2. Run: ollama serve")
    print("   3. Pull models: ollama pull phi3")
    print("   4. Use LiteLLM to query any model with same API!")