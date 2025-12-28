#!/usr/bin/env python3
"""
Model Comparison Demo
====================

This script demonstrates different AI models you can use with LiteLLM.
Run this to see which models work with your hardware setup.
Supports local (Ollama) and cloud-based models.

Usage:
    python model_comparison.py                    # Test cloud models
    python model_comparison.py --local           # Test local Ollama models
    python model_comparison.py --provider openai # Test specific provider
"""

import argparse
import os
import sys
from pathlib import Path
from litellm import completion, get_supported_openai_params
import litellm

# Enable LiteLLM debug mode for troubleshooting (optional)
# litellm.set_verbose = True


def test_model(model_name, config, description, provider_info=""):
    """
    Test a specific AI model configuration.

    Args:
        model_name: Name of the model to test (with provider prefix)
        config: Dictionary of generation parameters
        description: Human-readable description of the model
        provider_info: Additional provider information
    """
    print(f"\n{'=' * 60}")
    print(f"🔍 Testing: {model_name}")
    print(f"📝 Description: {description}")
    if provider_info:
        print(f"🔌 Provider: {provider_info}")
    print(f"⚙️  Configuration: {config}")
    print(f"{'=' * 60}")

    try:
        # Simple test prompt
        messages = [
            {"role": "user", "content": "Hello! Please introduce yourself briefly."}
        ]

        # Generate response
        response = completion(
            model=model_name,
            messages=messages,
            **config
        )

        # Extract response text
        response_text = response.choices[0].message.content

        print(f"✅ SUCCESS!")
        print(f"🤖 AI Response: {response_text[:200]}...")
        return True

    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False


def check_ollama_available():
    """Check if Ollama is running and available."""
    try:
        # Try to list models to verify Ollama is running
        response = completion(
            model="ollama/phi3",  # Use a common model
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1
        )
        return True
    except Exception as e:
        return False


def main():
    """Main function to test different model configurations."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test AI models with LiteLLM")
    parser.add_argument("--local", action="store_true",
                        help="Test local Ollama models only")
    parser.add_argument("--provider", type=str,
                        help="Test specific provider (openai, anthropic, ollama, etc.)")
    parser.add_argument("--list-providers", action="store_true",
                        help="List supported providers and exit")

    args = parser.parse_args()

    # Handle list providers
    if args.list_providers:
        list_providers()
        return

    print("🚀 AI Model Compatibility Test")
    print("This script will test different AI models with LiteLLM.")

    if args.local:
        print("🏠 Testing LOCAL Ollama models only")
        if not check_ollama_available():
            print("❌ Ollama doesn't appear to be running!")
            print("💡 Start Ollama first: ollama serve")
            return
    elif args.provider:
        print(f"🔌 Testing {args.provider.upper()} provider only")
    else:
        print("🌐 Testing multiple providers")

    # Generation parameters (common across models)
    generation_params = {
        "temperature": 0.7,
        "max_tokens": 100,
        "top_p": 0.9
    }

    # Model configurations to test
    models_to_test = []

    # Add models based on mode
    if args.local or args.provider == "ollama":
        models_to_test.extend([
            {
                "name": "ollama/phi3",
                "config": generation_params,
                "description": "Phi-3 Mini via Ollama - Excellent quality, runs locally",
                "provider": "Ollama (local)"
            },
            {
                "name": "ollama/mistral",
                "config": generation_params,
                "description": "Mistral 7B via Ollama - Outstanding quality, runs locally",
                "provider": "Ollama (local)"
            },
            {
                "name": "ollama/tinyllama",
                "config": generation_params,
                "description": "TinyLlama via Ollama - Very fast, basic quality",
                "provider": "Ollama (local)"
            }
        ])

    if not args.local and (not args.provider or args.provider == "openai"):
        if os.getenv("OPENAI_API_KEY"):
            models_to_test.extend([
                {
                    "name": "gpt-3.5-turbo",
                    "config": generation_params,
                    "description": "GPT-3.5 Turbo - Fast and cost-effective",
                    "provider": "OpenAI (cloud)"
                },
                {
                    "name": "gpt-4",
                    "config": generation_params,
                    "description": "GPT-4 - Highest quality",
                    "provider": "OpenAI (cloud)"
                }
            ])
        else:
            print("⚠️  OPENAI_API_KEY not set, skipping OpenAI models")

    if not args.local and (not args.provider or args.provider == "anthropic"):
        if os.getenv("ANTHROPIC_API_KEY"):
            models_to_test.extend([
                {
                    "name": "claude-3-haiku-20240307",
                    "config": generation_params,
                    "description": "Claude 3 Haiku - Fast and efficient",
                    "provider": "Anthropic (cloud)"
                },
                {
                    "name": "claude-3-sonnet-20240229",
                    "config": generation_params,
                    "description": "Claude 3 Sonnet - Balanced performance",
                    "provider": "Anthropic (cloud)"
                }
            ])
        else:
            print("⚠️  ANTHROPIC_API_KEY not set, skipping Anthropic models")

    if not args.local and (not args.provider or args.provider == "huggingface"):
        if os.getenv("HUGGINGFACE_API_KEY"):
            models_to_test.extend([
                {
                    "name": "huggingface/microsoft/Phi-3-mini-4k-instruct",
                    "config": generation_params,
                    "description": "Phi-3 Mini via HuggingFace API",
                    "provider": "HuggingFace (cloud)"
                }
            ])
        else:
            print("⚠️  HUGGINGFACE_API_KEY not set, skipping HuggingFace models")

    if not models_to_test:
        print("❌ No models to test!")
        print("💡 Either:")
        print("   1. Start Ollama and use --local flag")
        print("   2. Set API keys for cloud providers (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)")
        return

    successful_models = []

    # Test each model
    for model_info in models_to_test:
        success = test_model(
            model_info["name"],
            model_info["config"],
            model_info["description"],
            model_info.get("provider", "")
        )

        if success:
            successful_models.append(model_info)

    # Summary
    print(f"\n{'=' * 60}")
    print("📊 SUMMARY")
    print(f"{'=' * 60}")

    if successful_models:
        print(f"✅ Working models on your system:")
        for model in successful_models:
            print(f"   • {model['name']} ({model.get('provider', 'unknown')})")

        print(f"\n💡 Recommendations:")

        # Categorize by provider
        local_models = [m for m in successful_models if "ollama" in m["name"].lower()]
        cloud_models = [m for m in successful_models if "ollama" not in m["name"].lower()]

        if local_models:
            print(f"   🏠 Local (Ollama) - No API costs, private:")
            for m in local_models:
                print(f"      • {m['name']}")

        if cloud_models:
            print(f"   ☁️  Cloud - High quality, requires API keys:")
            for m in cloud_models:
                print(f"      • {m['name']}")

    else:
        print("❌ No models worked successfully.")
        print("💡 Try:")
        print("   1. For local models: Start Ollama with 'ollama serve'")
        print("   2. For cloud models: Set appropriate API keys")
        print("   3. Check internet connection for cloud providers")
        print("   4. Run with --list-providers to see all options")


def list_providers():
    """List all supported providers and their requirements."""
    print("🔌 LiteLLM Supported Providers:")
    print(f"{'=' * 60}")

    providers = [
        {
            "name": "Ollama",
            "prefix": "ollama/",
            "example": "ollama/phi3",
            "requirements": "Local Ollama server running",
            "env_vars": "None (local)"
        },
        {
            "name": "OpenAI",
            "prefix": "",
            "example": "gpt-3.5-turbo",
            "requirements": "OpenAI account and API key",
            "env_vars": "OPENAI_API_KEY"
        },
        {
            "name": "Anthropic",
            "prefix": "",
            "example": "claude-3-sonnet-20240229",
            "requirements": "Anthropic account and API key",
            "env_vars": "ANTHROPIC_API_KEY"
        },
        {
            "name": "HuggingFace",
            "prefix": "huggingface/",
            "example": "huggingface/microsoft/Phi-3-mini-4k-instruct",
            "requirements": "HuggingFace account and token",
            "env_vars": "HUGGINGFACE_API_KEY"
        },
        {
            "name": "Azure OpenAI",
            "prefix": "azure/",
            "example": "azure/gpt-35-turbo",
            "requirements": "Azure account and deployment",
            "env_vars": "AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION"
        },
        {
            "name": "Cohere",
            "prefix": "",
            "example": "command-nightly",
            "requirements": "Cohere account and API key",
            "env_vars": "COHERE_API_KEY"
        }
    ]

    for provider in providers:
        print(f"\n📦 {provider['name']}")
        print(f"   Prefix: {provider['prefix']}")
        print(f"   Example: {provider['example']}")
        print(f"   Requirements: {provider['requirements']}")
        print(f"   Environment: {provider['env_vars']}")

    print(f"\n{'=' * 60}")
    print("💡 Usage Examples:")
    print("   python model_comparison.py --local              # Test Ollama")
    print("   python model_comparison.py --provider openai    # Test OpenAI")
    print("   python model_comparison.py                      # Test all available")


if __name__ == "__main__":
    main()