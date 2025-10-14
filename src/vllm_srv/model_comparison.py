#!/usr/bin/env python3
"""
Model Comparison Demo
====================

This script demonstrates different AI models you can use with vLLM.
Run this to see which models work with your hardware setup.
Supports both local and remote models.

Usage: 
    python model_comparison.py                    # Use remote models (default)
    python model_comparison.py --local           # Use local models only
    python model_comparison.py --download        # Download models locally first
"""

import argparse
import os
import shutil
import sys
from huggingface_hub import snapshot_download
from pathlib import Path
from vllm import LLM, SamplingParams

# Local models directory
LOCAL_MODELS_DIR = Path("./models")


def ensure_local_models_dir():
    """Create local models directory if it doesn't exist."""
    LOCAL_MODELS_DIR.mkdir(exist_ok=True)
    return LOCAL_MODELS_DIR


def get_local_model_path(model_name):
    """
    Convert HuggingFace model name to local path.
    
    Args:
        model_name: HuggingFace model name (e.g., 'facebook/opt-125m')
    
    Returns:
        Path object for local model directory
    """
    # Replace slashes with underscores for filesystem compatibility
    safe_name = model_name.replace("/", "_")
    return LOCAL_MODELS_DIR / safe_name


def check_local_model_exists(model_name):
    """
    Check if model exists locally and has required files.
    
    Args:
        model_name: HuggingFace model name
    
    Returns:
        bool: True if model exists locally with required files
    """
    local_path = get_local_model_path(model_name)

    if not local_path.exists():
        return False

    # Check for essential model files
    required_files = ["config.json"]
    optional_files = ["pytorch_model.bin", "model.safetensors", "tokenizer.json"]

    # Must have config.json
    if not (local_path / "config.json").exists():
        return False

    # Must have at least one model weight file
    has_weights = any((local_path / f).exists() for f in optional_files)

    return has_weights


def download_model_locally(model_name):
    """
    Download a model from HuggingFace to local directory.
    
    Args:
        model_name: HuggingFace model name to download
    
    Returns:
        str: Local path to downloaded model, or None if failed
    """
    try:
        ensure_local_models_dir()
        local_path = get_local_model_path(model_name)

        print(f"📥 Downloading {model_name} to {local_path}...")

        # Download model from HuggingFace Hub
        snapshot_download(
            repo_id=model_name,
            local_dir=local_path,
            local_dir_use_symlinks=False,  # Create actual files, not symlinks
            resume_download=True,  # Resume if partially downloaded
        )

        print(f"✅ Successfully downloaded {model_name}")
        return str(local_path)

    except Exception as e:
        print(f"❌ Failed to download {model_name}: {str(e)}")
        # Clean up partial download
        if local_path.exists():
            shutil.rmtree(local_path)
        return None


def get_model_path(model_name, use_local=False, download_if_missing=False):
    """
    Get the appropriate model path (local or remote).
    
    Args:
        model_name: HuggingFace model name
        use_local: Whether to prefer local models
        download_if_missing: Whether to download if not found locally
    
    Returns:
        str: Path to use for model loading
    """
    # Check if local model exists
    local_exists = check_local_model_exists(model_name)

    if use_local:
        if local_exists:
            local_path = get_local_model_path(model_name)
            print(f"🏠 Using local model: {local_path}")
            return str(local_path)
        elif download_if_missing:
            downloaded_path = download_model_locally(model_name)
            if downloaded_path:
                return downloaded_path
            else:
                print(f"⚠️  Download failed, falling back to remote: {model_name}")
                return model_name
        else:
            print(f"⚠️  Local model not found, falling back to remote: {model_name}")
            return model_name
    else:
        # Use remote model (HuggingFace will cache automatically)
        print(f"🌐 Using remote model: {model_name}")
        return model_name


def test_model(model_name, config, description, use_local=False, download_if_missing=False):
    """
    Test a specific AI model configuration.
    
    Args:
        model_name: Name of the model to test
        config: Dictionary of model configuration parameters
        description: Human-readable description of the model
        use_local: Whether to use local models
        download_if_missing: Whether to download if not found locally
    """
    print(f"\n{'=' * 60}")
    print(f"🔍 Testing: {model_name}")
    print(f"📝 Description: {description}")
    print(f"⚙️  Configuration: {config}")
    print(f"{'=' * 60}")

    try:
        # Get the appropriate model path (local or remote)
        model_path = get_model_path(model_name, use_local, download_if_missing)

        # Create the model with specified configuration
        llm = LLM(model=model_path, **config)

        # Simple test prompt
        test_prompt = "[INST] Hello! Please introduce yourself briefly. [/INST]"
        sampling_params = SamplingParams(temperature=0.7, max_tokens=100)

        # Generate response
        outputs = llm.generate([test_prompt], sampling_params)
        response = outputs[0].outputs[0].text

        print(f"✅ SUCCESS!")
        print(f"🤖 AI Response: {response[:200]}...")
        return True

    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False


def main():
    """Main function to test different model configurations."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test AI models with vLLM")
    parser.add_argument("--local", action="store_true",
                        help="Use local models only (must be downloaded first)")
    parser.add_argument("--download", action="store_true",
                        help="Download models locally before testing")
    parser.add_argument("--list-local", action="store_true",
                        help="List available local models and exit")

    args = parser.parse_args()

    # Handle list local models
    if args.list_local:
        list_local_models()
        return

    print("🚀 AI Model Compatibility Test")
    print("This script will test different AI models with your hardware setup.")
    print("Some models may require HuggingFace authentication.")

    if args.local:
        print("🏠 Using LOCAL models only")
    elif args.download:
        print("📥 Will DOWNLOAD models locally first")
    else:
        print("🌐 Using REMOTE models (default)")

    # Model configurations to test (in order of resource requirements)
    models_to_test = [
        {
            "name": "facebook/opt-125m",
            "config": {
                "gpu_memory_utilization": 0.5,
                "dtype": "float16"
            },
            "description": "Tiny model (125M params) - Works on any GPU, basic quality"

        },
        {
            "name": "microsoft/Phi-3-mini-4k-instruct",
            "config": {
                "gpu_memory_utilization": 0.75,
                "dtype": "float16",
                "trust_remote_code": True
            },
            "description": "Small model (3.8B params) - Excellent quality, 8GB+ GPU needed"
        },
        {
            "name": "mistralai/Mistral-7B-Instruct-v0.2",
            "config": {
                "gpu_memory_utilization": 0.50,
                "dtype": "float16",
                "max_model_len": 512,
                "enforce_eager": True,
                "max_num_seqs": 1
            },
            "description": "Medium model (7B params) - Outstanding quality, 16GB GPU + auth needed (Ultra-conservative settings)"
        }
    ]

    successful_models = []

    # Test each model
    for model_info in models_to_test:
        success = test_model(
            model_info["name"],
            model_info["config"],
            model_info["description"],
            use_local=args.local,
            download_if_missing=args.download
        )

        if success:
            successful_models.append(model_info["name"])

    # Summary
    print(f"\n{'=' * 60}")
    print("📊 SUMMARY")
    print(f"{'=' * 60}")

    if successful_models:
        print(f"✅ Working models on your system:")
        for model in successful_models:
            print(f"   • {model}")

        print(f"\n💡 Recommendation:")
        if "microsoft/Phi-3-mini-4k-instruct" in successful_models:
            print("   Use 'microsoft/Phi-3-mini-4k-instruct' for best balance of quality and performance!")
        elif "facebook/opt-125m" in successful_models:
            print("   Use 'facebook/opt-125m' for basic testing, consider upgrading GPU for better models.")
        else:
            print("   Check your GPU setup and consider the troubleshooting guide in README.md")
    else:
        print("❌ No models worked successfully.")
        print("💡 Try:")
        print("   1. Check your GPU has at least 4GB VRAM")
        print("   2. Ensure vLLM is properly installed")
        print("   3. For Mistral models, set up HuggingFace authentication")

    # Show local model usage info
    if not args.local and not args.download:
        print(f"\n💾 Local Model Tips:")
        print(f"   • Use --download to save models locally for faster loading")
        print(f"   • Use --local to use only downloaded models")
        print(f"   • Use --list-local to see what's downloaded")
        print(f"   • Models will be saved to: {LOCAL_MODELS_DIR}")


def list_local_models():
    """List all locally downloaded models."""
    print("🏠 Local Models:")
    print(f"📁 Directory: {LOCAL_MODELS_DIR}")

    if not LOCAL_MODELS_DIR.exists():
        print("❌ No local models directory found")
        return

    model_dirs = [d for d in LOCAL_MODELS_DIR.iterdir() if d.is_dir()]

    if not model_dirs:
        print("❌ No local models found")
        print("💡 Use --download to download models locally")
        return

    print(f"✅ Found {len(model_dirs)} local models:")

    for model_dir in sorted(model_dirs):
        original_name = model_dir.name.replace("_", "/")
        if check_local_model_exists(original_name):
            size = get_directory_size(model_dir)
            print(f"   • {original_name} ({size})")
        else:
            print(f"   ⚠️  {original_name} (incomplete/corrupted)")


def get_directory_size(path):
    """Get human-readable size of directory."""
    total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())

    for unit in ['B', 'KB', 'MB', 'GB']:
        if total_size < 1024.0:
            return f"{total_size:.1f} {unit}"
        total_size /= 1024.0
    return f"{total_size:.1f} TB"


if __name__ == "__main__":
    main()
