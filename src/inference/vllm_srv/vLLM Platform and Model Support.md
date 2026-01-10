# vLLM Platform

> **vLLM does _not_ support native Windows.**
>
> vLLM is only supported on Linux (including WSL2) and MacOS. Native Windows installation will fail due to platform incompatibility.
>
> To use vLLM on Windows:
> 1. Install [WSL2 with Ubuntu](https://learn.microsoft.com/en-us/windows/wsl/install).
> 2. Install [Nvidia CUDA for WSL2](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
> 3. Set up your Python environment and install vLLM inside WSL2.
>
> Alternatively, use the [official vLLM Docker images](https://vllm.readthedocs.io/en/latest/getting_started/docker.html) with GPU support.

# 🚀 Getting Started with vLLM - Complete Setup Guide

## 📋 Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with at least 8GB VRAM (16GB+ recommended)
- **RAM**: 16GB+ system RAM
- **Storage**: 50GB+ free space for models
- **OS**: Linux (including WSL2) or macOS

### GPU Memory Requirements by Model
| Model | Parameters | VRAM Needed | Authentication | Quality | 16GB GPU Compatible |
|-------|------------|-------------|----------------|---------|-------------------|
| **facebook/opt-125m** | 125M | <1GB | ❌ None | Basic | ✅ Yes |
| **microsoft/Phi-3-mini-4k-instruct** | 3.8B | ~8GB | ❌ None | Excellent | ✅ **Recommended** |
| **mistralai/Mistral-7B-Instruct-v0.2** | 7B | ~20GB | ✅ Required | Outstanding | ❌ **No** (needs 20GB+) |
| **mistralai/Mixtral-8x7B-Instruct-v0.1** | 8x7B | ~24GB+ | ✅ Required | State-of-art | ❌ No (needs 24GB+) |

## 🛠️ Installation Steps

### Step 1: Set Up Python Environment

```bash
# Create project directory
mkdir vllm-srv
cd vllm-srv

# Create virtual environment with Python 3.10 (recommended for compatibility)
uv venv --python 3.10 --seed

# Activate environment
source .venv/bin/activate  # Linux/WSL2
# OR
.venv\Scripts\activate     # Windows (if using PowerShell in WSL)
```

### Step 2: Install vLLM

```bash
# Install vLLM and dependencies
uv pip install vllm

# For additional features (optional)
uv pip install "vllm[cpu]"     # CPU fallback support
uv pip install "vllm[ray]"     # Distributed inference
```

### Step 3: Test Basic Installation

```bash
# Quick test with small model (no authentication needed)
uv run python -c "from vllm import LLM; print('✅ vLLM installed successfully!')"
```

## 🔐 HuggingFace Authentication Setup

Many high-quality models require authentication. Here's how to set it up:

### Option 1: Web-based Setup
1. **Create Account**: Go to [HuggingFace.co](https://huggingface.co) and sign up
2. **Request Model Access**: 
   - Visit [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
   - Click "Request access" (usually approved instantly)
3. **Generate Token**:
   - Go to [Settings > Tokens](https://huggingface.co/settings/tokens)
   - Create new token with "Read" permissions
4. **Login via CLI**:
   ```bash
   uv run huggingface-cli login
   # Paste your token when prompted
   ```

### Option 2: Environment Variable
```bash
# Set token as environment variable
export HUGGING_FACE_HUB_TOKEN="your-token-here"
```

## 🎯 Running Your First AI Model

### Quick Start with Basic Model (No Auth Required)

```bash
# Create and run basic example
cat > test_basic.py << 'EOF'
from vllm import LLM, SamplingParams

# Load small, fast model
llm = LLM(model="facebook/opt-125m")

# Ask a question
prompts = ["The capital of France is"]
sampling_params = SamplingParams(temperature=0.7, max_tokens=50)

# Generate response
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(f"Question: {output.prompt}")
    print(f"Answer: {output.outputs[0].text}")
EOF

# Run the test
uv run python test_basic.py
```

### Recommended Model for Most Users

```bash
# Create advanced example with Phi-3 Mini (excellent quality, no auth needed)
cat > test_phi3.py << 'EOF'
from vllm import LLM, SamplingParams

# Load high-quality model (works great on 16GB+ GPUs)
llm = LLM(
    model="microsoft/Phi-3-mini-4k-instruct",
    gpu_memory_utilization=0.9,
    dtype="float16",
    trust_remote_code=True
)

# Ask questions using proper instruction format
prompts = [
    "[INST] Write a Python function to calculate fibonacci numbers. [/INST]",
    "[INST] Explain machine learning in simple terms. [/INST]"
]

sampling_params = SamplingParams(temperature=0.7, max_tokens=200)

# Generate responses
outputs = llm.generate(prompts, sampling_params)
for i, output in enumerate(outputs, 1):
    print(f"\n--- Question {i} ---")
    print(f"Human: {output.prompt}")
    print(f"AI: {output.outputs[0].text}")
EOF

# Run the advanced test
uv run python test_phi3.py
```

## 🖥️ Starting a vLLM Server

### Basic Server (API Compatible with OpenAI)

```bash
# Start server with recommended model
uv run vllm serve microsoft/Phi-3-mini-4k-instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --dtype float16 \
  --trust-remote-code

# Server will be available at http://127.0.0.1:8000
```

### Test Your Server

```bash
# Test with curl
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/Phi-3-mini-4k-instruct",
    "prompt": "[INST] Hello, how are you? [/INST]",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

## 🎛️ Model Selection Guide

### For Beginners (8GB+ GPU)
```bash
# Phi-3 Mini: Best balance of quality and resource usage
uv run vllm serve microsoft/Phi-3-mini-4k-instruct \
  --host 127.0.0.1 --port 8000 --trust-remote-code
```

### For Power Users (20GB+ GPU, with HuggingFace Auth)
```bash
# Mistral 7B: Professional-grade responses (REQUIRES 20GB+ GPU)
# ⚠️ NOT COMPATIBLE with 16GB GPUs due to memory requirements
uv run vllm serve mistralai/Mistral-7B-Instruct-v0.2 \
  --host 127.0.0.1 --port 8000 \
  --gpu-memory-utilization 0.85 \
  --dtype float16 \
  --max-model-len 2048
```

### Alternative for 16GB GPUs (Recommended)
```bash
# Phi-3 Mini: Excellent quality that actually works on 16GB
uv run vllm serve microsoft/Phi-3-mini-4k-instruct \
  --host 127.0.0.1 --port 8000 \
  --gpu-memory-utilization 0.75 \
  --dtype float16 \
  --trust-remote-code
```

### For Researchers (24GB+ GPU, with HuggingFace Auth)
```bash
# Mixtral 8x7B: State-of-the-art performance
uv run vllm serve mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --host 127.0.0.1 --port 8000 \
  --gpu-memory-utilization 0.85 \
  --dtype float16 \
  --max-model-len 4096
```

## 🐛 Troubleshooting Common Issues

### Authentication Errors
```
Error: 401 Client Error: Unauthorized
Solution: Set up HuggingFace authentication (see section above)
```

### Out of Memory Errors
```
Error: CUDA out of memory
Solutions:
1. Use smaller model (Phi-3 Mini instead of Mistral 7B)
2. Reduce gpu_memory_utilization to 0.8
3. Add --max-model-len 2048 to reduce context length
4. Use --kv-cache-dtype fp8 for memory optimization
```

### KV Cache Memory Errors
```
Error: ValueError: No available memory for the cache blocks. 
Try increasing `gpu_memory_utilization` when initializing the engine.

Solutions (counter-intuitive - REDUCE memory utilization):
1. DECREASE gpu_memory_utilization from 0.9 to 0.75 or 0.6
2. Add memory optimization flags:
   - enforce_eager=True (disables CUDA graphs)
   - kv_cache_dtype="fp8" (uses 8-bit cache)
   - quantization="fp8" (quantize model weights)
3. Reduce context length: max_model_len=2048 or 1024
4. Check for memory leaks: restart Python completely

⚠️ The error message is misleading - you usually need LESS memory utilization, not more!
```

**❌ REALITY CHECK: Mistral 7B Not Compatible with 16GB GPUs**

Based on testing, Mistral 7B requires ~13.5GB just for model weights, leaving negative memory for KV cache on 16GB GPUs. **No amount of optimization can fix this fundamental limitation.**

**✅ WORKING ALTERNATIVE for 16GB GPUs:**
```bash
# Use Phi-3 Mini instead - excellent quality, actually works
vllm serve microsoft/Phi-3-mini-4k-instruct \
  --gpu-memory-utilization 0.75 \
  --dtype float16 \
  --trust-remote-code
```

**For Mistral 7B, you need:**
- RTX 4090 (24GB)
- RTX 6000 Ada (48GB)  
- A6000 (48GB)
- Or similar 20GB+ GPU

### Import Errors
```
Error: ModuleNotFoundError: No module named 'vllm'
Solution: Ensure virtual environment is activated and vLLM is installed
```

## 📚 Next Steps

1. **Explore the Python Script**: Check out `basic.py` for a comprehensive, commented example
2. **Try Different Models**: Experiment with models from the comparison table above
3. **Set Up Server Integration**: Use the server with web applications or APIs
4. **Performance Tuning**: Adjust memory utilization and context length for your hardware

---

**💡 Pro Tip**: Start with `microsoft/Phi-3-mini-4k-instruct` - it provides excellent quality without authentication requirements and runs well on most modern GPUs!

**⚠️ IMPORTANT for 16GB GPUs**: Mistral 7B requires ~13.5GB just for model weights, leaving insufficient memory for KV cache. Use Phi-3 Mini (3.8B params) which uses ~7GB and provides comparable quality.

# vLLM Overview

**vLLM** (Very Large Language Model) is a high-performance inference and serving engine for large language models. It's designed to maximize throughput and minimize latency when serving LLMs at scale.

## Key Features:
- **High Throughput**: Uses PagedAttention for efficient memory management
- **Fast Inference**: Optimized CUDA kernels and continuous batching
- **OpenAI Compatible API**: Drop-in replacement for OpenAI API endpoints
- **Multi-GPU Support**: Tensor and pipeline parallelism for large models
- **Quantization Support**: GPTQ, AWQ, and other quantization methods

# vLLM CLI Cheat Sheet

## Basic Server Commands

https://docs.vllm.ai/en/latest/models/supported_models.html
If vLLM natively supports a model, its implementation can be found in [vllm/model_executor/models.](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/models)

### Start a basic server
```bash
vllm serve microsoft/DialoGPT-medium
```

### Serve with custom host and port
```bash
vllm serve facebook/opt-125m --host 0.0.0.0 --port 8000
```

### Serve with GPU memory fraction

```bash
vllm serve meta-llama/Llama-2-7b-hf --gpu-memory-utilization 0.8
```

## Model Loading Options

### Load quantized model (GPTQ)
```bash
vllm serve TheBloke/Llama-2-7B-Chat-GPTQ --quantization gptq
```

### Load AWQ quantized model
```bash
vllm serve TheBloke/Llama-2-7B-Chat-AWQ --quantization awq
```

### Load model with custom tensor parallel size
```bash
vllm serve meta-llama/Llama-2-13b-hf --tensor-parallel-size 2
```

### Load model with pipeline parallelism
```bash
vllm serve meta-llama/Llama-2-70b-hf --tensor-parallel-size 4 --pipeline-parallel-size 2
```

## Performance & Memory Options

### Set maximum model length
```bash
vllm serve facebook/opt-1.3b --max-model-len 2048
```

### Enable KV cache quantization
```bash
vllm serve meta-llama/Llama-2-7b-hf --kv-cache-dtype fp8
```

### Set block size for PagedAttention
```bash
vllm serve facebook/opt-125m --block-size 16
```

### Configure swap space
```bash
vllm serve meta-llama/Llama-2-7b-hf --swap-space 4
```

## API & Security Options

### Disable authentication
```bash
vllm serve facebook/opt-125m --disable-log-requests
```

### Enable API key authentication
```bash
vllm serve facebook/opt-125m --api-key your-secret-key
```

### Set custom model name for API
```bash
vllm serve facebook/opt-125m --served-model-name my-custom-model
```

## Advanced Configuration

### Load with custom trust remote code
```bash
vllm serve microsoft/DialoGPT-medium --trust-remote-code
```

### Set custom seed for reproducibility
```bash
vllm serve facebook/opt-125m --seed 42
```

### Enable streaming responses
```bash
vllm serve facebook/opt-125m --disable-log-stats
```

### Load with custom dtype
```bash
vllm serve meta-llama/Llama-2-7b-hf --dtype float16
```

## Offline Batch Inference

### Run offline inference on text file
```bash
vllm generate --model facebook/opt-125m --input-file prompts.txt --output-file results.txt
```

### Batch inference with custom parameters
```bash
vllm generate --model meta-llama/Llama-2-7b-hf --temperature 0.7 --max-tokens 100 --input-file prompts.txt
```

## Multi-GPU Configuration

### Distributed serving across multiple nodes
```bash
# On master node
vllm serve meta-llama/Llama-2-70b-hf --tensor-parallel-size 8 --distributed-executor-backend ray

# On worker nodes
ray start --address=<master-ip>:10001
```

## Monitoring & Debugging

### Enable detailed logging
```bash
vllm serve facebook/opt-125m --log-level DEBUG
```

### Monitor GPU memory usage
```bash
vllm serve meta-llama/Llama-2-7b-hf --gpu-memory-utilization 0.9 --enforce-eager
```

## Common Model Examples

### Serve Llama 2 7B Chat
```bash
vllm serve meta-llama/Llama-2-7b-chat-hf --host 0.0.0.0 --port 8000
```

### Serve Code Llama
```bash
vllm serve codellama/CodeLlama-7b-Python-hf --max-model-len 4096
```

### Serve Mistral 7B
```bash
vllm serve mistralai/Mistral-7B-Instruct-v0.1 --gpu-memory-utilization 0.8
```

### Serve Mixtral 8x7B (Mixture of Experts)
```bash
# Basic Mixtral 8x7B server (recommended for 16GB GPU)
vllm serve mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --host 127.0.0.1 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --dtype float16

# Memory-optimized for smaller GPUs
vllm serve mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --host 127.0.0.1 \
  --port 8000 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 2048 \
  --dtype float16 \
  --kv-cache-dtype fp8

# High-performance setup (requires sufficient VRAM)
vllm serve mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 8192 \
  --dtype float16 \
  --trust-remote-code
```

### Serve Phi-3 Mini
```bash
vllm serve microsoft/Phi-3-mini-4k-instruct --trust-remote-code
```

## Testing Your vLLM Server

### Test with curl
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/opt-125m",
    "prompt": "Hello, how are you?",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

### Test chat completions
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

# Mixtral 8x7B Usage Examples

## Overview
Mixtral 8x7B is a Mixture of Experts (MoE) model that provides excellent performance with efficient resource usage. It requires ~13-14GB VRAM in float16 precision.

## GPU Requirements
- **Minimum**: 16GB VRAM (RTX 4080, RTX 5080, A6000)
- **Recommended**: 24GB+ VRAM for optimal performance
- **Memory Usage**: ~13-14GB in float16, ~26-28GB in float32

## Server Setup Examples

### Basic Mixtral Server
```bash
vllm serve mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --host 127.0.0.1 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --dtype float16
```

### Production Server with API Key
```bash
vllm serve mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --dtype float16 \
  --api-key your-secret-key \
  --served-model-name mixtral-8x7b
```

## Testing Mixtral Server

### Test Completions (Instruction Format)
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "prompt": "[INST] Write a Python function to calculate fibonacci numbers. [/INST]",
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

### Test Chat Completions
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "messages": [
      {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    "max_tokens": 300,
    "temperature": 0.7
  }'
```

## Python Usage Examples

### Basic Mixtral Usage
```python
from vllm import LLM, SamplingParams

# Initialize Mixtral
llm = LLM(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    gpu_memory_utilization=0.9,
    dtype="float16"
)

# Prepare prompts (use instruction format)
prompts = [
    "[INST] Write a Python hello world program. [/INST]",
    "[INST] Explain the benefits of using virtual environments in Python. [/INST]"
]

# Generate responses
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Response: {output.outputs[0].text}")
    print("-" * 50)
```

### Advanced Mixtral Configuration
```python
from vllm import LLM, SamplingParams

# Advanced configuration for optimal performance
llm = LLM(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    gpu_memory_utilization=0.9,
    max_model_len=4096,
    dtype="float16",
    trust_remote_code=True,
    # Enable KV cache quantization for memory efficiency
    kv_cache_dtype="fp8"
)

# Optimized sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
    repetition_penalty=1.1,
    stop=["[INST]", "</s>"]  # Stop tokens for Mixtral
)
```