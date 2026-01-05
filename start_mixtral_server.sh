#!/bin/bash
# Mixtral 8x7B Server Startup Scripts

echo "Mixtral 8x7B vLLM Server Options"
echo "================================="

echo "1. Basic Mixtral Server (recommended for 16GB GPU):"
echo "vllm serve mistralai/Mixtral-8x7B-Instruct-v0.1 --host 127.0.0.1 --port 8000 --gpu-memory-utilization 0.9 --dtype float16"
echo ""

echo "2. Memory-optimized Mixtral Server:"
echo "vllm serve mistralai/Mixtral-8x7B-Instruct-v0.1 \\"
echo "  --host 127.0.0.1 \\"
echo "  --port 8000 \\"
echo "  --gpu-memory-utilization 0.85 \\"
echo "  --max-model-len 2048 \\"
echo "  --dtype float16 \\"
echo "  --kv-cache-dtype fp8"
echo ""

echo "3. High-performance Mixtral Server (if you have enough memory):"
echo "vllm serve mistralai/Mixtral-8x7B-Instruct-v0.1 \\"
echo "  --host 0.0.0.0 \\"
echo "  --port 8000 \\"
echo "  --gpu-memory-utilization 0.95 \\"
echo "  --max-model-len 8192 \\"
echo "  --dtype float16 \\"
echo "  --trust-remote-code"
echo ""

echo "Test the server with curl:"
echo 'curl http://localhost:8000/v1/completions \'
echo '  -H "Content-Type: application/json" \'
echo '  -d '\''{
echo '    "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
echo '    "prompt": "[INST] Hello, how are you? [/INST]",
echo '    "max_tokens": 100,
echo '    "temperature": 0.7
echo '  }'\'''
echo ""

echo "Or test chat completions:"
echo 'curl http://localhost:8000/v1/chat/completions \'
echo '  -H "Content-Type: application/json" \'
echo '  -d '\''{
echo '    "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
echo '    "messages": [{"role": "user", "content": "Write a Python hello world program."}],
echo '    "max_tokens": 200,
echo '    "temperature": 0.7
echo '  }'\'