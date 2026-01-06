# VLLM (LangChain Community) Initialization Arguments

Here are all possible arguments for instantiating the `VLLM` class from `langchain_community.llms`:

---

## Core Model Configuration

### `model` (str)
- **Type:** `str`
- **Default:** `""`
- **Description:** The name or path of a HuggingFace Transformers model. This can be either:
  - A model identifier from HuggingFace Hub (e.g., `"facebook/opt-125m"`)
  - A local path to a downloaded model directory
- **Usage:** Required parameter that specifies which language model to load and use for text generation.

---

## Hardware & Performance Configuration

### `tensor_parallel_size` (int)
- **Type:** `Optional[int]`
- **Default:** `1`
- **Description:** The number of GPUs to use for distributed execution with tensor parallelism. When set to a value greater than 1, the model's weight tensors are split across multiple GPUs to enable loading and running larger models.
- **Usage:** Set to the number of GPUs you want to use. For example, `tensor_parallel_size=2` will distribute the model across 2 GPUs.
- **Note:** Your system must have multiple GPUs available for values > 1.

### `dtype` (str)
- **Type:** `str`
- **Default:** `"auto"`
- **Description:** The data type for the model weights and activations. Controls the precision of numerical computations.
- **Valid Values:**
  - `"auto"` - Automatically selects the best dtype
  - `"float16"` - Half precision (16-bit floating point) - saves memory
  - `"float32"` - Full precision (32-bit floating point) - more accurate
  - `"bfloat16"` - Brain float 16 - good balance for modern hardware
- **Usage:** Use `"float16"` or `"bfloat16"` to reduce memory usage on GPU, or `"float32"` for maximum precision.

---

## Download & Storage Configuration

### `download_dir` (str)
- **Type:** `Optional[str]`
- **Default:** `None`
- **Description:** Directory to download and load the model weights. If not specified, uses the default HuggingFace cache directory (typically `~/.cache/huggingface/hub/`).
- **Usage:** Set this to control where models are stored locally. Example: `download_dir="./models"` will download models to a local `models` folder.

### `trust_remote_code` (bool)
- **Type:** `Optional[bool]`
- **Default:** `False`
- **Description:** Whether to trust and execute remote code from HuggingFace when downloading the model and tokenizer. Some models include custom code that needs to be executed.
- **Usage:** Set to `True` when using models that require custom code execution (e.g., certain Microsoft Phi models). Only enable this for models from trusted sources.

---

## Text Generation Parameters

### `max_new_tokens` (int)
- **Type:** `int`
- **Default:** `512`
- **Description:** Maximum number of new tokens to generate per output sequence. This controls the maximum length of the AI's response.
- **Usage:** Increase for longer responses, decrease for shorter ones. Example: `max_new_tokens=256` limits responses to ~256 words/tokens.

### `temperature` (float)
- **Type:** `float`
- **Default:** `1.0`
- **Range:** `0.0` to `2.0` (typically)
- **Description:** Controls the randomness of the sampling. Lower values make the output more focused and deterministic, while higher values make it more creative and random.
  - `0.0` - Completely deterministic (always picks the most likely token)
  - `0.7` - Balanced creativity (common default)
  - `1.0` - Standard sampling
  - `>1.0` - Very creative/random output
- **Usage:** Use lower values (0.1-0.5) for factual tasks, higher values (0.7-1.0) for creative writing.

### `top_p` (float)
- **Type:** `float`
- **Default:** `1.0`
- **Range:** `0.0` to `1.0`
- **Description:** Nucleus sampling parameter. Controls the cumulative probability of the top tokens to consider. The model only considers tokens whose cumulative probability adds up to `top_p`.
  - `0.1` - Very focused, only top 10% of probability mass
  - `0.9` - Balanced (common choice)
  - `1.0` - Consider all tokens
- **Usage:** Use with temperature. Typical value is `0.9` for good balance between quality and diversity.

### `top_k` (int)
- **Type:** `int`
- **Default:** `-1`
- **Description:** Integer that controls the number of highest probability vocabulary tokens to keep for top-k filtering.
  - `-1` - Disabled (consider all tokens)
  - `>0` - Only consider the top K most likely tokens
- **Usage:** Set to a positive value (e.g., `50`) to limit token selection to the top K choices. Often used with `top_p`.

---

## Advanced Sampling Parameters

### `n` (int)
- **Type:** `int`
- **Default:** `1`
- **Description:** Number of output sequences to return for each given prompt. The model will generate `n` different completions for each input.
- **Usage:** Set to a value > 1 when you want multiple different responses to the same prompt. Example: `n=3` returns 3 different completions.

### `best_of` (int)
- **Type:** `Optional[int]`
- **Default:** `None`
- **Description:** Number of output sequences that are generated from the prompt internally. The best `n` sequences are returned. Must be ≥ `n`.
- **Usage:** Used for quality control. Generate more candidates internally and return only the best ones. Example: `best_of=5, n=1` generates 5 sequences internally but only returns the best one.

### `use_beam_search` (bool)
- **Type:** `bool`
- **Default:** `False`
- **Description:** Whether to use beam search instead of sampling. Beam search explores multiple possible sequences in parallel and selects the best overall.
- **Usage:** Set to `True` for more deterministic and higher-quality outputs, but slower generation. Best for tasks requiring high accuracy.
- **Note:** When using beam search, `temperature`, `top_p`, and `top_k` are ignored.

---

## Penalty Parameters

### `presence_penalty` (float)
- **Type:** `float`
- **Default:** `0.0`
- **Range:** `-2.0` to `2.0` (typically)
- **Description:** Penalizes new tokens based on whether they appear in the generated text so far. Positive values reduce repetition by penalizing tokens that have already been used (regardless of how many times).
- **Usage:** Use positive values (0.1-1.0) to encourage topic diversity and reduce repetition.

### `frequency_penalty` (float)
- **Type:** `float`
- **Default:** `0.0`
- **Range:** `-2.0` to `2.0` (typically)
- **Description:** Penalizes new tokens based on their frequency in the generated text so far. Unlike presence penalty, this scales with how often a token has appeared.
- **Usage:** Use positive values (0.1-1.0) to strongly discourage word repetition. Higher values create more novel text.

---

## Stop Conditions

### `stop` (list of str)
- **Type:** `Optional[List[str]]`
- **Default:** `None`
- **Description:** List of strings that stop the generation when they are encountered in the output. When any of these strings is generated, text generation stops immediately.
- **Usage:** Use to control where generation ends. Example: `stop=["\n\n", "###", "END"]` stops at double newlines or special markers.

### `ignore_eos` (bool)
- **Type:** `bool`
- **Default:** `False`
- **Description:** Whether to ignore the EOS (End Of Sequence) token and continue generating tokens after the EOS token is generated.
- **Usage:** Set to `True` if you want generation to continue even after the model thinks it's done. Useful for forcing minimum length outputs.

---

## Output Control

### `logprobs` (int)
- **Type:** `Optional[int]`
- **Default:** `None`
- **Description:** Number of log probabilities to return per output token. Returns the probability scores for the generated tokens.
- **Usage:** Set to a positive integer (e.g., `5`) to get the top 5 most likely alternatives for each generated token. Useful for understanding model confidence and exploring alternative outputs.

---

## Extended Configuration

### `vllm_kwargs` (dict)
- **Type:** `Dict[str, Any]`
- **Default:** `{}` (empty dict)
- **Description:** Holds any additional model parameters that are valid for `vllm.LLM` initialization but not explicitly specified as class attributes. These are passed directly to the underlying vLLM engine.
- **Usage:** Use this for advanced vLLM-specific parameters like:
  ```python
  vllm_kwargs={
      "gpu_memory_utilization": 0.9,  # Use 90% of GPU memory
      "max_model_len": 4096,           # Maximum sequence length
      "quantization": "awq",           # Model quantization method
      "seed": 42                       # Random seed for reproducibility
  }
  ```

---

## Example Usage

```python
from langchain_community.llms import VLLM

# Basic usage
llm = VLLM(
    model="facebook/opt-125m",
    temperature=0.7,
    top_p=0.9,
    max_new_tokens=256
)

# Advanced usage with multiple parameters
llm = VLLM(
    model="microsoft/Phi-3-mini-4k-instruct",
    tensor_parallel_size=1,
    trust_remote_code=True,
    dtype="float16",
    download_dir="./models",
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    max_new_tokens=512,
    presence_penalty=0.5,
    frequency_penalty=0.5,
    stop=["\n\n", "###"],
    vllm_kwargs={
        "gpu_memory_utilization": 0.9,
        "max_model_len": 4096
    }
)
```

---

## Parameter Interaction Notes

1. **Sampling vs Beam Search**: When `use_beam_search=True`, temperature, top_p, and top_k are ignored.

2. **Multiple Outputs**: When using `n > 1` or `best_of`, memory usage increases proportionally.

3. **Memory Management**: Use `dtype="float16"` and configure `vllm_kwargs` with `gpu_memory_utilization` to optimize memory usage.

4. **Quality vs Speed**: Lower temperature + beam search = higher quality but slower. Higher temperature + sampling = faster but more random.

# VLLM Chat Model with Tool Calling Support

This implementation provides a custom chat model wrapper around VLLM that enables tool calling functionality, bridging the gap between VLLM's text completion interface and LangChain's chat model with structured tool calling.

## Features

- ✅ **Tool Calling Support**: Full `.bind_tools()` compatibility with LangChain
- ✅ **Multiple Tool Support**: Call multiple tools in a single response
- ✅ **Mistral Format**: Optimized for Mistral instruction format
- ✅ **Error Handling**: Robust error handling and fallback responses
- ✅ **Chat Interface**: Standard LangChain chat model interface
- ✅ **Message Conversion**: Seamless conversion between chat messages and VLLM prompts

## Architecture

```
┌─────────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   LangChain App     │ -> │  VLLMChatModel   │ -> │     VLLM        │
│                     │    │                  │    │                 │
│ • bind_tools()      │    │ • Message Conv   │    │ • Text Gen      │
│ • invoke()          │    │ • Tool Parsing   │    │ • Mistral Model │
│ • Chat Messages     │    │ • Response Parse │    │ • GPTQ Quant    │
└─────────────────────┘    └──────────────────┘    └─────────────────┘
```

## Installation

Ensure you have the required dependencies:

```bash
pip install langchain-core langchain-community vllm transformers
```

## Usage

### Basic Setup

```python
from minstral_langchain import create_vllm_chat_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

# Create the chat model
chat_model = create_vllm_chat_model(
    download_dir="./models",
    gpu_memory_utilization=0.8,
    max_tokens=256,
    temperature=0.7
)
```

### Define Tools

```python
@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    # Your weather API logic here
    return f"Weather in {location}: Sunny, 72°F"

@tool  
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        result = eval(expression)  # Use safely in production
        return f"{expression} = {result}"
    except:
        return "Error in calculation"
```

### Bind Tools and Use

```python
# Bind tools to the model
model_with_tools = chat_model.bind_tools([get_weather, calculate])

# Create messages
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What's the weather in San Francisco and what's 15 * 23?")
]

# Generate response
response = model_with_tools.invoke(messages)

print("Assistant:", response.content)

# Check for tool calls
if response.tool_calls:
    for tool_call in response.tool_calls:
        print(f"Tool: {tool_call['name']}")
        print(f"Args: {tool_call['args']}")
        # Execute tool and handle results...
```

### Agent Loop Example

```python
from langchain_core.messages import ToolMessage

def run_agent_loop(model_with_tools, messages, max_iterations=5):
    """Simple agent loop that executes tools."""
    
    current_messages = messages.copy()
    
    for i in range(max_iterations):
        # Generate response
        response = model_with_tools.invoke(current_messages)
        current_messages.append(response)
        
        # If no tool calls, we're done
        if not response.tool_calls:
            return response
        
        # Execute tool calls
        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            
            # Execute tool (implement your tool execution logic)
            if tool_name == 'get_weather':
                result = get_weather(**tool_args)
            elif tool_name == 'calculate':
                result = calculate(**tool_args)
            else:
                result = f"Unknown tool: {tool_name}"
            
            # Add tool result to messages
            tool_message = ToolMessage(
                content=result,
                name=tool_name,
                tool_call_id=tool_call['id']
            )
            current_messages.append(tool_message)
    
    return response

# Use the agent loop
final_response = run_agent_loop(model_with_tools, messages)
print("Final response:", final_response.content)
```

## Configuration Options

### Model Parameters

```python
chat_model = create_vllm_chat_model(
    download_dir="./models",           # Model download directory
    gpu_memory_utilization=0.8,       # GPU memory usage (0.1-0.95)
    max_model_len=8192,               # Maximum context length
    top_k=5,                          # Top-k sampling
    max_tokens=512,                   # Max tokens to generate
    temperature=0.7                   # Sampling temperature
)
```

### Tool Calling Format

The model expects and generates tool calls in this format:

```
TOOL_CALL: {"name": "tool_name", "arguments": {"param1": "value1", "param2": "value2"}}
```

Multiple tools:
```
TOOL_CALL: {"name": "get_weather", "arguments": {"location": "San Francisco"}}
TOOL_CALL: {"name": "calculate", "arguments": {"expression": "15 * 23"}}
```

## Advanced Features

### Custom Tool Instructions

The wrapper automatically adds tool instructions to prompts:

```
AVAILABLE TOOLS:
- get_weather: Get the current weather for a given location
  Parameters: location (str): The city and country/state to get weather for
- calculate: Safely calculate a mathematical expression  
  Parameters: expression (str): A mathematical expression to evaluate

INSTRUCTIONS:
- To call a tool, respond with: TOOL_CALL: {"name": "tool_name", "arguments": {"param1": "value1"}}
- You can call multiple tools by using multiple TOOL_CALL lines
- Always provide a human-readable response along with any tool calls
- If no tools are needed, respond normally without TOOL_CALL
```

### Message Format Conversion

The wrapper handles conversion between LangChain message types and Mistral format:

- `SystemMessage` -> System instructions
- `HumanMessage` -> "Human: ..."  
- `AIMessage` -> "Assistant: ..." (with tool calls)
- `ToolMessage` -> "Tool Result (tool_name): ..."

Final format: `<s>[INST] {combined_content} [/INST]`

### Error Handling

The implementation includes robust error handling:

- Malformed JSON in tool calls -> Logged and skipped
- VLLM generation errors -> Fallback error message
- Missing tool parameters -> Graceful degradation

## Testing

Run the test suite:

```bash
python test_vllm_chat.py
```

Run the example:

```bash
python vllm_chat_example.py
```

## Limitations

1. **Prompt Engineering**: Tool calling effectiveness depends on the model's ability to follow instructions
2. **Response Parsing**: Relies on regex parsing of structured output
3. **Single Response**: Tools are called within a single model response (no iterative calling)
4. **Model Specific**: Optimized for Mistral instruction format

## Future Enhancements

- [ ] Support for parallel tool execution
- [ ] Better structured output parsing (e.g., using grammar constraints)
- [ ] Integration with LangGraph for complex agent workflows
- [ ] Support for different model prompt formats
- [ ] Tool result validation and retry logic
- [ ] Streaming support for tool calls

## Troubleshooting

### Common Issues

1. **GPU Memory**: Reduce `gpu_memory_utilization` if out of memory
2. **Model Loading**: Ensure model files are accessible in `download_dir`
3. **Tool Calls Not Parsed**: Check that model generates exact `TOOL_CALL:` format
4. **Import Errors**: Verify all dependencies are installed

### Debug Mode

Enable logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

This implementation provides a foundation for VLLM tool calling. Contributions welcome for:

- Additional model format support
- Better parsing strategies  
- Performance optimizations
- Integration with other LangChain components

### Out of Memory Issues
```bash
# Reduce memory usage
vllm serve mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 2048 \
  --kv-cache-dtype fp8
```

### Slow Loading
```bash
# Monitor GPU memory during loading
watch -n 1 nvidia-smi

# Check download progress (first run only)
ls -la ~/.cache/huggingface/hub/
```

## Quick Reference

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--host` | Server host address | `--host 0.0.0.0` |
| `--port` | Server port | `--port 8000` |
| `--gpu-memory-utilization` | GPU memory fraction | `--gpu-memory-utilization 0.8` |
| `--tensor-parallel-size` | Number of GPUs for tensor parallelism | `--tensor-parallel-size 2` |
| `--max-model-len` | Maximum sequence length | `--max-model-len 2048` |
| `--quantization` | Quantization method | `--quantization gptq` |
| `--dtype` | Model data type | `--dtype float16` |
| `--api-key` | API authentication key | `--api-key secret` |
| `--trust-remote-code` | Allow remote code execution | `--trust-remote-code` |
| `--seed` | Random seed for reproducibility | `--seed 42` |

# vLLM Supported Models Compatible with RTX 5080 16GB

## **Model Compatibility Overview**

This documentation categorizes vLLM supported models according to the official vLLM taxonomy and their compatibility with RTX 5080 16GB hardware constraints.
# vLLM Supported Models Compatible with RTX 5080 16GB
├── List of Text-only Language Models<br>
│   ├── Generative Models<br>
│   │&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── Text Generation<br>
│   │&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── Microsoft Phi Models (✅ Recommended)<br>
│   │&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── Facebook OPT Models <br>
│   │&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── Google Gemma Models<br>
│   │&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── Other Compatible Models<br>
│   │&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── Models That May Work (⚠️)<br>
│   │&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── Models That Won't Work (❌)<br>
│   └── Pooling Models<br>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── Embedding<br>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── Classification<br>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── Cross-encoder/Reranker<br>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── Reward Modeling<br>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── Token Classification<br>
└── List of Multimodal Language Models<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── Generative Models<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── Text Generation (Multimodal)<br>

### Memory Requirements for RTX 5080 16GB
- **Total VRAM**: 16GB
- **System Overhead**: ~2GB (CUDA runtime, drivers, etc.)
- **Available for Models**: ~14GB
- **Memory Components**: Model weights + KV cache + activation memory + overhead