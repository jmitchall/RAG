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

## License

This implementation is provided as-is for educational and development purposes.