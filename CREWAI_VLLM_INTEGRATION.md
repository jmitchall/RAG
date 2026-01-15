# Running vLLM with CrewAI: Rules and Important Notes

## Overview

This document explains the requirements, rules, and important considerations for running a local vLLM model with CrewAI agents. CrewAI was designed primarily for cloud-based LLMs (OpenAI, Anthropic, etc.), so integrating a local vLLM instance requires specific architectural patterns and workarounds.

---

## How OpenAI Relates to These Changes

### CrewAI's OpenAI-Centric Design

CrewAI was originally built with OpenAI's API as the primary LLM provider. This design decision influences many aspects of the framework:

#### 1. **Environment Variable - Custom LLM vs String Model Names**

When using custom LLM instances (like our vLLM wrapper), the `OPENAI_API_KEY` environment variable is **NOT required**.

```python
# ✅ No environment variable needed with custom LLM instances
custom_llm = create_crewai_vllm_model(download_dir="./models")
agent = Agent(role="Assistant", goal="Help", backstory="...", llm=custom_llm)
```

**The OpenAI Connection:**
- OpenAI API requires an API key for authentication
- CrewAI checks for this key when you specify a model as a string (e.g., `"gpt-4"`, `"gpt-3.5-turbo"`)
- String model names trigger CrewAI to instantiate an OpenAI client
- Custom LLM instances bypass this entire code path

**When It's Required vs Not Required:**

| LLM Type | OPENAI_API_KEY Required? | Example |
|----------|-------------------------|---------|
| String model name | ✅ Yes | `llm="gpt-4"` |
| Custom LLM instance | ❌ No | `llm=create_crewai_vllm_model()` |

---

#### 2. **BaseLLM Interface Design**

The `BaseLLM` abstract class that all CrewAI LLMs must inherit from was designed to mirror OpenAI's API patterns:

| OpenAI API Pattern | CrewAI BaseLLM | vLLM Adaptation |
|-------------------|----------------|-----------------|
| `messages` parameter | `messages: str \| List[Dict]` | Must convert to LangChain format |
| `model` parameter | `model: str` (required) | Use dummy like "vllm/mistral" |
| `tools` parameter | `tools: List[Dict]` | Passed but not used natively |
| `temperature` | Passed in `**kwargs` | Handled during vLLM initialization |
| Returns text string | Returns `str` | Extract from LLM result object |

**Why this matters:**
- OpenAI's API is the "gold standard" interface CrewAI expects
- Custom LLMs must emulate this interface even if their internals differ
- The wrapper translates between CrewAI's OpenAI-like expectations and vLLM's reality

---

#### 3. **Function Calling Paradigm**

This is where OpenAI's influence is most significant:

**OpenAI's Native Function Calling:**
```python
# OpenAI API
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's the weather?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {...}
        }
    }]
)
# Returns structured function call: {"name": "get_weather", "arguments": {...}}
```

**vLLM with Mistral:**
```python
# vLLM (no native function calling)
response = vllm.generate(
    messages=[HumanMessage(content="What's the weather?")],
    # tools parameter is ignored by vLLM engine
)
# Returns plain text: "I need to check the weather for you..."
```

**How CrewAI Bridges This Gap:**

| Aspect | OpenAI (Cloud) | vLLM (Local) |
|--------|---------------|--------------|
| **Tool Definition** | JSON schema in API call | Described in prompt text |
| **Tool Invocation** | LLM returns `function_call` object | LLM returns text mentioning tool |
| **Tool Execution** | API-level handling | Agent executor parses text |
| **Result Passing** | Structured message role | Text appended to conversation |

**Example Flow Comparison:**

**OpenAI + CrewAI:**
```
1. Agent: "Use get_weather for New York"
2. OpenAI API: Returns {function_call: {name: "get_weather", args: {city: "NY"}}}
3. CrewAI: Executes function directly
4. CrewAI: Sends result back as function message
```

**vLLM + CrewAI:**
```
1. Agent: "Use refresh_question_context for Warlock info"
2. vLLM: Returns "I will use the refresh_question_context tool with question='Warlock'..."
3. CrewAI: Parses text to identify tool name and parameters
4. CrewAI: Executes tool based on text parsing
5. CrewAI: Appends result as new text message
```

**Why We Set `supports_function_calling = True`:**
```python
# Even though vLLM doesn't have OpenAI-style function calling:
supports_function_calling: bool = True
```

- This flag tells CrewAI: "This LLM can work with tools"
- It doesn't mean "native function calling like OpenAI"
- It means "can participate in tool-based workflows via text prompting"
- CrewAI adjusts its strategy based on whether native function calling is available

---

#### 4. **Message Format Translation**

**OpenAI's Message Format:**
```python
[
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
]
```

**LangChain's Message Format (used by vLLM):**
```python
[
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="Hello"),
    AIMessage(content="Hi there!")
]
```

**Why Conversion is Necessary:**
- CrewAI sends messages in OpenAI's dict format (industry standard)
- vLLM uses LangChain's object-oriented message types
- Our wrapper must translate between these two formats
- Without translation, vLLM would receive incompatible data types

**The Conversion Code:**
```python
def call(self, messages: str | List[Dict[str, Any]], **kwargs) -> str:
    # CrewAI gives us OpenAI-style dicts
    if isinstance(messages, str):
        lc_messages = [HumanMessage(content=messages)]
    else:
        lc_messages = []
        for msg in messages:
            # Map OpenAI roles → LangChain message types
            if msg['role'] == 'system':
                lc_messages.append(SystemMessage(content=msg['content']))
            elif msg['role'] == 'assistant':
                lc_messages.append(AIMessage(content=msg['content']))
            else:  # 'user' role
                lc_messages.append(HumanMessage(content=msg['content']))
    
    # Now vLLM can understand the messages
    result = self._vllm_chat_model.generate([lc_messages])
```

---

#### 5. **Why a Wrapper is Necessary**

The wrapper exists because of the impedance mismatch between three systems:

```
┌─────────────────────────────────────────────────────────────┐
│                         CrewAI                              │
│                 (Expects OpenAI-like Interface)             │
│  - Dict-based messages                                      │
│  - model parameter required                                 │
│  - tools as JSON schemas                                    │
│  - Returns plain strings                                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   CrewAIVLLMWrapper                         │
│                    (Translation Layer)                      │
│  - Inherits from BaseLLM ✓                                 │
│  - Converts dict → LangChain messages                       │
│  - Provides dummy model name                                │
│  - Handles tool calling via text parsing                    │
│  - Extracts strings from LLM results                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   VLLMChatModel                             │
│                 (LangChain Interface)                       │
│  - Uses LangChain message objects                           │
│  - No model parameter needed (local)                        │
│  - No native tool support                                   │
│  - Returns LLMResult objects                                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      vLLM Engine                            │
│                    (GPU Execution)                          │
│  - Runs Mistral-7B locally                                  │
│  - Generates text tokens                                    │
│  - No API, no authentication                                │
│  - Direct GPU inference                                     │
└─────────────────────────────────────────────────────────────┘
```

**Without the wrapper:**
- CrewAI would send OpenAI-style dicts to vLLM → Type Error
- CrewAI would expect function calling responses → Not Provided
- CrewAI would fail initialization checks → Missing Model Name

**With the wrapper:**
- ✅ CrewAI gets the OpenAI-like interface it expects
- ✅ vLLM gets the LangChain format it understands
- ✅ All validation checks pass
- ✅ Tool-based agents work via text prompting

---

### Key Differences: OpenAI vs vLLM

| Feature | OpenAI + CrewAI | vLLM + CrewAI |
|---------|-----------------|---------------|
| **Hosting** | Cloud API | Local GPU |
| **Authentication** | API key required | No authentication |
| **Cost** | Per-token pricing | Free (hardware cost only) |
| **Latency** | Network + inference | Inference only |
| **Function Calling** | Native support | Emulated via prompting |
| **Message Format** | Direct compatibility | Requires translation |
| **Model Parameter** | Real model name ("gpt-4") | Dummy name ("vllm/mistral") |
| **Tool Execution** | API-level structured calls | Agent-level text parsing |
| **Rate Limits** | Yes (requests/min) | Only GPU memory |
| **Context Length** | Model-specific (e.g., 128K) | Hardware-limited (e.g., 16K) |
| **Integration Effort** | Zero (built-in) | High (custom wrapper) |

---

### Why Not Just Use OpenAI?

Given all these complications, why use vLLM instead of OpenAI?

**Reasons to Use vLLM:**
1. **Privacy**: Data never leaves your infrastructure
2. **Cost**: No per-token charges for high-volume usage
3. **Control**: Full control over model, parameters, and hosting
4. **Offline**: Works without internet connection
5. **Customization**: Can fine-tune models for specific domains
6. **No Rate Limits**: Only limited by your hardware

**Reasons to Use OpenAI:**
1. **Simplicity**: Works out-of-the-box with CrewAI
2. **Power**: Access to GPT-4 and more capable models
3. **Native Function Calling**: Better tool integration
4. **No Infrastructure**: No GPU management needed
5. **Longer Context**: 128K tokens vs 16K typical with vLLM

---

### Summary: OpenAI's Influence on This Integration

1. **CrewAI's API design mirrors OpenAI's patterns** → Wrapper must emulate OpenAI interface
2. **Function calling follows OpenAI's paradigm** → vLLM must emulate it via text parsing
3. **Message format is OpenAI-standard** → Must convert to LangChain format
4. **BaseLLM expects OpenAI-like behavior** → Wrapper bridges the gap
5. **Environment variable only for string model names** → Custom LLMs bypass OpenAI initialization

The integration complexity exists because we're fitting a local, LangChain-based vLLM implementation into a framework designed around OpenAI's cloud API. The wrapper is the adapter that makes this possible.

---

## Architecture Pattern

### Two-Layer Wrapper Design

```
CrewAI Agent → CrewAIVLLMWrapper → VLLMChatModel → vLLM Engine → GPU
     ↓              ↓                    ↓                ↓
  BaseLLM      Message Format      LangChain        Mistral-7B
  Interface    Translation         Interface         Model
```

**Why this design?**
- vLLM was originally built for LangChain integration
- CrewAI has different interface requirements (BaseLLM)
- The wrapper bridges these two ecosystems without rewriting vLLM code

---

## Critical Rules

### 1. **Must Inherit from CrewAI's BaseLLM**

```python
from crewai.llms.base_llm import BaseLLM as CrewAIBaseLLM

class CrewAIVLLMWrapper(CrewAIBaseLLM):
    # Your implementation
```

**Why:** CrewAI agents only recognize LLMs that inherit from `BaseLLM`. Without this, CrewAI will reject your custom LLM.

**Consequence if violated:** `TypeError` or agent initialization failure.

---

### 2. **Must Implement the `call()` Method**

```python
def call(
    self,
    messages: str | List[Dict[str, Any]],
    tools: List[Dict[str, Any]] | None = None,
    callbacks: List[Any] | None = None,
    available_functions: Dict[str, Any] | None = None,
    from_task: Any = None,
    from_agent: Any = None,
    response_model: Any = None,
) -> str:
    # Your implementation must return a string
```

**Why:** This is the core method CrewAI agents use to communicate with LLMs. The signature must match exactly.

**Consequence if violated:** `AttributeError` or incorrect method signature errors.

---

### 3. **Must Provide a Model Name to BaseLLM**

```python
super().__init__(
    model="vllm/mistral",  # Required, even if not used
    **kwargs
)
```

**Why:** CrewAI's `BaseLLM.__init__()` requires a model parameter for internal bookkeeping.

**For local models:** Use a descriptive identifier like `"vllm/mistral"` or `"local/model-name"`.

**Consequence if violated:** `TypeError: missing required positional argument: 'model'`

---

### 4. **Message Format Conversion is Mandatory**

CrewAI sends messages as:
- Simple strings: `"What is a warlock?"`
- Message dicts: `[{"role": "user", "content": "Hello"}]`

vLLM expects:
- LangChain message objects: `[HumanMessage(content="Hello")]`

**Required conversion:**
```python
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Convert CrewAI format → LangChain format
if isinstance(messages, str):
    lc_messages = [HumanMessage(content=messages)]
else:
    lc_messages = []
    for msg in messages:
        if msg['role'] == 'system':
            lc_messages.append(SystemMessage(content=msg['content']))
        elif msg['role'] == 'assistant':
            lc_messages.append(AIMessage(content=msg['content']))
        else:
            lc_messages.append(HumanMessage(content=msg['content']))
```

**Consequence if violated:** Type errors or vLLM unable to process messages.

---

### 5. **Tool Calling Requires Special Handling**

**CRITICAL DIFFERENCE:** vLLM does NOT have native function calling like GPT-4 or Claude.

| LLM Type | How Tools Work |
|----------|---------------|
| **OpenAI/Anthropic** | Tools passed as API parameters; model returns structured function calls |
| **vLLM (Custom)** | Tools described in prompts; CrewAI parses text responses for tool usage |

**Implementation requirements:**

```python
class CrewAIVLLMWrapper(CrewAIBaseLLM):
    supports_function_calling: bool = True  # MUST set to True
```

**Why set to True if vLLM doesn't have native function calling?**
- CrewAI checks this flag to enable tool-based agents
- Setting to `True` tells CrewAI: "Handle tools via prompting"
- CrewAI's agent executor will:
  1. Include tool descriptions in the prompt
  2. Parse LLM's text response for tool invocations
  3. Execute tools and feed results back

**Consequence if set to False:** CrewAI will not allow agents to use tools with this LLM.

---

### 6. **Environment Variables with Custom LLMs**

When using custom LLM instances, no OpenAI-related environment variables are needed.

```python
# ✅ Clean setup with custom LLM - no env vars required
llm = create_crewai_vllm_model(download_dir="./models")
agent = Agent(role="Assistant", goal="Help", backstory="...", llm=llm)
```

**When you need OPENAI_API_KEY:**
- Only when using string model names: `llm="gpt-4"`, `llm="gpt-3.5-turbo"`, etc.
- CrewAI uses the key to authenticate with OpenAI's API
- Custom LLM instances completely bypass this code path

**Best Practice:**
- Don't set environment variables you don't need
- Only configure what your specific setup requires
- With pure custom LLM usage, no OpenAI configuration is necessary

---

## Important Limitations

### 1. **No Native Structured Output**

Unlike GPT-4 with JSON mode or function calling, vLLM with Mistral must generate structured output via:
- Careful prompting
- Output parsers (Pydantic)
- Post-processing

**Example:**
```python
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel

parser = PydanticOutputParser(pydantic_object=YourSchema)
prompt = f"...\n{parser.get_format_instructions()}"
```

### 2. **GPU Memory Management**

vLLM holds GPU memory that Python's garbage collector won't release:

```python
# Always cleanup when done
llm.cleanup_llm_memory()
force_gpu_memory_cleanup()  # If available
```

**When to cleanup:**
- End of script execution
- Before loading a different model
- In `finally` blocks for reliability

### 3. **Context Length Limitations**

vLLM models have hard context limits:
- Mistral-7B: typically 8192 or 16384 tokens
- Exceeding this causes errors or truncation

**Monitor your context:**
```python
max_model_len: int = 16384  # Set during initialization
```

### 4. **Batch Processing Differences**

The `generate()` method processes sequentially, not in true parallel:

```python
def generate(self, prompts: List[str], **kwargs) -> List[str]:
    return [self.call(messages=prompt, **kwargs) for prompt in prompts]
```

**Why:** Simplifies the wrapper; vLLM handles internal batching.

---

## Required Dependencies

### Minimum Requirements

```txt
crewai>=0.28.0
langchain-core>=0.1.0
vllm>=0.2.0
torch>=2.0.0
transformers>=4.35.0
```

### GPU Requirements

- NVIDIA GPU with CUDA support
- Minimum 8GB VRAM (for Mistral-7B)
- 16GB+ VRAM recommended for larger contexts

---

## Setup Checklist

- [ ] Install all required dependencies
- [ ] Verify GPU availability (`nvidia-smi`)
- [ ] Create wrapper inheriting from `CrewAI BaseLLM`
- [ ] Implement `call()` method with correct signature
- [ ] Convert message formats (CrewAI → LangChain)
- [ ] Set `supports_function_calling = True`
- [ ] Implement cleanup methods
- [ ] Test with simple agent before complex workflows

**Note:** `OPENAI_API_KEY` environment variable is NOT needed for custom LLM instances.

---

## Common Pitfalls

### ❌ **Pitfall 1: Setting Unnecessary Environment Variables**

```python
# ❌ UNNECESSARY - Adds confusion for custom LLMs
import os
os.environ["OPENAI_API_KEY"] = "dummy-key"
llm = create_crewai_vllm_model()
agent = Agent(role="...", goal="...", backstory="...", llm=llm)
```

```python
# ✅ CORRECT - Clean setup without unnecessary config
llm = create_crewai_vllm_model()
agent = Agent(role="...", goal="...", backstory="...", llm=llm)
```

**Why to avoid this:**
- Only needed when using string model names (e.g., `llm="gpt-4"`)
- Custom LLM instances don't use OpenAI's API
- Setting unnecessary variables creates confusion about requirements
- Keep your configuration clean and minimal

---

### ❌ **Pitfall 2: Forgetting Message Conversion**

```python
# WRONG - vLLM won't understand CrewAI's dict format
result = vllm_model.generate([messages])
```

```python
# CORRECT - Convert to LangChain format first
lc_messages = [HumanMessage(content=msg['content']) for msg in messages]
result = vllm_model.generate([lc_messages])
```

---

### ❌ **Pitfall 3: Not Setting Function Calling Support**

```python
# WRONG - Tools won't work
class Wrapper(CrewAIBaseLLM):
    pass  # supports_function_calling defaults to False
```

```python
# CORRECT - Explicitly enable
class Wrapper(CrewAIBaseLLM):
    supports_function_calling: bool = True
```

---

### ❌ **Pitfall 4: Forgetting GPU Cleanup**

```python
# WRONG - Memory leak
llm = create_crewai_vllm_model()
# ... use llm ...
# Script ends without cleanup
```

```python
# CORRECT - Always cleanup
try:
    llm = create_crewai_vllm_model()
    # ... use llm ...
finally:
    llm.cleanup_llm_memory()
```

---

### ❌ **Pitfall 5: Expecting Native Function Calling**

```python
# WRONG - vLLM doesn't return structured function calls
def call(self, messages, tools, **kwargs):
    result = self._vllm.generate(messages, tools=tools)  # tools parameter ignored
```

```python
# CORRECT - Let CrewAI handle tools via prompting
def call(self, messages, tools, **kwargs):
    # Tools are described in the prompt by CrewAI's agent
    # Just generate text; CrewAI parses it for tool usage
    result = self._vllm.generate(messages)
```

---

## Example: Minimal Working Implementation

```python
from crewai import Agent, Task, Crew
from inference.vllm_srv.minstral_crewai import create_crewai_vllm_model

# 1. Create CrewAI-compatible vLLM wrapper (no env vars needed)
llm = create_crewai_vllm_model(
    download_dir="./models",
    gpu_memory_utilization=0.75,
    max_model_len=16384
)

# 2. Create agent with vLLM
agent = Agent(
    role="Assistant",
    goal="Help users",
    backstory="A helpful AI assistant",
    llm=llm,  # Use our local vLLM
    verbose=True
)

# 3. Create and execute task
task = Task(
    description="Explain what a warlock is in D&D",
    expected_output="A clear explanation",
    agent=agent
)

# 4. Run the crew
crew = Crew(agents=[agent], tasks=[task], verbose=True)

try:
    result = crew.kickoff()
    print(result)
finally:
    # 5. Always cleanup GPU memory
    llm.cleanup_llm_memory()
```

---

## Debugging Tips

### Check GPU Status
```bash
nvidia-smi
watch -n 1 nvidia-smi  # Monitor in real-time
```

### Enable Verbose Logging
```python
agent = Agent(
    # ...
    verbose=True  # Shows reasoning steps and tool calls
)

crew = Crew(
    # ...
    verbose=True  # Shows task execution flow
)
```

### Verify Message Format
```python
def call(self, messages, **kwargs):
    logger.info(f"Received messages: {messages}")
    logger.info(f"Message type: {type(messages)}")
    # ... rest of implementation
```

### Test Incrementally
1. Test vLLM model alone (without CrewAI)
2. Test wrapper with simple string prompt
3. Test wrapper with message list
4. Test agent without tools
5. Test agent with tools

---

## Performance Considerations

### GPU Memory Allocation
```python
llm = create_crewai_vllm_model(
    gpu_memory_utilization=0.75,  # Leave 25% for other processes
)
```

**Recommendations:**
- Single GPU: 0.85-0.90
- Shared GPU: 0.60-0.75
- Development: 0.50-0.70

### Context Length vs Speed
Longer contexts require more memory and processing time:
- 4K tokens: Fast, moderate memory
- 8K tokens: Balanced
- 16K tokens: Slower, high memory

### Response Length
```python
llm = create_crewai_vllm_model(
    max_tokens=512,  # Limit response length
)
```

Shorter responses = faster generation

---

## Testing and Verification

### Verify Custom LLM Works Without OPENAI_API_KEY

To confirm that your setup doesn't require the environment variable:

```python
import os

# Explicitly remove the key if it exists
if 'OPENAI_API_KEY' in os.environ:
    del os.environ['OPENAI_API_KEY']

# Import and create your vLLM wrapper
from inference.vllm_srv.minstral_crewai import create_crewai_vllm_model
from crewai import Agent, Task, Crew

# This should work without any errors
llm = create_crewai_vllm_model(download_dir="./models")

agent = Agent(
    role="Test Agent",
    goal="Verify setup",
    backstory="Testing that custom LLM works without OpenAI key",
    llm=llm
)

task = Task(
    description="Say hello",
    expected_output="A greeting",
    agent=agent
)

crew = Crew(agents=[agent], tasks=[task], verbose=True)

try:
    result = crew.kickoff()
    print("✅ SUCCESS: Custom LLM works without OPENAI_API_KEY!")
    print(f"Result: {result}")
finally:
    llm.cleanup_llm_memory()
```

If this runs successfully, your setup is correct and doesn't need the environment variable.

### Quick Health Check

```python
# Minimal test to verify wrapper functionality
def test_vllm_wrapper():
    from inference.vllm_srv.minstral_crewai import create_crewai_vllm_model
    
    print("1. Creating vLLM wrapper...")
    llm = create_crewai_vllm_model(
        download_dir="./models",
        max_tokens=50  # Short for quick test
    )
    
    print("2. Testing direct call...")
    response = llm.call("Say hello in one sentence.")
    print(f"   Response: {response}")
    
    print("3. Testing message list format...")
    response = llm.call([
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hi"}
    ])
    print(f"   Response: {response}")
    
    print("4. Cleaning up...")
    llm.cleanup_llm_memory()
    
    print("✅ All tests passed!")

# Run the test
test_vllm_wrapper()
```

---

## Summary

**Key Requirements:**

1. ✅ **Must inherit from CrewAI BaseLLM**
2. ✅ **Must implement call() method with exact signature**
3. ✅ **Must convert CrewAI messages to LangChain format**
4. ✅ **Must set supports_function_calling = True**
5. ✅ **Must cleanup GPU memory when done**
6. ⚠️ **Tool calling works via prompting, not native function calls**
7. ⚠️ **No native structured output - use parsers**
8. ⚠️ **Context length limits are hard constraints**

**Environment Variables:**
- `OPENAI_API_KEY` is **NOT required** when using custom LLM instances
- Only needed when using string model names like `llm="gpt-4"`
- Modern CrewAI versions (0.28.0+) properly handle custom LLMs

**When in doubt:** Check the wrapper implementation in `minstral_crewai.py` for reference.

---

## Additional Resources

- [CrewAI Documentation](https://docs.crewai.com/)
- [vLLM Documentation](https://docs.vllm.ai/)
- [LangChain Documentation](https://python.langchain.com/)
- Project file: `src/inference/vllm_srv/minstral_crewai.py`
- Example usage: `src/experiments/crewai_llm_agent_test.py`

---

## Document Version History

### Version 2.0 (January 14, 2026)
**Corrected and Clarified Requirements**

- ✅ **Verified via testing** that `OPENAI_API_KEY` is NOT required for custom LLMs
- 🧹 **Removed all debunked/false statements** from documentation
- 📝 **Streamlined content** to present only accurate, tested information
- ➕ **Added testing verification** section to help users confirm their setup
- 🔄 **Reorganized pitfalls** with cleaner numbering and better explanations
- 📊 **Clarified when environment variables are needed** (string model names only)

**Key Clarification:** The environment variable is only required when using string model names (e.g., `llm="gpt-4"`), not when passing custom LLM instances.

### Version 1.0 (January 14, 2026)
- Initial documentation created
- Covered architecture, rules, limitations, and examples


