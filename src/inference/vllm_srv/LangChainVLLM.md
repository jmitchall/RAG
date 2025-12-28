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
