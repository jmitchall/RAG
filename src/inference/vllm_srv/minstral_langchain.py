# Import the main libraries we need from vLLM
from langchain_community.llms import VLLM
from langchain_core.prompt_values import StringPromptValue

def convert_chat_prompt_to_minstral_prompt_value(chat_prompt_value):
    """
    Convert ChatPromptValue to a plain string by concatenating message contents.
    This removes role labels like "System:" and "Human:" for BaseLLM compatibility.
    
    IMPORTANT: This concatenates messages WITHOUT adding separators, because
    the messages themselves already contain the necessary newlines/formatting.
    
    Example:
        SystemMessage: "You are helpful..."
        HumanMessage:  "\n\nWhat is the capital..."
        Result:        "You are helpful...\n\nWhat is the capital..."
    
    Args:
        chat_prompt_value: ChatPromptValue object from ChatPromptTemplate
    Returns:
        string_prompt_value: StringPromptValue Mistral format: <s>[INST] {instruction} [/INST]
    """
    # Concatenate all message contents WITHOUT a separator
    # The messages already contain their own formatting/newlines
    combined_content = "\n\n".join([message.content for message in chat_prompt_value.messages])
    prompt = f"<s>[INST] {combined_content} [/INST]"
    # Create and return a StringPromptValue
    return StringPromptValue(text=prompt)


def get_langchain_vllm_mistral_quantized(download_dir=None , gpu_memory_utilization: float = 0.85,
                                         max_model_len: int = 16384,  top_k: int = 5,
                     max_tokens: int = 512, temperature: float = 0.7) -> VLLM:
        model_name="TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
        if download_dir is None:
            return VLLM(
            model=model_name,
            max_new_tokens= max_tokens,
            top_k= top_k,
            temperature= temperature,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,  # Reduced from 0.95 to fit available memory
            max_model_len=max_model_len,  # Full context window
            quantization="gptq",
            dtype="half",
            trust_remote_code=True
            )
        else:
            return VLLM(model=model_name, download_dir=download_dir,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,  # Reduced from 0.95 to fit available memory
            max_model_len=max_model_len,  # Full context window
            quantization="gptq",
            dtype="half",
            trust_remote_code=True
            )

