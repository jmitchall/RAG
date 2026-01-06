#!/usr/bin/env python3
# Import the main libraries we need from vLLM
from langchain_community.llms import VLLM
from langchain_core.prompt_values import StringPromptValue
from vllm import SamplingParams  # LLM = Large Language Model, SamplingParams = AI response settings

# =============================================================================
# CONFIGURATION SECTION - Choose your questions and AI model settings
# =============================================================================

# These settings control HOW the AI generates responses
# =============================================================================
# SAMPLING SETTINGS: Random vs Deterministic
# =============================================================================
# Two modes available:

# MODE 1: CREATIVE/RANDOM OUTPUT (Different every time)
# - Use for creative writing, brainstorming, varied responses
# - Each run produces different output
sampling_params_random = SamplingParams(
    temperature=0.3,  # 👈 LOWERED: 0.3 = less random, more coherent (0.7 was too high for small models)
    top_p=0.9,  # Nucleus sampling
    top_k=50,  # 👈 ADDED: Limit to top 50 tokens (prevents garbage/whitespace tokens)
    max_tokens=256,
    repetition_penalty=1.05  # 👈 LOWERED: 1.05 is gentler (1.1 can over-penalize and cause issues)
)

# MODE 2: DETERMINISTIC OUTPUT (Same every time)
# - Use for consistent results, testing, factual queries
# - Each run produces IDENTICAL output
sampling_params_deterministic = SamplingParams(
    temperature=0.0,  # 👈 No randomness = greedy decoding (always pick most likely token)
    top_p=0.9,  # Not used when temperature=0
    max_tokens=256,
    repetition_penalty=1.05,  # 👈 LOWERED: 1.05 is gentler (1.1 can over-penalize and cause issues)
    seed=42  # 👈 Fixed random seed for reproducibility
)

# Choose which mode to use:
sampling_params = sampling_params_random  # 👈 Change this to switch modes


def get_langchain_vllm_facebook_opt_125m(download_dir=None):
    model_name = "facebook/opt-125m"
    # OPTION 1: Small, fast model (works on any computer with basic GPU)
    if download_dir is None:
        llm = VLLM(model=model_name,
                   max_new_tokens=sampling_params.max_tokens,
                   temperature=sampling_params.temperature,
                   top_p=sampling_params.top_p,
                   top_k=sampling_params.top_k,
                   presence_penalty=sampling_params.presence_penalty
                   )  # Only 125 million parameters

    else:
        # ALTERNATIVE Load. Load and save model locally
        llm = VLLM(model=model_name, download_dir=download_dir,
                   max_new_tokens=sampling_params.max_tokens,
                   temperature=sampling_params.temperature,
                   top_p=sampling_params.top_p,
                   top_k=sampling_params.top_k,
                   presence_penalty=sampling_params.presence_penalty
                   )  # Only 125 million parameters

    # Only 125 million parameters, if "./models/opt-125m" does not exist, 
    # it will be downloaded and saved there. 
    print(f"✅ AI model loaded successfully!")
    print(f"📊 Model: {llm}")
    return llm


def convert_chat_prompt_to_facebook_prompt_value(chat_prompt_value):
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
        string_prompt_value: StringPromptValue with concatenated message contents
    """
    # Concatenate all message contents WITHOUT a separator
    # The messages already contain their own formatting/newlines
    combined_content = "".join([message.content for message in chat_prompt_value.messages])

    # Create and return a StringPromptValue
    return StringPromptValue(text=combined_content)
