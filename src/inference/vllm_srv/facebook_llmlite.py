#!/usr/bin/env python3
# Import the main libraries we need from vLLM
from vllm import LLM, SamplingParams  # LLM = Large Language Model, SamplingParams = AI response settings
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
    temperature=0.3,        # 👈 LOWERED: 0.3 = less random, more coherent (0.7 was too high for small models)
    top_p=0.9,              # Nucleus sampling
    top_k=50,               # 👈 ADDED: Limit to top 50 tokens (prevents garbage/whitespace tokens)
    max_tokens=256,
    repetition_penalty=1.05 # 👈 LOWERED: 1.05 is gentler (1.1 can over-penalize and cause issues)
)

# MODE 2: DETERMINISTIC OUTPUT (Same every time)
# - Use for consistent results, testing, factual queries
# - Each run produces IDENTICAL output
sampling_params_deterministic = SamplingParams(
    temperature=0.0,        # 👈 No randomness = greedy decoding (always pick most likely token)
    top_p=0.9,              # Not used when temperature=0
    max_tokens=256,
    repetition_penalty=1.05, # 👈 LOWERED: 1.05 is gentler (1.1 can over-penalize and cause issues)
    seed=42                 # 👈 Fixed random seed for reproducibility
)

# Choose which mode to use:
sampling_params = sampling_params_random  # 👈 Change this to switch modes


def get_vllm_facebook_opt_125m(download_dir=None):
    model_name = "facebook/opt-125m"
    # OPTION 1: Small, fast model (works on any computer with basic GPU)
    if download_dir is None:
        llm = LLM(model=model_name)  # Only 125 million parameters
    else:
        # ALTERNATIVE Load. Load and save model locally
        llm = LLM(model=model_name, download_dir=download_dir)
        
    # Only 125 million parameters, if "./models/opt-125m" does not exist, 
    # it will be downloaded and saved there. 
    print(f"✅ AI model loaded successfully!")
    print(f"📊 Model: {llm.llm_engine.model_config.model}")
    return llm

def facebook_opt_125m(download_dir="./models"):
    """
    Main function that runs our AI demo.
    This function loads an AI model and asks it questions.
    """
    # These are the questions we'll ask the AI
    prompts = [
    "Hello, can you introduce yourself?",  # Simple greeting
    "What is the capital of France and why is it important?",  # Factual question
    "Explain the future of AI in 2-3 sentences.",  # Opinion question
    "Write a short Python function to calculate fibonacci numbers.",  # Programming task
]
    llm= get_vllm_facebook_opt_125m(download_dir=download_dir)
    # =================================================================
    # STEP 2: Ask the AI all our questions
    # =================================================================

    print(f"\n🗣️  Asking {len(prompts)} questions to the AI...")
    print("⏳ This might take a minute - the AI is thinking...")

    # This is where we actually send our questions to the AI
    # The AI processes ALL questions at once (called "batch processing")
    outputs = llm.generate(prompts, sampling_params)

    # =================================================================
    # STEP 3: Display the results in a nice format
    # =================================================================

    print("\n🎯 AI Responses:")
    print("=" * 80)

    # Loop through each question-answer pair
    for i, output in enumerate(outputs, 1):  # enumerate gives us a counter starting from 1

        # Extract the original question we asked
        original_question = output.prompt

        # Extract the AI's response (it might have multiple responses, we take the first one)
        ai_response = output.outputs[0].text

        # Display everything in a nice format
        print(f"\n📝 Question #{i}:")
        print(f"❓ Human: {original_question}")
        print(f"🤖 AI: {ai_response}")
        print("-" * 80)  # Visual separator line

    print("\n✨ Demo completed! The AI has answered all your questions.")
    print("💡 Try editing the 'prompts' list above to ask different questions!")