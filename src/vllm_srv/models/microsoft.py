#!/usr/bin/env python3
"""
AI Language Model Demo Script
============================

This script demonstrates how to use the vLLM library to run large language models (LLMs) 
locally on your computer. It's like having ChatGPT running on your own machine!

What this script does:
1. Loads an AI language model into your computer's memory
2. Sends several questions/prompts to the AI
3. Gets responses back from the AI
4. Prints out both the questions and AI's answers

Requirements:
- A computer with a decent graphics card (GPU) 
- Python environment with vLLM installed
- For some models: HuggingFace account and authentication

Author: Educational Demo
License: Apache-2.0
"""

# Import the main libraries we need from vLLM
from vllm import LLM, SamplingParams  # LLM = Large Language Model, SamplingParams = AI response settings

# =============================================================================
# CONFIGURATION SECTION - Choose your questions and AI model settings
# =============================================================================

# These are the questions we'll ask the AI
# The [INST] and [/INST] tags tell the AI "this is an instruction for you"
prompts = [
    "[INST] Hello, can you introduce yourself? [/INST]",  # Simple greeting
    "[INST] What is the capital of France and why is it important? [/INST]",  # Factual question
    "[INST] Explain the future of AI in 2-3 sentences. [/INST]",  # Opinion question
    "[INST] Write a short Python function to calculate fibonacci numbers. [/INST]",  # Programming task
]

# These settings control HOW the AI generates responses
sampling_params = SamplingParams(
    temperature=0.7,  # How "creative" the AI is (0.0 = very predictable, 1.0 = very creative)
    top_p=0.9,  # Another creativity control (0.1 = focused, 1.0 = considers all options)
    max_tokens=256,  # Maximum length of each AI response (in "tokens" - roughly words)
    repetition_penalty=1.1  # Prevents the AI from repeating itself too much
)


# =============================================================================
# MAIN FUNCTION - This is where the magic happens!
# =============================================================================

def main():
    """
    Main function that runs our AI demo.
    This function loads an AI model and asks it questions.
    """

    print("🤖 Starting AI Language Model Demo...")
    print("=" * 50)

    # =================================================================
    # STEP 1: Choose and load an AI model
    # =================================================================

    # Here we choose which AI model to use. Think of this like choosing 
    # which "brain" you want to talk to. Different models have different
    # capabilities and memory requirements.

    # OPTION 2: Medium quality model (works great, no authentication needed)
    llm = LLM(
        model="microsoft/Phi-3-mini-4k-instruct",  # 3.8 billion parameters
        gpu_memory_utilization=0.9,  # Use 90% of graphics card memory
        max_model_len=4096,  # Handle conversations up to 4096 words
        dtype="float16",  # Use half-precision numbers (saves memory)
        trust_remote_code=True,  # Allow the model to run custom code (needed for Phi-3)
        model_path="./models/phi-3-mini-4k-instruct"  # Optional: specify local path to save/load model
    )

    print(f"✅ AI model loaded successfully!")
    print(f"📊 Model: {llm.llm_engine.model_config.model}")

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


# =============================================================================
# PROGRAM ENTRY POINT
# =============================================================================

# This special Python code means "only run main() if this file is executed directly"
# (not if it's imported by another Python file)
if __name__ == "__main__":
    # Start the program by calling our main function
    main()
