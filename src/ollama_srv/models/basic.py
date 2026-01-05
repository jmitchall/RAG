#!/usr/bin/env python3
"""
AI Language Model Demo Script
============================

This script demonstrates how to use the LiteLLM library to run large language models (LLMs)
locally on your computer. It's like having ChatGPT running on your own machine!

What this script does:
1. Configures an AI language model connection
2. Sends several questions/prompts to the AI
3. Gets responses back from the AI
4. Prints out both the questions and AI's answers

Requirements:
- A computer with a decent graphics card (GPU) for local models, or API keys for cloud models
- Python environment with LiteLLM installed
- For some models: HuggingFace account and authentication or API keys

Author: Educational Demo
License: Apache-2.0
"""

# Import the main library we need from LiteLLM
from litellm import completion
import os

# =============================================================================
# CONFIGURATION SECTION - Choose your questions and AI model settings
# =============================================================================

# These are the questions we'll ask the AI
prompts = [
    "Hello, can you introduce yourself?",  # Simple greeting
    "What is the capital of France and why is it important?",  # Factual question
    "Explain the future of AI in 2-3 sentences.",  # Opinion question
    "Write a short Python function to calculate fibonacci numbers.",  # Programming task
]

# These settings control HOW the AI generates responses
generation_params = {
    "temperature": 0.7,  # How "creative" the AI is (0.0 = very predictable, 1.0 = very creative)
    "top_p": 0.9,  # Another creativity control (0.1 = focused, 1.0 = considers all options)
    "max_tokens": 256,  # Maximum length of each AI response (in "tokens" - roughly words)
    "frequency_penalty": 0.1  # Prevents the AI from repeating itself too much
}


# =============================================================================
# MAIN FUNCTION - This is where the magic happens!
# =============================================================================

def main():
    """
    Main function that runs our AI demo.
    This function configures an AI model and asks it questions.
    """

    print("🤖 Starting AI Language Model Demo...")
    print("=" * 50)

    # =================================================================
    # STEP 1: Choose and configure an AI model
    # =================================================================

    # Here we choose which AI model to use. LiteLLM supports many providers
    # and can run local models via Ollama or use cloud APIs.

    # OPTION 1: Local model via Ollama (free, runs on your computer)
    model = "ollama/phi3"  # Phi-3 mini model via Ollama
    # model = "ollama/mistral"  # Mistral 7B via Ollama
    # model = "ollama/mixtral"  # Mixtral 8x7B via Ollama

    # OPTION 2: OpenAI models (requires API key)
    # os.environ["OPENAI_API_KEY"] = "your-api-key-here"
    # model = "gpt-3.5-turbo"
    # model = "gpt-4"

    # OPTION 3: Anthropic Claude (requires API key)
    # os.environ["ANTHROPIC_API_KEY"] = "your-api-key-here"
    # model = "claude-3-sonnet-20240229"

    # OPTION 4: Local HuggingFace model via LiteLLM
    # model = "huggingface/microsoft/Phi-3-mini-4k-instruct"

    print(f"✅ AI model configured: {model}")

    # =================================================================
    # STEP 2: Ask the AI all our questions
    # =================================================================

    print(f"\n🗣️  Asking {len(prompts)} questions to the AI...")
    print("⏳ This might take a minute - the AI is thinking...")

    responses = []

    # Process each question one by one
    for prompt in prompts:
        # Format the prompt as a chat message
        messages = [
            {"role": "user", "content": prompt}
        ]

        # Send the question to the AI
        response = completion(
            model=model,
            messages=messages,
            **generation_params
        )

        # Store the response
        responses.append({
            "prompt": prompt,
            "response": response.choices[0].message.content
        })

    # =================================================================
    # STEP 3: Display the results in a nice format
    # =================================================================

    print("\n🎯 AI Responses:")
    print("=" * 80)

    # Loop through each question-answer pair
    for i, item in enumerate(responses, 1):  # enumerate gives us a counter starting from 1

        # Extract the original question we asked
        original_question = item["prompt"]

        # Extract the AI's response
        ai_response = item["response"]

        # Display everything in a nice format
        print(f"\n📝 Question #{i}:")
        print(f"❓ Human: {original_question}")
        print(f"🤖 AI: {ai_response}")
        print("-" * 80)  # Visual separator line

    print("\n✨ Demo completed! The AI has answered all your questions.")
    print("💡 Try editing the 'prompts' list above to ask different questions!")
    print("💡 Or change the 'model' variable to try different AI models!")


# =============================================================================
# PROGRAM ENTRY POINT
# =============================================================================

# This special Python code means "only run main() if this file is executed directly"
# (not if it's imported by another Python file)
if __name__ == "__main__":
    # Start the program by calling our main function
    main()