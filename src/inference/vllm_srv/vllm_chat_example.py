#!/usr/bin/env python3
"""
vLLM Chat Model Example with Tool Calling Support

Author: Jonathan A. Mitchall
Version: 1.0
Last Updated: January 10, 2026

License: MIT License

Copyright (c) 2026 Jonathan A. Mitchall

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Revision History:
    2026-01-10 (v1.0): Initial comprehensive documentation

Example usage of the VLLMChatModel with tool calling support.

This demonstrates how to use the custom chat model wrapper that enables
"""
tool calling functionality with VLLM and Mistral models.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from refection_logger import logger
from minstral_langchain import create_vllm_chat_model


# Define some example tools
@tool
def get_weather(location: str) -> str:
    """Get the current weather for a given location.
    
    Args:
        location: The city and country/state to get weather for
    """
    # Mock weather data - in real usage this would call a weather API
    weather_data = {
        "san francisco": "Sunny, 72°F",
        "new york": "Cloudy, 65°F",
        "london": "Rainy, 55°F",
        "tokyo": "Clear, 68°F"
    }

    location_key = location.lower()
    for key in weather_data:
        if key in location_key:
            return f"Weather in {location}: {weather_data[key]}"

    return f"Weather data not available for {location}"


@tool
def calculate(expression: str) -> str:
    """Safely calculate a mathematical expression.
    
    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 2", "10 * 5")
    """
    try:
        # Safe evaluation of basic mathematical expressions
        allowed_names = {
            k: v for k, v in __builtins__.items()
            if k in ('abs', 'round', 'min', 'max', 'sum', 'len')
        }
        allowed_names.update({
            'pow': pow, 'sqrt': lambda x: x ** 0.5
        })

        # Basic safety check
        if any(char in expression for char in ['import', 'exec', 'eval', '__']):
            return "Error: Invalid expression"

        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"{expression} = {result}"

    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"


def main():
    """Main example function."""

    logger.info("🚀 Creating VLLM Chat Model with tool calling support...")

    # Create the chat model with tool calling support
    # Adjust parameters based on your GPU capabilities
    chat_model = create_vllm_chat_model(
        download_dir="./models",  # Change this to your model directory
        gpu_memory_utilization=0.8,
        max_model_len=8192,
        max_tokens=256,
        temperature=0.7
    )

    logger.info("✅ Model created successfully!")

    # Define tools to bind
    tools = [get_weather, calculate]

    # Bind tools to the model
    model_with_tools = chat_model.bind_tools(tools)

    logger.info("🔧 Tools bound to model")

    # Example 1: Simple question without tools
    logger.info("\n" + "=" * 50)
    logger.info("Example 1: Simple conversation")
    logger.info("=" * 50)

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Hello! How are you today?")
    ]

    response = model_with_tools.invoke(messages)
    logger.info(f"Assistant: {response.content}")

    # Example 2: Weather tool calling
    logger.info("\n" + "=" * 50)
    logger.info("Example 2: Weather tool calling")
    logger.info("=" * 50)

    messages = [
        SystemMessage(content="You are a helpful assistant that can check weather."),
        HumanMessage(content="What's the weather like in San Francisco?")
    ]

    response = model_with_tools.invoke(messages)
    logger.info(f"Assistant: {response.content}")

    if response.tool_calls:
        logger.info(f"Tool calls made: {len(response.tool_calls)}")
        for i, tool_call in enumerate(response.tool_calls):
            logger.info(f"  {i + 1}. {tool_call['name']} with args: {tool_call['args']}")

            # Execute the tool (in real usage, you'd handle this in an agent loop)
            if tool_call['name'] == 'get_weather':
                result = get_weather(**tool_call['args'])
                logger.info(f"     Result: {result}")

    # Example 3: Math tool calling
    logger.info("\n" + "=" * 50)
    logger.info("Example 3: Math calculation")
    logger.info("=" * 50)

    messages = [
        SystemMessage(content="You are a helpful assistant that can perform calculations."),
        HumanMessage(content="Can you calculate what 15 * 23 + 7 equals?")
    ]

    response = model_with_tools.invoke(messages)
    logger.info(f"Assistant: {response.content}")

    if response.tool_calls:
        logger.info(f"Tool calls made: {len(response.tool_calls)}")
        for i, tool_call in enumerate(response.tool_calls):
            logger.info(f"  {i + 1}. {tool_call['name']} with args: {tool_call['args']}")

            if tool_call['name'] == 'calculate':
                result = calculate(**tool_call['args'])
                logger.info(f"     Result: {result}")

    # Example 4: Multiple tool usage
    logger.info("\n" + "=" * 50)
    logger.info("Example 4: Multiple tool usage")
    logger.info("=" * 50)

    messages = [
        SystemMessage(content="You are a helpful assistant with access to weather and calculation tools."),
        HumanMessage(
            content="I'm planning a trip to Tokyo. Can you check the weather there and also calculate how much I'd spend if I budget $50 per day for 7 days?")
    ]

    response = model_with_tools.invoke(messages)
    logger.info(f"Assistant: {response.content}")

    if response.tool_calls:
        logger.info(f"Tool calls made: {len(response.tool_calls)}")
        for i, tool_call in enumerate(response.tool_calls):
            logger.info(f"  {i + 1}. {tool_call['name']} with args: {tool_call['args']}")

            if tool_call['name'] == 'get_weather':
                result = get_weather(**tool_call['args'])
                logger.info(f"     Weather result: {result}")
            elif tool_call['name'] == 'calculate':
                result = calculate(**tool_call['args'])
                logger.info(f"     Calculation result: {result}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.info(f"❌ Error running example: {e}")
        logger.info("Make sure you have VLLM installed and a compatible GPU available.")
