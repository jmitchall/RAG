# Import the main libraries we need from vLLM
import json

import re
from inference.vllm_srv.cleaner import check_gpu_memory_status, force_gpu_memory_cleanup
from inference.vllm_srv.cleaner import cleanup_vllm_engine
from langchain_community.llms import VLLM
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompt_values import StringPromptValue, ChatPromptValue
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field
from typing import Any, Dict, List, Optional, Sequence, Union, Callable
from refection_logger import logger


def messages_to_mistral_prompt(messages: Sequence[BaseMessage]) -> str:
    """Convert LangChain messages to Mistral prompt format."""

    # Separate system messages from conversation
    system_parts = []
    conversation_parts = []

    for message in messages:
        match message:
            case SystemMessage():
                system_parts.append(message.content)

            case HumanMessage():
                conversation_parts.append(f"Human: {message.content}")

            case AIMessage() if message.tool_calls:
                # Format AI message with tool calls
                tool_calls_text = format_tool_calls_for_prompt(message.tool_calls)
                conversation_parts.append(f"Assistant: {message.content}\n{tool_calls_text}")

            case AIMessage():
                conversation_parts.append(f"Assistant: {message.content}")

            case ToolMessage():
                # Format tool results
                conversation_parts.append(f"Tool Result ({message.name}): {message.content}")

            case _:
                # Handle any unexpected message types gracefully
                logger.warning(f"Unknown message type: {type(message)}")
                conversation_parts.append(f"Unknown: {message.content}")

    # Combine system and conversation
    system_text = " ".join(system_parts) if system_parts else ""
    conversation_text = "\n".join(conversation_parts)

    # Create Mistral instruction format
    match (bool(system_text), bool(conversation_text)):
        case (True, True):
            combined_content = f"{system_text}\n\n{conversation_text}"
        case (True, False):
            combined_content = system_text
        case _:
            combined_content = conversation_text

    return f"<s>[INST] {combined_content} [/INST]"


def format_tool_calls_for_prompt(tool_calls: List[Dict[str, Any]]) -> str:
    """Format tool calls for inclusion in prompt."""
    formatted_calls = []
    for call in tool_calls:
        formatted_calls.append(f"TOOL_CALL: {json.dumps({'name': call['name'], 'arguments': call['args']})}")
    return "\n".join(formatted_calls)


def parse_tool_calls(response_text: str) -> tuple[List[Dict[str, Any]], str]:
    """Parse tool calls from model response."""

    tool_calls = []
    clean_content = response_text

    # Find all TOOL_CALL: occurrences
    tool_call_starts = []
    for match in re.finditer(r'TOOL_CALL:\s*', response_text):
        tool_call_starts.append(match.end())

    for start_pos in tool_call_starts:
        try:
            # Find the complete JSON object starting at this position
            json_str = extract_complete_json(response_text, start_pos)
            if json_str:
                # Remove the tool call from clean content (including TOOL_CALL: prefix)
                tool_call_pattern = r'TOOL_CALL:\s*' + re.escape(json_str)
                clean_content = re.sub(tool_call_pattern, '', clean_content).strip()

                # Parse the JSON
                tool_call_json = json.loads(json_str)

                # Create tool call in LangChain format
                tool_call = {
                    "name": tool_call_json.get("name"),
                    "args": tool_call_json.get("arguments", {}),
                    "id": f"call_{len(tool_calls)}",  # Simple ID generation
                    "type": "tool_call"
                }

                tool_calls.append(tool_call)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse tool call JSON - {e}")
            continue
        except Exception as e:
            logger.warning(f"Error processing tool call - {e}")
            continue

    return tool_calls, clean_content


def extract_complete_json(text: str, start_pos: int) -> Optional[str]:
    """Extract a complete JSON object starting at the given position."""
    if start_pos >= len(text) or text[start_pos] != '{':
        return None

    brace_count = 0
    in_string = False
    escape_next = False

    for i in range(start_pos, len(text)):
        char = text[i]

        if escape_next:
            escape_next = False
            continue

        if char == '\\' and in_string:
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return text[start_pos:i + 1]

    return None


def convert_chat_prompt_to_minstral_prompt_value(chat_prompt_value: ChatPromptValue):
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
    prompt = messages_to_mistral_prompt(chat_prompt_value.messages)
    # Create and return a StringPromptValue
    return StringPromptValue(text=prompt)


def add_tool_instructions_to_prompt(prompt: str, tools: List[Dict[str, Any]]) -> str:
    """Add tool calling instructions to the prompt."""

    # Create tool descriptions
    tool_descriptions = []
    for tool in tools:
        tool_info = tool.get('function', {})
        name = tool_info.get('name', 'unknown')
        description = tool_info.get('description', 'No description')
        parameters = tool_info.get('parameters', {})

        tool_desc = f"- {name}: {description}"
        if parameters.get('properties'):
            props = parameters['properties']
            params_desc = ", ".join([f"{k} ({v.get('type', 'any')}): {v.get('description', '')}"
                                     for k, v in props.items()])
            tool_desc += f"\n  Parameters: {params_desc}"

        tool_descriptions.append(tool_desc)

    tools_text = "\n".join(tool_descriptions)

    # Add tool instructions before [/INST]
    tool_instructions = f"""

AVAILABLE TOOLS:
{tools_text}

INSTRUCTIONS:
- To call a tool, respond with: TOOL_CALL: {{"name": "tool_name", "arguments": {{"param1": "value1", "param2": "value2"}}}}
- You can call multiple tools by using multiple TOOL_CALL lines
- Always provide a human-readable response along with any tool calls
- If no tools are needed, respond normally without TOOL_CALL

"""

    # Insert instructions before [/INST]
    return prompt.replace(' [/INST]', tool_instructions + ' [/INST]')


class VLLMChatModel(BaseChatModel):
    """
    Custom Chat Model wrapper around VLLM that implements tool calling logic.
    
    This class bridges VLLM's text completion interface with LangChain's chat model
    interface, enabling structured tool calling functionality.
    """

    # Core VLLM instance
    vllm_model: VLLM = Field(description="The underlying VLLM model instance")

    # Tool calling configuration
    tools: List[Dict[str, Any]] = Field(default_factory=list, description="Bound tools for this model")
    tool_choice: Optional[str] = Field(default=None, description="Tool choice strategy")
    ignore_tools: bool = Field(default=False, description="Allow tool parsing and definition")

    # Pydantic Configuration Class
    # ============================
    # This Config class tells Pydantic (the data validation library that LangChain uses)
    # how to handle this model class and its attributes during validation and serialization.
    class Config:
        # Allow arbitrary types that Pydantic can't normally serialize
        # ============================================================
        # By default, Pydantic only accepts basic Python types (str, int, dict, list, etc.)
        # and rejects complex objects it doesn't understand. Our VLLMChatModel contains
        # a 'vllm_model' field that holds a VLLM instance - a complex object with:
        #   - GPU memory allocations
        #   - Model weights 
        #   - CUDA contexts
        #   - Internal state that can't be easily serialized to JSON
        #
        # Without arbitrary_types_allowed = True, Pydantic would raise an error like:
        # "TypeError: Object of type 'VLLM' is not JSON serializable"
        #
        # Setting this to True tells Pydantic:
        # "Trust me, I know this field contains a complex object that you can't serialize,
        #  but store it anyway and don't try to validate or convert it."
        #
        # Alternative approaches:
        # 1. Use @validator decorators to handle VLLM objects specially
        # 2. Store VLLM as a private attribute (self._vllm_model) outside Pydantic
        # 3. Create a custom serializer for VLLM objects
        #
        # We chose this approach because:
        # - It's simple and explicit
        # - We don't need to serialize the VLLM instance (it stays in memory)
        # - LangChain models often contain non-serializable client objects
        arbitrary_types_allowed = True

    def __init__(self, vllm_model: VLLM, **kwargs):
        # Explicitly pass callbacks=None to avoid deprecation warning issues
        if 'callbacks' not in kwargs:
            kwargs['callbacks'] = None
        super().__init__(vllm_model=vllm_model, **kwargs)

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "vllm_chat_wrapper"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get identifying parameters by looking up self.vllm_model """
        return {"model_name": getattr(self.vllm_model, 'model', 'unknown')}

    def unbind_tools(self, pass_through):
        """Remove tool binding from the model."""
        self.ignore_tools = True
        return pass_through

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[Any] = None,
            **kwargs: Any,
    ) -> ChatResult:
        """Generate chat result from messages."""

        # Convert messages to prompt string
        prompt = messages_to_mistral_prompt(messages)

        # Add tool instructions if tools are bound
        if self.tools and not self.ignore_tools:
            prompt = add_tool_instructions_to_prompt(prompt, self.tools)

        # Generate response using VLLM
        try:
            logger.info(f"LLM PROMPT:\n{prompt} ")
            response_text = self.vllm_model.invoke(prompt, stop=stop, **kwargs)
            logger.info(f"LLM RESPONSE to prompt:\n {response_text} ")
            if self.ignore_tools:
                tool_calls = []
                clean_content = response_text
            else:
                # Parse response for tool calls
                tool_calls, clean_content = parse_tool_calls(response_text)

            # Create AIMessage with tool calls
            ai_message = AIMessage(
                content=clean_content,
                tool_calls=tool_calls if tool_calls else []
            )

            generation = ChatGeneration(message=ai_message)
            return ChatResult(generations=[generation])

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # Fallback response
            ai_message = AIMessage(content=f"Error: {str(e)}")
            generation = ChatGeneration(message=ai_message)
            return ChatResult(generations=[generation])

    def bind_tools(
            self,
            tools: Sequence[Union[Dict[str, Any], type, Callable, BaseTool]],
            *,
            tool_choice: Optional[str] = None,
            **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Bind tools to this chat model."""

        # Convert tools to OpenAI format
        formatted_tools = []
        for tool in tools:
            if isinstance(tool, dict):
                formatted_tools.append(tool)
            else:
                formatted_tools.append(convert_to_openai_tool(tool))

        # MEMORY OPTIMIZATION: Reuse existing instance instead of creating new one
        # This prevents memory leaks from multiple VLLMChatModel instances
        logger.info(f"🔧 Binding {len(formatted_tools)} tools to existing VLLMChatModel instance")
        self.tools = formatted_tools
        self.tool_choice = tool_choice
        self.ignore_tools = False
        # Return self instead of creating new instance to prevent memory leaks
        return self

    def with_structured_output(
            self,
            schema: Union[Dict, type],
            **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[Dict, Any]]:
        """Add structured output support (basic implementation)."""
        # This is a simplified implementation
        # You can extend this for more sophisticated structured output
        raise NotImplementedError("Structured output not yet implemented for VLLMChatModel")

    def cleanup_llm_memory(self):
        """
        Clean up LLM instance and free GPU memory.
        
        This method should be called when the LLM is no longer needed
        to ensure proper resource cleanup.
        """
        if hasattr(self, 'vllm_model') and self.vllm_model is not None:
            logger.info("🧹 Cleaning up LLM resources...")

            try:
                # Clean up the LLM instance
                cleanup_vllm_engine(self.vllm_model)

                # Remove reference
                self.vllm_model = None

                logger.info("✅ LLM cleanup completed")

            except Exception as e:
                logger.warning(f"⚠️  LLM cleanup issue (non-critical): {e}")

    def __del__(self):
        """Ensure cleanup when object is destroyed."""
        try:
            self.cleanup_llm_memory()
        except:
            pass  # Ignore errors in destructor 


def get_langchain_vllm_mistral_quantized(download_dir=None, gpu_memory_utilization: float = 0.6,
                                         max_model_len: int = 16384, top_k: int = 5,
                                         max_tokens: int = 512, temperature: float = 0.7) -> VLLM:
    model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
    logger.info(check_gpu_memory_status())
    # Try full configuration first
    try:
        # Common VLLM kwargs for both branches
        vllm_kwargs = {
            "model": model_name,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "trust_remote_code": True,
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_k": top_k
        }

        # Add download_dir if provided
        if download_dir is not None:
            vllm_kwargs["download_dir"] = download_dir
        logger.info(vllm_kwargs)
        return_vllm = VLLM(**vllm_kwargs)
        return return_vllm

    except Exception as e:
        logger.warning(f"Failed to create VLLM with full configuration: {e}")
        logger.info("Trying simplified VLLM configuration...")
        logger.info(check_gpu_memory_status())
        # Fallback to simpler configuration
        try:
            simple_kwargs = {
                "model": model_name,
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": 0.5,  # Very conservative memory usage
                "trust_remote_code": True,
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_k": top_k
            }

            # Add download_dir if provided
            if download_dir is not None:
                simple_kwargs["download_dir"] = download_dir

            return_vllm = VLLM(**simple_kwargs)
            return return_vllm

        except Exception as e2:
            logger.info(check_gpu_memory_status())
            logger.error(f"Failed to create VLLM with simplified configuration: {e2}")
            # Try minimal configuration as last resort
            minimal_kwargs = {
                "model": model_name,
                "trust_remote_code": True,
                "max_new_tokens": max_tokens,
                "temperature": temperature
            }

            if download_dir is not None:
                minimal_kwargs["download_dir"] = download_dir

            return VLLM(**minimal_kwargs)


def create_vllm_chat_model(download_dir=None, gpu_memory_utilization: float = 0.75,
                           max_model_len: int = 16384, top_k: int = 5,
                           max_tokens: int = 512, temperature: float = 0.7) -> VLLMChatModel:
    """
    Create a VLLM Chat Model with tool calling support.
    
    Args:
        download_dir: Directory to download model files
        gpu_memory_utilization: GPU memory utilization ratio (reduced to 0.75 for stability)
        max_model_len: Maximum model context length
        top_k: Top-k sampling parameter
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        
    Returns:
        VLLMChatModel instance with tool calling support
    """

    try:
        # Create underlying VLLM model
        vllm_model = get_langchain_vllm_mistral_quantized(
            download_dir=download_dir,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            top_k=top_k,
            max_tokens=max_tokens,
            temperature=temperature
        )

        # Wrap in chat model
        return VLLMChatModel(vllm_model=vllm_model, callbacks=None)

    except Exception as e:
        logger.error(f"Failed to create VLLM chat model: {e}")
        logger.info("Attempting fallback with reduced parameters...")

        # Try with reduced memory utilization as fallback
        try:
            vllm_model = get_langchain_vllm_mistral_quantized(
                download_dir=download_dir,
                gpu_memory_utilization=0.5,  # Reduced memory utilization
                max_model_len=8192,  # Reduced context length
                top_k=top_k,
                max_tokens=max_tokens,
                temperature=temperature
            )

            return VLLMChatModel(vllm_model=vllm_model, callbacks=None)

        except Exception as e2:
            logger.error(f"Fallback also failed: {e2}")
            raise RuntimeError(f"Unable to create VLLM chat model. Original error: {e}, Fallback error: {e2}")
