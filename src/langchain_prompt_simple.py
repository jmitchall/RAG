from inference.vllm_srv.cleaner import cleanup_vllm_engine 
from inference.vllm_srv.facebook_llmlite import get_langchain_vllm_facebook_opt_125m
from langchain_core.prompts import PromptTemplate
from langchain_core.prompt_values import StringPromptValue
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# =================================================================
# CHAIN WITH MESSAGE STRUCTURE INSPECTORS
# =================================================================
# If you're working with chat-based models, you can use ChatPromptTemplate
# to structure the conversation with system and user messages.
# =================================================================
# INSPECTING ChatPromptValue MESSAGE STRUCTURES
# =================================================================
# ChatPromptTemplate produces a ChatPromptValue object with message structures.
# To inspect these BEFORE they go to the LLM, use RunnableLambda as a debugger.

def inspect_chat_prompt_value(prompt_value):
    """
    Custom function to inspect and print ChatPromptValue message structures.
    This runs BETWEEN the prompt template and the LLM in the chain.
    
    Args:
        prompt_value: ChatPromptValue object from ChatPromptTemplate
        
    Returns:
        The prompt_value unchanged (so the chain continues)
    """
    print("\n" + "="*80)
    print("📋 INSPECTING ChatPromptValue OBJECT (INPUT TO LLM):")
    print("="*80)
    
    # 1. Print the object type
    print(f"\n🔍 Type: {type(prompt_value)}")
    print(f"   Class: {prompt_value.__class__.__name__}")
    
    # 2. Print all messages in the ChatPromptValue
    print(f"\n💬 Messages ({len(prompt_value.messages)} total):")
    for i, message in enumerate(prompt_value.messages, 1):
        print(f"\n   Message #{i}:")
        print(f"   - Type: {type(message).__name__}")
        print(f"   - Content: {message.content}")
        print(f"   - Additional kwargs: {message.additional_kwargs}")
        print(f"   - Response metadata: {message.response_metadata}")
    
    # 3. Print the string representation
    print(f"\n📝 String Representation (.to_string()):")
    print("-" * 80)
    print(prompt_value.to_string())
    print("-" * 80)
    
    # 4. IMPORTANT: Return the prompt_value so the chain continues
    # If you don't return it, the chain will break
    return prompt_value

def inspect_llm_input(llm_input, chain_name=""):
    """
    Custom function to inspect the LLM's input message/string.
    This runs BEFORE the LLM generates output.
    
    Args:
        llm_input: The input to the LLM (could be string or message object)
        chain_name: Optional label to identify which chain is being inspected
    Returns:
        The llm_input unchanged (so the chain continues to the LLM)
    """
    label = f" ({chain_name})" if chain_name else ""
    print("\n" + "="*80)
    print(f"🧐 INSPECTING LLM INPUT MESSAGE{label}:")
    print("="*80)
    print(f"\n🔍 Type: {type(llm_input)}")
    print(f"   Class: {llm_input.__class__.__name__}")
    print(f"\n💬 Content:")
    print("-" * 80)
    print(llm_input)
    print("-" * 80)
    print(f"\n📏 Length: {len(str(llm_input))} characters")
    
    # Show repr for debugging
    print(f"\n🔬 Repr: {repr(llm_input)[:200]}...")
    
    return llm_input

def inspect_llm_output(llm_output, chain_name=""):
    """
    Custom function to inspect the LLM's output message/string.
    This runs AFTER the LLM generates output but BEFORE the output parser.
    
    Args:
        llm_output: The output from the LLM (could be string or message object)
        chain_name: Optional label to identify which chain is being inspected
        
    Returns:
        The llm_output unchanged (so the chain continues to the parser)
    """
    print("\n" + "="*80)
    print(f"🤖 INSPECTING LLM OUTPUT MESSAGE{f' ({chain_name})' if chain_name else ''}:")
    print("="*80)
    
    # 1. Print the object type
    print(f"\n🔍 Type: {type(llm_output)}")
    print(f"   Class: {llm_output.__class__.__name__}")
    
    print(f"\n💬 Content:"            )
    print("-" * 80)
    print(llm_output)
    print("-" * 80)
    return llm_output

if __name__ == "__main__":
 
    # Initialize llm variable outside try block so it's accessible in finally
    llm = None
    
    try:
        # Start the program by calling our main function
        llm = get_langchain_vllm_facebook_opt_125m(download_dir="./models")
        
        # =================================================================
        # PROMPT TEMPLATE SETUP
        # =================================================================
        # Create a prompt template that expects two variables: {country} and {question}
        # This template defines the structure of the message we'll send to the AI
        prompt_template = PromptTemplate.from_template(
            """You are a helpful assistant that is an expert in geography and social studies.\nWhat is the capital of {country}.\n{question}"""
        )
  
        prompt_chain = (prompt_template 
                 #| RunnableLambda(lambda x: inspect_llm_input(x, "PromptTemplate Chain"))  # 👈 Inspect with label
                 | llm 
                 #| RunnableLambda(lambda x: inspect_llm_output(x, "PromptTemplate Chain"))                   # 👈 Inspect LLM's raw output
                 | StrOutputParser())

    
        # =================================================================
        # INVOKE THE CHAIN
        # =================================================================
        # Call the chain with a dictionary containing all required variables
        # The data flows through: Input Dict → RunnablePassthrough → Prompt → LLM → Parser → String
        prompt_result = prompt_chain.invoke({"country": "France", "question": "Why is it important?"})
        # Print the final result (a plain string)
        print("\n🎯 Prompt Chain Result 1:")
        print("=" * 80)
        print(prompt_result)
     
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\n\n⚠️  Interrupted by user. Cleaning up...")
        
    except Exception as e:
        # Catch any other errors and print them
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # =================================================================
        # CLEANUP: Always run this code, even if errors occurred
        # =================================================================
        # This ensures proper resource cleanup and prevents the engine crash error
        if llm is not None:
            print("\n🧹 Cleaning up vLLM resources...")
            cleanup_vllm_engine(llm)
            print("✅ Cleanup completed successfully!")
                                                  
    