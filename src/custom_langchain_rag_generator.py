from langchain_core.prompt_values import StringPromptValue
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from typing import List
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
    from inference.vllm_srv.cleaner import cleanup_vllm_engine 
    from inference.vllm_srv.facebook_langchain import get_langchain_vllm_facebook_opt_125m, convert_chat_prompt_to_facebook_prompt_value
    from inference.vllm_srv.minstral_langchain import get_langchain_vllm_mistral_quantized, convert_chat_prompt_to_minstral_prompt_value

    try:
        # 1. Setup Retriever (e.g., RAG)
        # from langchain_community.embeddings import OllamaEmbeddings
        # embeddings = OllamaEmbeddings(model="llama3") # Or whatever you use
        # from langchain_community.vectorstores import FAISS # Example vector store
        # vectorstore = FAISS.from_texts(["Your documents here"], embeddings)
        # retriever = vectorstore.as_retriever()
        from vectordatabases.chroma_vector_db import ChromaVectorDB
        from vectordatabases.faiss_vector_db import FaissVectorDB
        from vectordatabases.qdrant_vector_db import QdrantVectorDB
        from vectordatabases.vector_db_factory import VectorDBFactory
        from embeddings.embedding_manager import EmbeddingManager
        from embeddings.huggingface_transformer.local_model import HuggingFaceLocalModel

        root_path = "/home/jmitchall/vllm-srv"
        collection_name = "dnd_player"
        chunk_lib = "langchain"
        DATABASE_TYPE = "chroma"
        vector_db_persisted_path =f"{root_path}/custom/{DATABASE_TYPE}/{chunk_lib}_{collection_name}"

        # Create vector database
        vector_db = ChromaVectorDB(
            persist_path=vector_db_persisted_path,
        )

        vector_db_embedding_dim =  vector_db.get_embedding_dim()

        # Initialize embedding manager
        print("📊 Initializing embedding manager...")
        embedding_interface = HuggingFaceLocalModel(model_name="BAAI/bge-large-en-v1.5" ,safety_level="max",  use_server=False,)
        safe_embedding_dim = embedding_interface.safe_embedding_dim
    
        embedding_manager = EmbeddingManager(
            embedding_model_instance=embedding_interface,
            embedding_dim=vector_db_embedding_dim if vector_db_embedding_dim else safe_embedding_dim
        )

        # Create LangChain retriever
        print("\n🔗 Creating LangChain retriever...")
        retriever = vector_db.as_langchain_retriever(
            embedding_function=embedding_manager.get_embedding,
            top_k=5
        )

        # Use the retriever
        print("\n🔍 Searching with LangChain retriever...")
        query = "What is a Rogue?"

        # Start the program by calling our main function
        llm = get_langchain_vllm_mistral_quantized(download_dir="./models") #get_langchain_vllm_facebook_opt_125m(download_dir="./models")

        # Prompt Definition
        from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
        chat_prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("""You are an expert Dungeuns and dragon Game Master. You recall all the rules in the following sources:
1) Dungeon Master’s Guide - Dungeons & Dragons - Sources - D&D Beyond
2) Monster Manual - Dungeons & Dragons - Sources - D&D Beyond
3) Player’s Handbook - Dungeons & Dragons - Sources - D&D Beyond
4) Horror Adventures - Van Richten’s Guide to Ravenloft - Dungeons & Dragons - Sources - D&D Beyond

Before thinking of a response or answer make sure you initially use the following  
CONTEXT: 
{context}

"""),
            HumanMessagePromptTemplate.from_template("Provide a comprehensive and detailed response to the following \nTASK:\n{question}")
        ])
        from langchain_core.runnables import RunnablePassthrough, RunnableLambda
        chat_chain = ( 
            {
                "context": retriever | RunnableLambda(lambda docs: retriever.format_list_documents_as_string(results=docs)), 
                "question": RunnablePassthrough()
            } 
            | chat_prompt_template 
            | RunnableLambda(convert_chat_prompt_to_minstral_prompt_value)  # 👈 Convert WITHOUT role labels
            | RunnableLambda(lambda x: inspect_llm_input(x, "ChatPromptTemplate Chain"))
            | llm 
            #| RunnableLambda(lambda x: inspect_llm_output(x, "ChatPromptTemplate Chain"))
            | StrOutputParser()                                    # 👈 Parse to clean string
        )
        
        chat_result = chat_chain.invoke(query)
        print("\n🎯 Chat Prompt Result 3:")
        print("=" * 80)
        print(chat_result)

     
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
                                                  
    