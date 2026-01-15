#!/usr/bin/env python3
"""
LangChain RAG Generator - Experimental RAG Chain Utilities

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
"""

from langchain_core.output_parsers import StrOutputParser
from pathlib import Path
import sys
# Add parent directory to path for imports when running directly
if __name__ == "__main__":
    src_path = Path(__file__).parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
from refection_logger import logger


        
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
    Custom function to inspect and logger.info ChatPromptValue message structures.
    This runs BETWEEN the prompt template and the LLM in the chain.
    
    Args:
        prompt_value: ChatPromptValue object from ChatPromptTemplate
        
    Returns:
        The prompt_value unchanged (so the chain continues)
    """
    logger.info("\n" + "=" * 80)
    logger.info("📋 INSPECTING ChatPromptValue OBJECT (INPUT TO LLM):")
    logger.info("=" * 80)

    # 1. logger.info the object type
    logger.info(f"\n🔍 Type: {type(prompt_value)}")
    logger.info(f"   Class: {prompt_value.__class__.__name__}")

    # 2. logger.info all messages in the ChatPromptValue
    logger.info(f"\n💬 Messages ({len(prompt_value.messages)} total):")
    for i, message in enumerate(prompt_value.messages, 1):
        logger.info(f"\n   Message #{i}:")
        logger.info(f"   - Type: {type(message).__name__}")
        logger.info(f"   - Content: {message.content}")
        logger.info(f"   - Additional kwargs: {message.additional_kwargs}")
        logger.info(f"   - Response metadata: {message.response_metadata}")

    # 3. logger.info the string representation
    logger.info(f"\n📝 String Representation (.to_string()):")
    logger.info("-" * 80)
    logger.info(prompt_value.to_string())
    logger.info("-" * 80)

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
    logger.info("\n" + "=" * 80)
    logger.info(f"🧐 INSPECTING LLM INPUT MESSAGE{label}:")
    logger.info("=" * 80)
    logger.info(f"\n🔍 Type: {type(llm_input)}")
    logger.info(f"   Class: {llm_input.__class__.__name__}")
    logger.info(f"\n💬 Content:")
    logger.info("-" * 80)
    logger.info(llm_input)
    logger.info("-" * 80)
    logger.info(f"\n📏 Length: {len(str(llm_input))} characters")

    # Show repr for debugging
    logger.info(f"\n🔬 Repr: {repr(llm_input)[:200]}...")

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
    logger.info("\n" + "=" * 80)
    logger.info(f"🤖 INSPECTING LLM OUTPUT MESSAGE{f' ({chain_name})' if chain_name else ''}:")
    logger.info("=" * 80)

    # 1. logger.info the object type
    logger.info(f"\n🔍 Type: {type(llm_output)}")
    logger.info(f"   Class: {llm_output.__class__.__name__}")

    logger.info(f"\n💬 Content:")
    logger.info("-" * 80)
    logger.info(llm_output)
    logger.info("-" * 80)
    return llm_output


def format_list_documents_as_string(**kwargs) -> str:
    """Format a list of Documents into a human-readable string with metadata.
    
    Args:
        results: List of Document objects with page_content and metadata
        similarity_threshold: Minimum similarity score to include a document (default: 0.8)
        
    Returns:
        Formatted string with each document's content, source, similarity score, and rank
    """
    results = kwargs.get('results', [])
    if not results:
        return ''
    context = ""
    for i, doc in enumerate(results, 1):
        context += doc.page_content + "\n\n"
    return context


def get_retriever_and_vector_stores(vdb_type: str, vector_db_persisted_path: str,
                                    collection_ref: str, retriever_embeddings):
    from vectordatabases.qdrant_vector_db_commands import QdrantClientSmartPointer, quadrant_does_collection_exist, \
        get_quadrant_client, get_qdrant_retriever
    from vectordatabases.fais_vector_db_commands import create_faiss_vectorstore, get_faiss_retriever
    from vectordatabases.chroma_vector_db_commands import get_chroma_vectorstore, get_chroma_retriever
    langchain_retriever = None
    qdrant_client: QdrantClientSmartPointer = None
    # test persisted vector store loading and retriever creation
    logger.info(
        f"\n🔍 Testing loading of persisted vector store for collection '{collection_ref}' from path: {vector_db_persisted_path} ...")
    match vdb_type:
        case "qdrant":
            # Reconnect to the persisted Qdrant database
            qdrant_client: QdrantClientSmartPointer = get_quadrant_client(vector_db_persisted_path)
            if quadrant_does_collection_exist(qdrant_client, collection_ref):
                langchain_retriever = get_qdrant_retriever(qdrant_client, collection_ref,
                                                           embeddings=retriever_embeddings, k=5)
                logger.info(f"✅ Created Qdrant retriever wrapper for collection '{collection_ref}'")
            else:
                logger.info(f"⚠️  Collection '{collection_ref}' does not exist yet. Skipping retrieval test.")
                langchain_retriever = None
        case "faiss":
            loaded_vectorstore_wrapper = create_faiss_vectorstore(
                vector_db_persisted_path,
                retriever_embeddings
            )
            langchain_retriever = get_faiss_retriever(loaded_vectorstore_wrapper, k=5)
        case "chroma":
            loaded_vectorstore = get_chroma_vectorstore(collection_ref, vector_db_persisted_path,
                                                        retriever_embeddings)
            langchain_retriever = get_chroma_retriever(loaded_vectorstore, k=5)
        case _:
            raise ValueError(
                f"Unsupported DATABASE_TYPE: {vdb_type}. Supported types are 'qdrant', 'faiss', 'chroma'.")
    return langchain_retriever, qdrant_client


if __name__ == "__main__":
    # Initialize llm variable outside try block so it's accessible in finally
    llm = None
    from inference.vllm_srv.cleaner import cleanup_vllm_engine, force_gpu_memory_cleanup
    from embeddings.huggingface_transformer.langchain_embedding import HuggingFaceOfflineEmbeddings

    force_gpu_memory_cleanup()
    try:

        # 1. Setup Embeddings
        embedding_model = "BAAI/bge-large-en-v1.5"
        embeddings = HuggingFaceOfflineEmbeddings(model_name=embedding_model)

        # 2. Setup Retriever (e.g., RAG)
        db_type = 'qdrant'  # Change to "qdrant", "faiss", "chroma"
        DATABASE_TYPE = db_type.lower()
        collection_names = [
            "vtm",
            "dnd_dm",
            "dnd_mm",
            "dnd_raven",
            "dnd_player"
        ]
        collection_name = collection_names[-1]  # Choose one collection for this example
        root_path = "/home/jmitchall/vllm-srv"
        root_persist_path = f"{root_path}/db"
        vector_db_persisted_path = f"{root_persist_path}/llamaindex_{collection_name}_{DATABASE_TYPE}"
        retriever, qdrant_client_ref = get_retriever_and_vector_stores(DATABASE_TYPE, vector_db_persisted_path,
                                                                       collection_name, embeddings)
        if retriever is None:
            logger.info(f"❌ Retriever could not be created. Exiting.")
            exit(1)

        # 3. Setup LLM 
        from inference.vllm_srv.minstral_langchain import get_langchain_vllm_mistral_quantized, \
            convert_chat_prompt_to_minstral_prompt_value
        from inference.vllm_srv.facebook_langchain import get_langchain_vllm_facebook_opt_125m, \
            convert_chat_prompt_to_facebook_prompt_value

        llm = get_langchain_vllm_mistral_quantized(
            download_dir="./models")  # get_langchain_vllm_facebook_opt_125m(download_dir="./models")

        # 4. Prompt Definition and setup ChatPromptTemplate chain
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
            HumanMessagePromptTemplate.from_template(
                "Provide a comprehensive and detailed response to the following \nTASK:\n{question}")
        ])

        format_function = retriever.format_list_documents_as_string if hasattr(retriever,
                                                                               'format_list_documents_as_string') else format_list_documents_as_string

        # 5. Setup the chain with inspectors and LCEL components 
        from langchain_core.runnables import RunnablePassthrough, RunnableLambda

        chat_chain = (
                {
                    "context": retriever | RunnableLambda(lambda docs: format_function(results=docs)),
                    "question": RunnablePassthrough()
                }
                | chat_prompt_template
                | RunnableLambda(convert_chat_prompt_to_minstral_prompt_value)  # 👈 Convert WITHOUT role labels
                | RunnableLambda(lambda x: inspect_llm_input(x, "ChatPromptTemplate Chain"))
                | llm
                # | RunnableLambda(lambda x: inspect_llm_output(x, "ChatPromptTemplate Chain"))
                | StrOutputParser()  # 👈 Parse to clean string
        )

        # 6. Invoke LECL Chain with test queries
        query = "What is a Rogue?"
        chat_result = chat_chain.invoke(query)
        logger.info("\n🎯 Chat Prompt Result 3:")
        logger.info("=" * 80)
        logger.info(chat_result)

    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        logger.info("\n\n⚠️  Interrupted by user. Cleaning up...")

    except Exception as e:
        # Catch any other errors and logger.info them
        logger.info(f"\n❌ Error occurred: {e}")
        import traceback

        traceback.logger.info_exc()

    finally:
        # =================================================================
        # CLEANUP: Always run this code, even if errors occurred
        # =================================================================
        # This ensures proper resource cleanup and prevents the engine crash error
        if llm is not None:
            logger.info("\n🧹 Cleaning up vLLM resources...")
            cleanup_vllm_engine(llm)
            logger.info("✅ Cleanup completed successfully!")
