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

def dict_to_str(d: dict) -> str:
    """Convert a dictionary to a formatted string.
    
    Args:
        d: Dictionary to convert
        
    Returns:
        Formatted string representation of the dictionary
    """
    if not d:
        return ''
    context =""
    for key, value in d.items():
        context += f"{key}: {value}\n"
    return context

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
    context =""
    for i, doc in enumerate(results, 1):
        # context += doc.page_content + "\n\n" 
        context += f"""
        Metadata Source/File path:{doc.metadata.get('source', doc.metadata.get('file_path', 'Unknown'))}
        page: {doc.metadata.get('page', doc.metadata.get('page-label', 'N/A'))}
        format: {doc.metadata.get('format',  doc.metadata.get('file_type', 'N/A'))}
        Total Pages in Source/File: {doc.metadata.get('total_pages',  doc.metadata.get('total-pages', 'N/A'))}
        Similarity Score: {doc.metadata.get('similarity_score', 'N/A')}
        """
        context += "=" * 80 + "\n"
    return context


if __name__ == "__main__":
    # Initialize llm variable outside try block so it's accessible in finally
    llm = None
    from inference.vllm_srv.cleaner import cleanup_vllm_engine 

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

        # Prompt Definition

        format_function = format_list_documents_as_string 
        from langchain_core.runnables import RunnablePassthrough, RunnableLambda
        chat_chain = ( 
                {
                    "context": retriever | RunnableLambda(lambda docs: format_function(results=docs)), 
                    "question": RunnablePassthrough()
                } 
                | RunnableLambda(lambda x: dict_to_str(x))
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
                                                  
    