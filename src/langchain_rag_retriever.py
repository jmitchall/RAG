from langchain_core.output_parsers import StrOutputParser


def dict_to_str(d: dict) -> str:
    """Convert a dictionary to a formatted string.
    
    Args:
        d: Dictionary to convert
        
    Returns:
        Formatted string representation of the dictionary
    """
    if not d:
        return ''
    context = ""
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
    context = ""
    for i, doc in enumerate(results, 1):
        # context += doc.page_content + "\n\n" 
        context += f"""
        Metadata Source/File path:{doc.metadata.get('source', doc.metadata.get('file_path', 'Unknown'))}
        page: {doc.metadata.get('page', doc.metadata.get('page-label', 'N/A'))}
        format: {doc.metadata.get('format', doc.metadata.get('file_type', 'N/A'))}
        Total Pages in Source/File: {doc.metadata.get('total_pages', doc.metadata.get('total-pages', 'N/A'))}
        Similarity Score: {doc.metadata.get('similarity_score', 'N/A')}
        """
        context += "=" * 80 + "\n"
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
    print(
        f"\n🔍 Testing loading of persisted vector store for collection '{collection_ref}' from path: {vector_db_persisted_path} ...")
    match vdb_type:
        case "qdrant":
            # Reconnect to the persisted Qdrant database
            qdrant_client: QdrantClientSmartPointer = get_quadrant_client(vector_db_persisted_path)
            if quadrant_does_collection_exist(qdrant_client, collection_ref):
                langchain_retriever = get_qdrant_retriever(qdrant_client, collection_ref,
                                                           embeddings=retriever_embeddings, k=5)
                print(f"✅ Created Qdrant retriever wrapper for collection '{collection_ref}'")
            else:
                print(f"⚠️  Collection '{collection_ref}' does not exist yet. Skipping retrieval test.")
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

        chunker_func = [
            "langchain",
            "llamaindex"
        ]

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

        for chunker_key in chunker_func:
            vector_db_persisted_path = f"{root_path}/{chunker_key}_{collection_name}_{DATABASE_TYPE}"
            retriever, qdrant_client_ref = get_retriever_and_vector_stores(DATABASE_TYPE, vector_db_persisted_path,
                                                                           collection_name, embeddings)
            if retriever is None:
                print(f"❌ Retriever could not be created. Exiting.")
                exit(1)

            format_function = format_list_documents_as_string
            # Setup the chain with inspectors and LCEL components 
            from langchain_core.runnables import RunnablePassthrough, RunnableLambda

            chat_chain = (
                    {
                        "context": retriever | RunnableLambda(lambda docs: format_function(results=docs)),
                        "question": RunnablePassthrough()
                    }
                    | RunnableLambda(lambda x: dict_to_str(x))
                    | StrOutputParser()  # 👈 Parse to clean string
            )

            # 6. Invoke LECL Chain with test queries
            query = "What is a Rogue?"
            chat_result = chat_chain.invoke(query)
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
