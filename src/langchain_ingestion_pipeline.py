from ingestion.base_document_chunker import BaseDocumentChunker
from ingestion.doc_parser_langchain import DocumentChunker
from ingestion.doc_parser_llama_index import LlamaIndexDocumentChunker
from embeddings.huggingface_transformer.langchain_embedding import HuggingFaceOfflineEmbeddings
from embeddings.vllm.langchain_embedding import VLLMOfflineEmbeddings
from langchain.schema import Document
from typing import List, Tuple
import os
from langchain_chroma import Chroma


def consolidate_collections_to_all(root_data_path = "/home/jmitchall/vllm-srv/data", document_collections = [
        "vtm",
        "dnd",
        "all"
    ]):
    """
    Consolidate documents from multiple collection directories into a single 'all' collection.
    
    Creates symlinks from individual collection directories (vtm, dnd) into a unified 'all' 
    directory to avoid file duplication while maintaining access to all documents in one place.
    
    Args:
        root_data_path: Root directory containing collection subdirectories
        document_collections: List of collection names to process
        
    Example:
        Files in /home/jmitchall/vllm-srv/data/vtm and 
        /home/jmitchall/vllm-srv/data/dnd are symlinked into 
        /home/jmitchall/vllm-srv/data/all
    """
    for collection_name in document_collections:
        files = [f for f in os.listdir(os.path.join(root_data_path, collection_name)) if
                 os.path.isfile(os.path.join(root_data_path, collection_name, f))]
        print(f"Collection '{collection_name}' has {len(files)} files: {files}")
        # copy files into 'all' collection
        if collection_name != "all":
            all_collection_path = os.path.join(root_data_path, "all")
            if not os.path.exists(all_collection_path):
                os.makedirs(all_collection_path)
            for f in files:
                src = os.path.join(root_data_path, collection_name, f)
                dst = os.path.join(all_collection_path, f)
                if not os.path.exists(dst):
                    os.symlink(src, dst)  # Create a symlink to avoid duplication
   
def load_and_chunk_texts(chunker: BaseDocumentChunker,  max_token_validator: int) -> Tuple[List[str], List[Document]]:
    """
    Load and chunk documents using any document chunker implementation.
    
    This unified method works with any chunker that implements the BaseDocumentChunker interface,
    whether it's LangChain-based, LlamaIndex-based, or any future implementation.
    
    Args:
        chunker: An instance of a document chunker implementing BaseDocumentChunker
        d_path: Directory path containing documents to load
        max_token_validator: Maximum tokens allowed by the embedding model
        
    Returns:
        List of validated and chunked text strings
    """
    # Load Documents
    documents = chunker.directory_to_documents()
    if not documents or len(documents) == 0:
        raise ValueError("No documents found in the specified directory.")
    
    # Calculate optimal chunk parameters size and overlap 
    chunker.calculate_optimal_chunk_parameters_given_max_tokens(max_token_validator)
    chunk_texts , chunks = chunker.get_chunked_texts_list(documents)

    # Additional validation and fixing taking into consideration embedding model limits
    print(f"🔍 Validating chunk sizes based on max_tokens: {max_token_validator} ...")
    return chunker.validate_and_fix_chunks(chunk_texts, max_token_validator), chunks

def load_llamaIndex_texts(d_path: str, max_token_validator: int) -> Tuple[List[str], List[Document]]:
    """
    Legacy wrapper for LlamaIndex-based chunking. Consider using load_and_chunk_texts directly.
    Args:
        d_path: Directory path containing documents to load
        max_token_validator: Maximum tokens allowed by the embedding model
    Returns:
        List of validated and chunked text strings
    """
    print(f"🔍 Loading and LlamaIndex-based chunking documents from {d_path} ...")
    chunker = LlamaIndexDocumentChunker(d_path)
    return load_and_chunk_texts(chunker, max_token_validator)

def load_langchain_texts(d_path: str, max_token_validator: int) -> Tuple[List[str], List[Document]]:
    """
    Legacy wrapper for LangChain-based chunking. Consider using load_and_chunk_texts directly.

    Args:
        d_path: Directory path containing documents to load
        max_token_validator: Maximum tokens allowed by the embedding model
    Returns:
        List of validated and chunked text strings
    """
    print(f"🔍 Loading and LangChain-based chunking documents from {d_path} ...")
    chunker = DocumentChunker(d_path)
    return load_and_chunk_texts(chunker, max_token_validator)


def ingest_documents(vector_db_collection_name:str, chunked_documents: List[Document],
                     persistence_target_path: str, embedding_object):
    from vectordatabases.qdrant_vector_db_commands import QdrantClientSmartPointer, get_quadrant_client, \
        qdrant_create_from_documents
    from vectordatabases.fais_vector_db_commands import faiss_create_from_documents
    from vectordatabases.chroma_vector_db_commands import chroma_create_from_documents

    # Initialize embedding and vectore stores
    match DATABASE_TYPE:
        case "qdrant":
            # Create Qdrant client with local persistent storage
            qdrant_client: QdrantClientSmartPointer = get_quadrant_client(persistence_target_path)
            db_vectorstore = qdrant_create_from_documents(
                qdrant_client,
                vector_db_collection_name,
                chunked_documents,
                embedding_object
            )
            # close client connection
            qdrant_client.close()
        case "faiss":
            db_vectorstore = faiss_create_from_documents(
                persistence_target_path,
                chunked_documents,
                embedding_object
            )
        case "chroma":
            db_vectorstore = chroma_create_from_documents(
                vector_db_collection_name,
                persistence_target_path,
                chunked_documents,
                embedding_object
            )
        case _:
            raise ValueError(
                f"Unsupported DATABASE_TYPE: {DATABASE_TYPE}. Supported types are 'qdrant', 'faiss', 'chroma'.")
    return db_vectorstore


def get_retriever_and_vector_stores(vdb_type:str, vector_db_persisted_path:str, 
                                    collection_ref:str, retriever_embeddings) :
    from vectordatabases.qdrant_vector_db_commands import QdrantClientSmartPointer, quadrant_does_collection_exist ,get_quadrant_client, get_qdrant_retriever
    from vectordatabases.fais_vector_db_commands import create_faiss_vectorstore
    from vectordatabases.chroma_vector_db_commands import get_chroma_vectorstore
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
                langchain_retriever = get_qdrant_retriever(qdrant_client, collection_ref, embeddings=retriever_embeddings, k=5)
                print(f"✅ Created Qdrant retriever wrapper for collection '{collection_ref}'")
            else:
                print(f"⚠️  Collection '{collection_ref}' does not exist yet. Skipping retrieval test.")
                langchain_retriever = None
        case "faiss":
            loaded_vectorstore_wrapper = create_faiss_vectorstore(
                vector_db_persisted_path,
                retriever_embeddings
            )
            loaded_vectorstore = loaded_vectorstore_wrapper.faiss_vectorstore
            langchain_retriever = loaded_vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
        case "chroma":
            loaded_vectorstore = get_chroma_vectorstore(collection_ref, vector_db_persisted_path, 
                                                        retriever_embeddings)
            langchain_retriever = loaded_vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
        case _:
            raise ValueError(
                f"Unsupported DATABASE_TYPE: {vdb_type}. Supported types are 'qdrant', 'faiss', 'chroma'.")
    return langchain_retriever, qdrant_client


if __name__ == "__main__":
    root_path = "/home/jmitchall/vllm-srv"
    root_data_path = f"{root_path}/data"
    document_collections = [
        "vtm",
        "dnd",
        "all"
    ]
    consolidate_collections_to_all(root_data_path=root_data_path, document_collections=document_collections)
    embedding_model = "BAAI/bge-large-en-v1.5"

    database_types = [
        "qdrant",
        "faiss",
        "chroma"
    ]
    embeddings = HuggingFaceOfflineEmbeddings(model_name=embedding_model)
    max_tokens = embeddings.max_tokens
    print(f"\n🧮 Using embedding model '{embedding_model}' with max tokens: {max_tokens}")
    for db_type in database_types:
        DATABASE_TYPE = db_type.lower()
        qdrant_client_ref = None
        for collection_name in document_collections:
            directory_path = f"{root_data_path}/{collection_name}" 
            # place Collection name in persist path for FAISS because there is no concept of collection there
            # for Qdrant and Chroma we can use collection name directly
            vector_db_persisted_path =f"{root_path}/langchain_vector_db_{collection_name}_{DATABASE_TYPE}"
            print(f"\n📂 Processing collection '{collection_name}' from directory: {directory_path }" )
            # Load and chunk documents
            chunk_texts, documents = load_langchain_texts(directory_path, max_token_validator=max_tokens)

            ingest_documents(collection_name, documents, vector_db_persisted_path, embeddings)

            retriever , qdrant_client_ref = get_retriever_and_vector_stores(DATABASE_TYPE, vector_db_persisted_path, collection_name, embeddings)
            # perform a test retrieval
            query = "Tell me about vampires."
            if retriever is None:
                print(f"⚠️  Retriever could not be created for collection '{collection_name}'. Skipping retrieval test.")
                continue

            results = retriever.get_relevant_documents(query)
            print(f"✅ Retrieved {len(results)} documents for query: '{query}'")    
            for i, doc in enumerate(results):
                print(f"--- Document {i+1} ---")
                print(doc.page_content)
                print("---------------------")
        
    print("🎉 Ingestion complete!")


    

    