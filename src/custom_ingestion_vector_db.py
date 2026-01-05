from vectordatabases.faiss_vector_db import FaissVectorDB
from vectordatabases.vector_db_factory import VectorDBFactory
from embeddings.embedding_manager import EmbeddingManager
from embeddings.embedding_model_interace import EmbeddingModelInterface
from embeddings.huggingface_transformer.local_model import  HuggingFaceLocalModel
from ingestion.base_document_chunker import BaseDocumentChunker
from ingestion.doc_parser_langchain import DocumentChunker
from ingestion.doc_parser_llama_index import LlamaIndexDocumentChunker
from vectordatabases.vector_db_interface import VectorDBInterface
from langchain.schema import Document
from typing import List, Tuple, Optional
import os
import re
import numpy as np

def consolidate_collections_to_all(root_data_path = "/home/jmitchall/vllm-srv/data", document_collections = [
        "vtm",
        "dnd_dm",
        "dnd_mm",
        "dnd_raven",
        "dnd_player"
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



def load_and_chunk_texts(chunker: BaseDocumentChunker,  max_token_validator: int, **kwargs ) -> Tuple[List[str], List[Document]]:
    """
    Load and chunk documents using any document chunker implementation.
    
    This unified method works with any chunker that implements the BaseDocumentChunker interface,
    whether it's LangChain-based, LlamaIndex-based, or any future implementation.
    
    Args:
        chunker: An instance of a document chunker implementing BaseDocumentChunker
        d_path: Directory path containing documents to load
        max_token_validator: Maximum tokens allowed by the embedding model
        avg_words_per_token: Average number of words per token for the embedding model
        
    Returns:
        List of validated and chunked text strings
    """
    # Load Documents
    documents = chunker.directory_to_documents()
    if not documents or len(documents) == 0:
        raise ValueError("No documents found in the specified directory.")
    embedding_manager= kwargs.get('embedding_manager', None)
    if embedding_manager is None:
        avg_words_per_token = 0.75  # Default practical value for English
    else:
        chunk_test_sample:List[str] = [ doc.page_content if hasattr(doc, 'page_content') else doc.text for doc in documents[:10]]
        avg_words_per_token = embedding_manager.embedding_model.calculate_avg_words_per_token(chunk_test_sample)
        print(f"   Calculated average words per token: {avg_words_per_token:.4f}")

    # Calculate optimal chunk parameters size and overlap 
    chunker.calculate_optimal_chunk_parameters_given_max_tokens(max_token_validator, avg_words_per_token=avg_words_per_token)
    chunk_texts , chunks = chunker.get_chunked_texts_list(documents)

    # Additional validation and fixing taking into consideration embedding model limits
    print(f"🔍 Validating chunk sizes based on max_tokens: {max_token_validator} ...")
    return chunker.validate_and_fix_chunks(chunk_texts, max_token_validator), chunks

def load_llamaIndex_texts(d_path: str, max_token_validator: int, **kwargs) -> Tuple[List[str], List[Document]]:
    """
    Legacy wrapper for LlamaIndex-based chunking. Consider using load_and_chunk_texts directly.
    Args:
        d_path: Directory path containing documents to load
        max_token_validator: Maximum tokens allowed by the embedding model
    Returns:
        List of validated and chunked text strings
    """
    print(f"🔍 Loading and LlamaIndex-based chunking documents from {d_path} ...")
    chunker = LlamaIndexDocumentChunker(d_path, **kwargs)
    return load_and_chunk_texts(chunker, max_token_validator, **kwargs)

def load_langchain_texts(d_path: str, max_token_validator: int , **kwargs) -> Tuple[List[str], List[Document]]:
    """
    Legacy wrapper for LangChain-based chunking. Consider using load_and_chunk_texts directly.

    Args:
        d_path: Directory path containing documents to load
        max_token_validator: Maximum tokens allowed by the embedding model
    Returns:
        List of validated and chunked text strings
    """
    print(f"🔍 Loading and LangChain-based chunking documents from {d_path} ...")
    chunker = DocumentChunker(d_path, **kwargs) 
    return load_and_chunk_texts(chunker, max_token_validator, **kwargs)


def _truncate_to_max_tokens(embedding_manager: EmbeddingManager, text: str, max_tokens: int) -> str:
    """
    Truncate text to fit within max_tokens using the actual tokenizer.
    
    Args:
        embedding_manager: Manager with access to the tokenizer
        text: Text to truncate
        max_tokens: Maximum number of tokens allowed
        
    Returns:
        Truncated text that fits within max_tokens
    """
    try:
        # Get the tokenizer from the embedding model
        tokenizer = embedding_manager.embedding_model.tokenizer
        
        # Tokenize the text
        tokens = tokenizer.encode(text, add_special_tokens=True)
        
        # If within limits, return original text
        if len(tokens) <= max_tokens:
            return text
        
        # Truncate tokens to max_tokens
        truncated_tokens = tokens[:max_tokens]
        
        # Decode back to text
        truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        
        return truncated_text
    except Exception as e:
        # Fallback to word-based truncation if tokenizer access fails
        words = text.split()
        estimated_max_words = int(max_tokens * 0.75)  # Conservative estimate
        return ' '.join(words[:estimated_max_words])

def _extract_token_info_from_error(error_msg: str, chunk: str, chunk_index: int) -> None:
    """
    Extract and log token overflow information from error messages.
    
    Args:
        error_msg: The error message string
        chunk: The chunk text that caused the error
        chunk_index: The index of the chunk (1-based for display)
    """
    if "input tokens" in error_msg:
        try:
            match = re.search(r'(\d+)\s+input tokens', error_msg)
            if match:
                actual_tokens = int(match.group(1))
                word_count = len(chunk.split())
                ratio = actual_tokens / word_count if word_count > 0 else 0
                print(f"   ❌ Chunk {chunk_index} token overflow: {actual_tokens} tokens "
                      f"for {word_count} words (ratio: {ratio:.2f})")
            else:
                print(f"   ❌ Chunk {chunk_index} failed with token overflow")
        except Exception:
            print(f"   ❌ Chunk {chunk_index} failed with token overflow")
    else:
        print(f"   ❌ Chunk {chunk_index} failed: {error_msg[:100]}")

def _attempt_progressive_truncation(
    embedding_manager: EmbeddingManager,
    chunk: str,
    chunk_index: int,
    retry_count: int,
    max_tokens: int
) -> Optional[np.ndarray]:
    """
    Attempt to generate embedding using tokenizer-based truncation.
    
    Uses the actual tokenizer to truncate text to progressively smaller token counts
    (90%, 75%, 50%, 25% of max_tokens) until successful.
    
    Args:
        embedding_manager: Manager for generating embeddings
        chunk: The text chunk to embed
        chunk_index: The index of the chunk (1-based for display)
        retry_count: Current count of retries (for logging purposes)
        max_tokens: Maximum tokens allowed by the embedding model
        
    Returns:
        Embedding array if successful, None otherwise
    """
    # Try progressively smaller token limits using actual tokenizer
    truncation_percentages = [0.90, 0.75, 0.50, 0.25, 0.10]
    
    for percentage in truncation_percentages:
        try:
            target_tokens = max(1, int(max_tokens * percentage))
            truncated_chunk = _truncate_to_max_tokens(embedding_manager, chunk, target_tokens)
            
            if not truncated_chunk or len(truncated_chunk.strip()) == 0:
                continue
            
            embedding = embedding_manager.get_embedding(truncated_chunk)
            
            # Log success for first few attempts, then suppress output
            if retry_count <= 10:
                print(f"   🆘 Emergency truncation succeeded for chunk {chunk_index} "
                      f"(truncated to {target_tokens} tokens, {int(percentage*100)}% of max)")
            elif retry_count == 11:
                print(f"   ... (continuing emergency truncations silently)")
            
            return embedding
        except Exception:
            continue
    
    return None

def embed_chunks_with_progressive_truncation(
    embedding_manager: EmbeddingManager,
    chunk_texts: List[str],
    max_tokens: int
) -> List[np.ndarray]:
    """
    Generate embeddings for text chunks with automatic fallback truncation.
    
    Processes chunks individually to avoid batch size issues. When a chunk exceeds
    token limits, progressively truncates it (7→5→3→2→1 words) until successful.
    Skips chunks that fail even after maximum truncation.
    
    Args:
        embedding_manager: Manager instance for generating embeddings
        chunk_texts: List of text chunks to embed
        max_tokens: Maximum token limit for the embedding model
        
    Returns:
        List of embedding arrays for successfully processed chunks
        
    Raises:
        Exception: If no chunks could be processed successfully
    """
    print(f"🔄 Processing {len(chunk_texts)} chunks with max_tokens={max_tokens}")
    print(f"⚠️  Using individual processing with progressive truncation fallback")
    
    all_embeddings = []
    truncation_count = 0

    for i, chunk in enumerate(chunk_texts):
        try:
            # Attempt normal embedding generation
            embedding = embedding_manager.get_embedding(chunk)
            all_embeddings.append(embedding)

            # Progress reporting every 1000 chunks
            if (i + 1) % 1000 == 0:
                print(f"   ✅ Processed {i + 1}/{len(chunk_texts)} chunks")

        except Exception as e:
            truncation_count += 1
            chunk_index = i + 1
            
            # Log the error with token information if available
            _extract_token_info_from_error(str(e), chunk, chunk_index)
            
            # Attempt progressive truncation as fallback
            embedding = _attempt_progressive_truncation(
                embedding_manager, chunk, chunk_index, truncation_count, max_tokens
            )
            
            if embedding is not None:
                all_embeddings.append(embedding)
            else:
                print(f"   ❌ Skipping chunk {chunk_index} - all truncation attempts failed")

    # Validate results
    if not all_embeddings:
        raise Exception("No chunks could be processed - all failed")
    
    # Report final statistics
    success_rate = (len(all_embeddings) / len(chunk_texts)) * 100
    print(f"✅ Embedding generation completed:")
    print(f"   ✅ Successful: {len(all_embeddings)}/{len(chunk_texts)} ({success_rate:.1f}%)")
    print(f"   🆘 Required truncation: {truncation_count}")
    
    return all_embeddings

def get_pre_existing_vector_db_embedding_dim(vector_db_path: str , database_type: str) -> Tuple[Optional[int], Optional[VectorDBInterface]]:
    """
    Retrieve the embedding dimension from an existing vector database.
    
    Args:
        vector_db_path: Path to the persisted vector database
        database_type: Type of the vector database (e.g., 'faiss')
    Returns:
        Embedding dimension if available, None otherwise
    """
    # Retrieving pre-existing vector database embedding dimension
    print("Retrieving pre-existing vector database embedding dimension...")
    vector_db = VectorDBFactory.get_vector_db(db_path=vector_db_path)
    embedding_dim = VectorDBFactory.get_actual_db_embedding_dim(vector_db)
    if embedding_dim is None or embedding_dim <=0:
        print(f"❌ Failed to get embedding dimension from vector database because \n{database_type} Vector database is not at path {vector_db_path}")
    
    return embedding_dim , vector_db 

def get_embedding_manager_and_max_tokens(embedding_tester: EmbeddingModelInterface, vector_db_embedding_dim: Optional[int] = None) -> Tuple[EmbeddingManager, int , int]:
    """
    Initialize the embedding manager and determine optimal max tokens.
     
    Args:
        embedding_tester: An instance of EmbeddingModelInterface
        vector_db_embedding_dim: Optional embedding dimension from vector database
    Returns:
        Tuple of EmbeddingManager instance and optimal max tokens
        """
    safe_embedding_dim = embedding_tester.safe_embedding_dim
    max_tokens = embedding_tester.max_tokens
    embedding_manager = EmbeddingManager( 
        embedding_dim=vector_db_embedding_dim if vector_db_embedding_dim else safe_embedding_dim,
        embedding_model_instance=embedding_tester
    )
    
    # Verify the dimension (optional double-check)
    verified_dim = embedding_manager.get_actual_embedding_dimension()
    if verified_dim != safe_embedding_dim:
        print(f"⚠️  Dimension mismatch! Detected: {safe_embedding_dim}, Verified: {verified_dim}")
        safe_embedding_dim = verified_dim  # Use the verified dimension

    return embedding_manager, max_tokens , safe_embedding_dim

def get_embedded_document_chunks_from_path(document_path :str, embedding_manager: EmbeddingManager, 
                                           max_tokens: int, chunk_func: callable) -> Tuple[List[np.ndarray], List[Document]]:
    """
    Load, chunk, and embed documents from a specified path.
    
    Args:
        document_path: Path to the directory containing documents
        embedding_manager: Manager for generating embeddings
        max_tokens: Maximum tokens allowed by the embedding model
        use_llama_index_chunking: Whether to use LlamaIndex for chunking
    Returns:
        Tuple of list of embedded text chunks and list of Document objects
    """
    chunk_texts, document_chunks  = chunk_func( document_path, max_tokens, embedding_manager=embedding_manager)
    print(f"🔄 Getting embeddings for {len(chunk_texts)} text chunks...")
    chunk_embeddings= embed_chunks_with_progressive_truncation(embedding_manager,chunk_texts,max_tokens)
    # If we lost some chunks, update our data structures
    if len(chunk_embeddings) < len(chunk_texts):
        print(f"⚠️  Some chunks were skipped. Using {len(chunk_embeddings)} out of {len(chunk_texts)} chunks")
        print(f"⚠️  Warning: Document-embedding alignment may be approximate")
        print(f"🔧 Adjusting document chunks to match embeddings count...")
        document_chunks = document_chunks[:len(chunk_embeddings)]
    return chunk_embeddings, document_chunks

def persist_vector_database(collection_name:str, vector_db_persisted_path:str, embedding_manager: EmbeddingManager,
                            max_tokens:int,  embedding_manager_dim:int, chunker_function: callable ,
                            source_path:str, use_gpu:bool =True):
    """
    Persist a vector database to disk.
    
    Args:
        collection_name: Name of the collection within the vector database
        vector_db_path: Path where the vector database should be persisted
        source_path: Source path for the vector database
        use_gpu: Whether to use GPU acceleration
    """
    
    print(f"📥 Loading documents from {source_path} and embedding chunks with dimension {embedding_manager_dim}...")
    # Load, chunk, and embed documents uing the embedding manager's supported max_tokens and embedding dimensions
    chunked_embeddings, document_chunks = get_embedded_document_chunks_from_path(
        document_path=source_path,
        embedding_manager=embedding_manager,
        max_tokens=max_tokens,
        chunk_func= chunker_function
    )

    # Create/load vector database using factory with CORRECT dimension, collection_name, GPU setting and Path
    vector_db = VectorDBFactory.create_vector_db(
        db_type=DATABASE_TYPE,
        embedding_dim=safe_embedding_dim,  # Use detected dimension
        persist_path=vector_db_persisted_path,
        use_gpu=use_gpu,
        collection_name=collection_name,
    )
    vector_db.add_documents(document_chunks, chunked_embeddings)

    # Save the database
    vector_db.save()

    return vector_db 

def re_initialize_vector_database(collection_name:str, vector_db_persisted_path:str, database_type:str) -> Optional[int]:
    """
    Re-initialize the vector database by removing existing collection if it exists.
    
    Args:
        collection_name: Name of the collection within the vector database
        vector_db_path: Path where the vector database is persisted
        database_type: Type of the vector database (e.g., 'faiss')
    Returns:
        Tuple of embedding dimension if available and vector database instance
    """   
   # Retrieve pre-existing vector database embedding dimension if available
    vector_embedding_dim , vector_db = get_pre_existing_vector_db_embedding_dim(vector_db_path=vector_db_persisted_path , database_type=database_type)
    # Remove existing collection to start fresh ingestion
    if vector_db and vector_embedding_dim:
        # Delete collection_name if it exists to start ingestion from scratch
        vector_db.delete_collection(collection_name=collection_name, persist_path=vector_db_persisted_path)
    else:
        vector_embedding_dim = None  # Will be detected later
        vector_db = None # Will be created/loaded later with the correct dimensions from embedding manager
    return vector_embedding_dim

if __name__ == "__main__":
    root_path = "/home/jmitchall/vllm-srv"
    root_data_path = f"{root_path}/data"
    document_collections = [
        "vtm",
        "dnd_dm",
        "dnd_mm",
        "dnd_raven",
        "dnd_player",
    ]
    database_types = [
        "qdrant",
        "faiss",
        "chroma"
    ]

    chunker_functions = {
     "langchain": load_langchain_texts,
     "llamaindex": load_llamaIndex_texts
    }
    #consolidate_collections_to_all(root_data_path=root_data_path, document_collections=document_collections)
    
    embedding_model = "BAAI/bge-large-en-v1.5"
    embedding_tester: EmbeddingModelInterface = HuggingFaceLocalModel(model_name=embedding_model ,safety_level="max") 

    for db_type in database_types:
        DATABASE_TYPE = db_type.lower()
        use_llama_index_chunking = False
        use_gpu: bool = True

        # Check for vector database availability
        if not VectorDBFactory.get_available_vector_databases(validated_db = db_type):
            print (f" Unable to import {db_type}")
            continue

        for chunk_lib, chunker_func in chunker_functions.items():
            # for each document collection, persist the vector database
            for collection_name in document_collections:
                directory_path = f"{root_data_path}/{collection_name}"
                vector_db_persisted_path =f"{root_path}/custom/{DATABASE_TYPE}/{chunk_lib}_{collection_name}"
                
                # Re-initialize vector database and get pre-existing embedding dimension if available
                vector_embedding_dim = re_initialize_vector_database(
                    collection_name=collection_name,
                    vector_db_persisted_path=vector_db_persisted_path,
                    database_type=DATABASE_TYPE
                )

                # Initialize embedding manager based on detected or existing dimension of embedding_model name 
                embedding_manager, max_tokens , safe_embedding_dim = get_embedding_manager_and_max_tokens( 
                    embedding_tester= embedding_tester,
                    vector_db_embedding_dim=vector_embedding_dim
                )

        
                # Persist the vector database with embedded documents
                vector_db = persist_vector_database(
                    collection_name=collection_name,
                    vector_db_persisted_path=vector_db_persisted_path,
                    embedding_manager=embedding_manager,
                    max_tokens=max_tokens,
                    embedding_manager_dim=safe_embedding_dim,
                    chunker_function=chunker_func,
                    source_path=directory_path,
                    use_gpu=use_gpu
                )


    print("🎉 Ingestion complete!")


    

    