from langchain_chroma import Chroma
from langchain.schema import Document
from typing import List
from langchain_core.embeddings import Embeddings

def create_chroma_vectore_store(collection_name: str,  vector_db_persisted_path: str)-> Chroma:
    """     
    Create a Chroma vector database.
    Simple Explanation:
    - This function creates a new Chroma collection.
    Example:
        Input: "my_collection", "/path/to/folder"
        Output: Chroma vector store object  
    Args:

        collection_name (str): The name of the collection to create.
        vector_db_persisted_path (str): Path to the local folder for Chroma storage.
    Returns:
         Chroma: A Chroma vector store object.
    """
    chroma_vectorstore = Chroma(
        persist_directory=vector_db_persisted_path,
        collection_name=collection_name,
    )
    print(f"✅ Chroma vector store loaded from {vector_db_persisted_path}")
    return chroma_vectorstore
    
def chroma_does_collection_exist( collection_name: str, vector_db_persisted_path: str , embeddings:Embeddings = None) -> bool:
    """
    Check if a specific collection exists in the Chroma database.
    Simple Explanation:
    - This function checks if a named collection (like a folder) exists in the Chroma database.
    Example:                    
        Input: "my_collection", "/path/to/local/folder"
        Output: True (if it exists) or False (if it doesn't)
    Args:   
        collection_name (str): The name of the collection to check.
        vector_db_persisted_path (str): Path to the local folder for Chroma storage.
    Returns:    
        bool: True if the collection exists, False otherwise.
    """
    try:
        # Try to create a Chroma instance to check for the collection
        chroma_vectorstore = create_chroma_vectore_store(collection_name, vector_db_persisted_path, embeddings)
        # Attempt to access the collection to verify its existence
        _ = chroma_vectorstore.get()
        return True , chroma_vectorstore  # Collection exists
    except Exception:
        return False, None  # Collection does not exist   
    

def chroma_create_from_documents(collection_name: str,
    vector_db_persisted_path: str, documents: List[Document],
    embeddings:Embeddings)-> Chroma:
    """
    Create or load a Chroma vector database from documents.
    Simple Explanation:
    - This function either creates a new Chroma collection from documents or loads an existing one.
    Example:
        Input: "my_collection", "/path/to/folder", documents, embeddings
        Output: Chroma vector store object
    Args:
        collection_name (str): The name of the collection to create or load.
        vector_db_persisted_path (str): Path to the local folder for Chroma storage.
        documents (List[Document]): List of Document objects to add to the vector store.
        embeddings: Embedding model to convert documents into vectors.
    Returns:
        Chroma: A Chroma vector store object.
    """
    exists, chroma_vectorstore = chroma_does_collection_exist(collection_name, vector_db_persisted_path , embeddings)
    if exists:
        print(f"✅ Chroma collection '{collection_name}' already exists at {vector_db_persisted_path}") 
        #UPDATE THE VECTOR STORE WITH NEW DOCUMENTS AND ENSURE NO REPEATS
        chroma_vectorstore.add_documents(documents)
    else:
        chroma_vectorstore = Chroma.from_documents(
            documents,
            embeddings,
            persist_directory=vector_db_persisted_path,
            collection_name=collection_name
        )
    # Note: Data is automatically persisted when persist_directory is provided
    print(f"✅ Chroma vector store created and saved to {vector_db_persisted_path}")
    return chroma_vectorstore
    

def get_chroma_vectorstore(collection_name: str,  vector_db_persisted_path: str, retriever_embeddings: Embeddings) -> Chroma:
    """ 
    Get a Chroma vector store for the specified collection.
    Simple Explanation:
    - This function retrieves a Chroma vector store for a given collection name and path.
    Example:


        Input: "my_collection", "/path/to/folder", embeddings
        Output: Chroma vector store object
    Args:
        collection_name (str): The name of the collection to retrieve.
        vector_db_persisted_path (str): Path to the local folder for Chroma storage.
        retriever_embeddings (Embeddings): Embedding model used for the vector store.
    Returns:

        Chroma: A Chroma vector store object.
    """
                        
    return Chroma(
        persist_directory=vector_db_persisted_path,
        collection_name=collection_name,
        embedding_function=retriever_embeddings,
        )
