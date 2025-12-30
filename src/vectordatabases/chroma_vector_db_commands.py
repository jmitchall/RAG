from langchain_chroma import Chroma
from langchain.schema import Document
from typing import List, Optional
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun


def get_chroma_retriever(
    chroma_vectorstore_ptr: Chroma,
    k=5,
    score_threshold: float = None,
    search_type: str = "mmr",
    fetch_k: int = 20,
    lambda_mult: float = 0.5
) -> BaseRetriever:
    """
    Create a LangChain retriever from a Chroma vector store.
    Simple Explanation:
    - This function creates a retriever that can fetch relevant documents from a Chroma vector store.
    - Supports MMR for diverse results.
    
    Example:
        Input: chroma_vectorstore_ptr, k=5, search_type="mmr"
        Output: LangChain BaseRetriever object  
    Args:
        chroma_vectorstore_ptr (Chroma): Pointer to the Chroma vector store
        k (int): Number of documents to return
        score_threshold (float): NOT USED - Chroma returns L2 distances, not 0-1 scores
        search_type (str): "similarity" for pure relevance, "mmr" for diversity
        fetch_k (int): For MMR - number of candidates before diversity filtering
        lambda_mult (float): For MMR - balance between relevance (1.0) and diversity (0.0)
    Returns:
        BaseRetriever: A LangChain retriever object.
    """ 
    # Create custom retriever wrapper that includes similarity scores
    class ChromaRetrieverWrapper(BaseRetriever):
        vectorstore: Chroma
        top_k: int = 5
        score_threshold: Optional[float] = None
        search_type: str = "similarity"
        fetch_k: int = 20
        lambda_mult: float = 0.5
        
        def get_relevant_documents(self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None) -> List[Document]:
            documents = []
            
            if self.search_type == "mmr":
                # Use MMR for diverse, relevant results
                print(f"   🔍 Using MMR search (fetch_k={self.fetch_k}, lambda={self.lambda_mult})")
                docs = self.vectorstore.max_marginal_relevance_search(
                    query,
                    k=self.top_k,
                    fetch_k=self.fetch_k,
                    lambda_mult=self.lambda_mult
                )
                
                # Compute similarity scores for MMR results to show relevance
                # Get the query embedding
                query_embedding = self.vectorstore._embedding_function.embed_query(query)
                
                for idx, doc in enumerate(docs):
                    doc.metadata['search_type'] = 'mmr'
                    doc.metadata['similarity_score'] = None  # MMR doesn't provide scores
                    self.update_doc_metadata_cosine_score(doc, query_embedding, idx)
                    source = doc.metadata.get('source', 'unknown')
                    cosine_similarity = doc.metadata.get('cosine_similarity', None)
                    if cosine_similarity is not None:
                        print(f"   📄 MMR result | Source: {source} | Cosine Similarity: {cosine_similarity:.4f}")
                    else:
                        print(f"   📄 MMR result | Source: {source}")
                    documents.append(doc)
            else:
                # Use similarity_search_with_score to get raw distances without normalization warnings
                search_results = self.vectorstore.similarity_search_with_score(query, k=self.top_k)
                
                for doc, distance in search_results:
                    # Chroma returns L2 distances (lower = more similar)
                    # Note: Score threshold filtering won't work well with raw distances
                    # Store both distance and inverted score for compatibility
                    doc.metadata['distance'] = distance
                    doc.metadata['similarity_score'] = -distance  # Negative distance (less negative = more similar)
                    doc.metadata['search_type'] = 'similarity'
                    source = doc.metadata.get('source', 'unknown')
                    print(f"   📄 Distance: {distance:.4f} (lower=better) | Source: {source}")
                    documents.append(doc)
            
            return documents
        
        def update_doc_metadata_cosine_score(self, doc: Document, query_embedding: List[float], idx: int):
            """
            Update document metadata with cosine similarity score to the query.
            Args:
                doc (Document): The document to update
                query_embedding (List[float]): The embedding of the query
                idx (int): Index of the document in the result set
            """
            from numpy import dot
            from numpy.linalg import norm
            import numpy as np
            
            doc_embedding = self.vectorstore._embedding_function.embed_query(doc.page_content)
            # Compute cosine similarity
            cosine_sim = dot(np.array(query_embedding), np.array(doc_embedding)) / (norm(np.array(query_embedding)) * norm(np.array(doc_embedding)) + 1e-10)
            doc.metadata['similarity_score'] = cosine_sim
            doc.metadata['result_index'] = idx + 1  # 1-based index for display


    return ChromaRetrieverWrapper(
        vectorstore=chroma_vectorstore_ptr,
        top_k=k,
        score_threshold=score_threshold,
        search_type=search_type,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult
    )

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
