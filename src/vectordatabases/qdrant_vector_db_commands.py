from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore as Qdrant
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun

from langchain.schema import Document
from typing import List, Optional


class QdrantClientSmartPointer:
    """
    A smart pointer class to manage QdrantClient connections.
    Simple Explanation:
    - This class helps manage the connection to a Qdrant database.
    - It ensures that the connection is properly closed when no longer needed.
    Example:
        client_pointer = QuadrantClientSamartPointer("/path/to/local/folder")
        qdrant_client = client_pointer.get_client()
        # Use qdrant_client...
        client_pointer.close()  # Close connection when done
    Args:
        vector_db_persisted_path (str): Path to the local folder for Qdrant storage.
    """
    def __init__(self, vector_db_persisted_path: str):
        self.vector_db_persisted_path = vector_db_persisted_path
        self._client = None

    def get_client(self) -> QdrantClient:
        """Get the Qdrant client, creating it if it doesn't exist."""
        if self._client is None:
            self._client = QdrantClient(path=self.vector_db_persisted_path)
        return self._client

    def close(self):
        """Close the Qdrant client connection if it exists."""
        if self._client is not None:
            try:
                print(f"🧹 Closing Qdrant client connection at {self.vector_db_persisted_path} ...")
                self._client.close()
            except Exception as e:
                print(f"⚠️ Warning: Exception occurred while closing Qdrant client close: {e}")   
            self._client = None
    
    def __del__(self):
        """Ensure the client is closed when the smart pointer is deleted."""
        self.close()



def get_quadrant_client(vector_db_persisted_path):
    """
    Create and return a Qdrant client connected to a local persistent storage.
    Simple Explanation:
    - This function sets up a connection to a Qdrant vector database.
    - It uses a local folder to store the database files.
    Example:
        Input: "/path/to/local/folder"
        Output: Qdrant client object connected to that folder.
    Args:
        vector_db_persisted_path (str): Path to the local folder for Qdrant storage.
    Returns:
        QdrantClient: A client object to interact with the Qdrant database.
    """
    # Create Qdrant client with local persistent storage
    qdrant_client = QdrantClientSmartPointer(vector_db_persisted_path=vector_db_persisted_path)
    return qdrant_client


def quadrant_does_collection_exist(
    qdrant_client_ptr: QdrantClientSmartPointer,
    collection_name: str,
) -> bool:
    """
    Check if a specific collection exists in the Qdrant database.
    Simple Explanation:
    - This function checks if a named collection (like a folder) exists in the Qdrant database.
    Example:
        Input: qdrant_client, "my_collection"
        Output: True (if it exists) or False (if it doesn't)
    Args:
        qdrant_client (QdrantClient): The Qdrant client object.
        collection_name (str): The name of the collection to check.
    Returns:
        bool: True if the collection exists, False otherwise.
    """
    try:
        # Try to get information about the collection
        qdrant_client_ptr.get_client().get_collection(collection_name=collection_name)
        return True  # Collection exists
    except Exception:
        return False  # Collection does not exis

def qdrant_only_add_documents_not_in_collection(
    qdrant_client_ptr: QdrantClientSmartPointer,
    collection_name: str,
    documents: List[Document],
) -> List[Document]:
    """
    Filter out documents that already exist in the Qdrant collection.
    Simple Explanation:
    - This function checks which documents are already in the Qdrant collection.
    - It returns only the new documents that are not already stored.
    - Uses document content hash to identify duplicates.
    Example:
        Input: qdrant_client, "my_collection", [doc1, doc2, doc3]
        Output: [doc2, doc3] (if doc1 is already in the collection)
    Args:
        qdrant_client (QdrantClient): The Qdrant client object.
        collection_name (str): The name of the collection to check against.
        documents (List[Document]): List of documents to filter.
    Returns:
        List[Document]: List of documents not already in the collection.
    """
    # If collection doesn't exist, all documents are new
    if not quadrant_does_collection_exist(qdrant_client_ptr, collection_name):
        return documents
    
    try:
        # Get all existing points (documents) from the collection
        # We'll scroll through all points to get their payloads
        existing_points, _ = qdrant_client_ptr.get_client().scroll(
            collection_name=collection_name,
            limit=10000,  # Get up to 10k points at once
            with_payload=True,
            with_vectors=False  # We don't need the vectors, just the metadata
        )
        
        # Create a set of existing document identifiers for fast lookup
        # We'll use a hash of the page_content as the identifier
        existing_content_hashes = set()
        
        for point in existing_points:
            # Get the payload (metadata) from each point
            payload = point.payload
            
            # Check if page_content exists in the payload
            if payload and 'page_content' in payload:
                content = payload['page_content']
                # Create a hash of the content for comparison
                content_hash = hash(content)
                existing_content_hashes.add(content_hash)
        
        # Filter out documents that already exist
        # Compare each document's content hash against existing hashes
        new_documents = []
        for doc in documents:
            content_hash = hash(doc.page_content)
            if content_hash not in existing_content_hashes:
                new_documents.append(doc)
        
        # Print statistics for user feedback
        duplicate_count = len(documents) - len(new_documents)
        if duplicate_count > 0:
            print(f"🔍 Found {duplicate_count} duplicate document(s), adding {len(new_documents)} new document(s)")
        
        return new_documents
        
    except Exception as e:
        # If there's any error during checking, log it and return all documents
        # Better to add duplicates than to lose data
        print(f"⚠️  Warning: Could not check for duplicates: {e}. Adding all documents.")
        return documents

def qdrant_create_from_documents(qdrant_client_ptr: QdrantClientSmartPointer, collection_name: str, documents: List[Document],
    embeddings):
    """
    Create or load a Qdrant vector database from documents.
    Simple Explanation:
    - This function either creates a new Qdrant collection from documents or loads an existing one.
    Example:
        Input: qdrant_client, "my_collection", "/path/to/folder", documents, embeddings
        Output: Qdrant vector store object
    Args:
        qdrant_client (QdrantClient): The Qdrant client object.
        collection_name (str): The name of the collection to create or load.
        vector_db_persisted_path (str): Path to the local folder for Qdrant storage.
        documents (List[Document]): List of documents to add to the collection.
        embeddings: Embedding model to use for vectorization.
    Returns:
        Qdrant: The Qdrant vector store object.
    """
    vector_db: Qdrant = None
    # Add documents to the collection (creates collection if it doesn't exist)
    if quadrant_does_collection_exist(qdrant_client_ptr=qdrant_client_ptr, collection_name=collection_name):     
        # Check for duplicate documents before adding new ones
        filtered_documents = qdrant_only_add_documents_not_in_collection(
            qdrant_client_ptr,
            collection_name,
            documents
        )
        if filtered_documents:
            vector_db = Qdrant(
                client=qdrant_client_ptr.get_client(),
                collection_name=collection_name,
                embedding=embeddings,
            )   
            vector_db.add_documents(filtered_documents)
            print(f"✅ Added {len(filtered_documents)} new document(s) to Qdrant collection '{collection_name}'")
        else:
            print(f"ℹ️  No new documents to add - all {len(documents)} document(s) already exist in collection '{collection_name}'")
    else:
        # Collection doesn't exist, create it manually using the client
        # First, get embedding dimension by creating a sample embedding
        sample_embedding = embeddings.embed_query("sample text")
        vector_size = len(sample_embedding)
        
        # Create the collection with proper vector configuration
        qdrant_client_ptr.get_client().create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print(f"📦 Created empty Qdrant collection '{collection_name}' with vector size {vector_size}")
        vector_db = Qdrant(
            client=qdrant_client_ptr.get_client(),
            collection_name=collection_name,
            embedding=embeddings,
        )
        
        # Now add the documents
        vector_db.add_documents(documents)
        print(f"✅ Added {len(documents)} document(s) to new collection '{collection_name}'")
        
        # Note: Data is automatically persisted to disk when using QdrantClient with path parameter
        # No explicit persist() call is needed
        print(f"💾 Data automatically persisted to disk (Qdrant local storage)")

    return vector_db
    

def get_qdrant_retriever(qdrant_client_ptr, collection_name, embeddings, k=5, 
                        score_threshold: float = None, search_type: str = "mmr") -> BaseRetriever:
    """
    Get a Qdrant retriever for searching the vector database.
    Simple Explanation:
    - This function sets up a retriever to search through the Qdrant vector database.
    - Supports filtering by minimum similarity score and MMR for diverse results.
    
    Example:
        Input: qdrant_client, "my_collection", embeddings, k=5, score_threshold=0.7
        Output: Qdrant retriever object
    Args:
        qdrant_client (QdrantClient): The Qdrant client object.
        collection_name (str): The name of the collection to search.
        embeddings: Embedding model used for vectorization.
        k (int): Number of top results to retrieve.
        score_threshold (float): Minimum similarity score (0-1). Only return docs above this threshold.
        search_type (str): "similarity" for pure relevance, "mmr" for diversity
    Returns:
        Qdrant retriever object.
    """
    print(f"🔍 Creating Qdrant vectorstore from client type: {type(qdrant_client_ptr.get_client())}")
    
    loaded_vectorstore = Qdrant(
        client=qdrant_client_ptr.get_client(),
        collection_name=collection_name,
        embedding=embeddings,
    )
    
    print(f"🔍 Created vectorstore type: {type(loaded_vectorstore)}")
    print(f"🔍 Vectorstore has similarity_search: {hasattr(loaded_vectorstore, 'similarity_search')}")
    
    # Workaround: Use similarity_search directly instead of as_retriever()
    # Create a simple wrapper that acts as a retriever
    class QdrantRetrieverWrapper(BaseRetriever):

        vectorstore: Qdrant
        embeddings: object  # Store embeddings for computing scores
        top_k: int = 5
        score_threshold: Optional[float] = None
        search_type: str = "mmr"
        
        def get_relevant_documents(self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None) -> List[Document]:
            print(f"🔍 get_relevant_documents called with vectorstore type: {type(self.vectorstore)}")

            documents = []
            
            if self.search_type == "mmr":
                # Use MMR for diverse results (Qdrant supports this)
                print(f"   🔍 Using MMR search")
                docs = self.vectorstore.max_marginal_relevance_search(query, k=self.top_k)
                query_embedding = self.embeddings.embed_query(query)
                for idx, doc in enumerate(docs):
                        doc.metadata['search_type'] = 'mmr'
                        doc.metadata['similarity_score'] = None  # MMR doesn't provide scores
                        self.update_doc_metadata_cosine_score(doc, query_embedding, idx)
                        source = doc.metadata.get('source', 'unknown')
                        similarity_score = doc.metadata.get('similarity_score', None)
                        if similarity_score is not None:
                            print(f"   📄 MMR result | Source: {source} | Similarity Score: {similarity_score:.4f}")
                        else:
                            print(f"   📄 MMR result | Source: {source}")
                        documents.append(doc)
            else:
                search_results = self.vectorstore.similarity_search_with_relevance_scores(query, k=self.top_k)
                
                for doc, score in search_results:
                    # Apply score threshold filtering if specified
                    if self.score_threshold is not None and score < self.score_threshold:
                        print(f"   ⏭️  Skipping doc with score {score:.4f} < threshold {self.score_threshold}")
                        continue
                        
                    doc.metadata['similarity_score'] = score
                    doc.metadata['search_type'] = 'similarity'
                    source = doc.metadata.get('source', 'unknown')
                    print(f"   📄 Similarity: {score:.4f} | Source: {source}")
                    documents.append(doc)
            
            
            if self.score_threshold and len(documents) == 0:
                print(f"   ⚠️  No documents met the score threshold of {self.score_threshold}")
            
            return documents
        
        def update_doc_metadata_cosine_score(self, doc: Document, query_embedding: List[float], idx: int):
                """
                Update document metadata with cosine similarity score to the query.
                Args:
                    doc (Document): The document to update
                    query_embedding (List[float]): The embedding of the query
                    idx (int): Index of the document in the result set
                """
                # Get document embedding from vectorstore
                # Compute cosine similarity or distance
                try:
                    doc_embedding = self.embeddings.embed_query(doc.page_content)
                    # Compute cosine similarity
                    from numpy import dot
                    from numpy.linalg import norm
                    import numpy as np
                    
                    cosine_similarity = dot(np.array(query_embedding), np.array(doc_embedding)) / (norm(np.array(query_embedding)) * norm(np.array(doc_embedding)))
                    doc.metadata['similarity_score'] = float(cosine_similarity)
                except Exception as e:
                    print(f"   ⚠️  Warning: Could not compute cosine similarity for doc idx {idx}: {e}")
                    doc.metadata['similarity_score'] = None


        
        def format_list_documents_as_string(self, **kwargs) -> str:
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
                context += doc.page_content + "\n\n"
            return context
    
    retriever = QdrantRetrieverWrapper(
        vectorstore=loaded_vectorstore,
        embeddings=embeddings,  # Pass embeddings explicitly
        top_k=k,
        score_threshold=score_threshold,
        search_type=search_type
    )
    return retriever



    