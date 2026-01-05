from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from typing import List, Optional
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from llama_index.vector_stores.faiss import FaissVectorStore
#https://developers.llamaindex.ai/python/examples/vector_stores/faissindexdemo/

class FAISSVectorStore:
    def __init__(self , faiss_vectorstore: FAISS, persist_path: str = None):
        self.faiss_vectorstore = faiss_vectorstore
        self.persist_path = persist_path

    def save(self):
        # Clean up resources if needed
        if self.persist_path:
            try:
                print(f"🧹 Saving  FAISS vectorstore resources at {self.persist_path}")
                self.faiss_vectorstore.save_local(self.persist_path)
                print(f"✅ FAISS vectorstore saved successfully")
            except ImportError as e:
                pass
            except Exception as e:
                print(f"⚠️  Error saving FAISS vectorstore: {e}")
    
    def __del__(self):
        try:
            if hasattr(self, 'persist_path') and self.persist_path:
                self.save()
            if hasattr(self, 'faiss_vectorstore'):
                del self.faiss_vectorstore
        except Exception as e:
            # Silently handle exceptions in destructor to avoid polluting output
            pass

def get_faiss_retriever(
    faiss_vectorstore_ptr: FAISSVectorStore,
     k=5,
     search_type: str = "mmr",  # "similarity" or "mmr"
     fetch_k: int = 20,  # For MMR: fetch more candidates before diversity filtering
     lambda_mult: float = 0.5  # For MMR: 0=max diversity, 1=max relevance
) -> BaseRetriever:
    """
    Create a LangChain retriever from a FAISS vector store.
    Simple Explanation:
    - This function creates a retriever that can fetch relevant documents from a FAISS vector store.
    - Supports both similarity search and MMR (Maximal Marginal Relevance) for diverse results.
    
    Example:
        Input: faiss_vectorstore_ptr, k=5, search_type="mmr"
        Output: LangChain BaseRetriever object

    Args:
        faiss_vectorstore_ptr (FAISSVectorStore): Pointer to the FAISS vector store
        k (int): Number of documents to return
        search_type (str): "similarity" for pure relevance, "mmr" for diversity
        fetch_k (int): For MMR - number of candidates to fetch before diversity filtering
        lambda_mult (float): For MMR - balance between relevance (1.0) and diversity (0.0)
    Returns:

        BaseRetriever: A LangChain retriever object.
    """ 
    # Create custom retriever wrapper that includes similarity scores
    class FAISSRetrieverWrapper(BaseRetriever):
            vectorstore: FAISS
            embeddings: object  # Store embeddings for computing scores
            top_k: int = 5
            search_type: str = "mmr"
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
                    query_embedding = self.embeddings.embed_query(query)
                    # MMR doesn't return scores, so we note that
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
                    # Use similarity_search_with_score to get raw distances without normalization warnings
                    search_results = self.vectorstore.similarity_search_with_score(query, k=self.top_k)
                    
                    for doc, distance in search_results:
                        # FAISS returns L2 distances (lower = more similar)
                        # Convert to similarity score: we'll store the raw distance but note the inversion
                        # For display purposes, you can normalize: similarity = 1 / (1 + distance)
                        # But we'll keep raw distance in metadata with negative sign to match original behavior
                        doc.metadata['similarity_score'] = -distance  # Negative distance (less negative = more similar)
                        doc.metadata['distance'] = distance  # Raw distance (lower = more similar)
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
                # Get document embedding from vectorstore
                # Compute cosine similarity or distance
                try:
                    doc_embedding = self.embeddings.embed_query(doc.page_content)
                    # Compute cosine similarity
                    from numpy import dot
                    from numpy.linalg import norm
                    import numpy as np
                    
                    cosine_similarity = dot(np.array(query_embedding), np.array(doc_embedding)) / (norm(np.array(query_embedding)) * norm(np.array(doc_embedding)))
                    doc.metadata['similarity_score'] = cosine_similarity
                    print(f"      🔢 Similarity Score: {cosine_similarity:.4f}")
                except Exception as e:
                    print(f"      ⚠️ Error computing cosine similarity for doc idx {idx}: {e}")



    # Get embeddings from vectorstore
    embeddings = faiss_vectorstore_ptr.faiss_vectorstore.embedding_function
    
    return FAISSRetrieverWrapper(
        vectorstore=faiss_vectorstore_ptr.faiss_vectorstore,
        embeddings=embeddings,  # Pass embeddings explicitly
        top_k=k,
        search_type=search_type,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult
    )

def create_faiss_vectorstore(vector_db_persisted_path: str, embeddings: Embeddings) -> FAISSVectorStore:
    """
    Create an empty FAISS vector database.
    Simple Explanation:
    - This function creates a new, empty FAISS vector store that can be used to store document embeddings.
    Example:
        Input: "/path/to/folder", embeddings
        Output: FAISS vector store object
    Args:
        vector_db_persisted_path (str): Path to the local folder for FAISS storage.
        embeddings: Embedding model to convert documents into vectors.
    Returns:
        FAISS: A FAISS vector store object.
    """
    try:
        if not vector_db_persisted_path:
            raise ValueError("Vector DB persisted path is required.")
        local_file = vector_db_persisted_path
        
        print(f"🔍 Testing loading of persisted vector store from path: {local_file} ...")
        faiss_vectorstore = FAISS.load_local(
            local_file,
            embeddings,
            # FAISS stores its vector index using Python's pickle format for serialization. 
            # Pickle can execute arbitrary code during deserialization, 
            # making it a security risk if you load pickle files from untrusted sources 
            # (an attacker could embed malicious code in the pickle file that runs when you load it).
            # LangChain's FAISS implementation now requires explicit acknowledgment of this risk via
            allow_dangerous_deserialization=True
        )
        print(f"✅ FAISS vector store loaded from {local_file}")
        faiss_vectorstore_wrapper = FAISSVectorStore(faiss_vectorstore, vector_db_persisted_path)
    except Exception as e:
        print(f"⚠️  WARNING unable to load FAISS vector store: {e}")
        # If loading fails (e.g., store doesn't exist), create a new one with a dummy document
        print(f"📝 Creating new empty FAISS vector store at {vector_db_persisted_path}")
        dummy_doc = Document(page_content="Initialization document", metadata={"source": "init"})
        faiss_vectorstore = FAISS.from_documents(
            [dummy_doc],
            embeddings
        )
        # Remove the dummy document immediately
        faiss_vectorstore.delete([faiss_vectorstore.index_to_docstore_id[0]])
        # Save the newly created vector store to disk for future use
        faiss_vectorstore.save_local(vector_db_persisted_path)
        faiss_vectorstore_wrapper = FAISSVectorStore(faiss_vectorstore, vector_db_persisted_path)
        print(f"✅ FAISS vector store created and saved to {vector_db_persisted_path}")
    return faiss_vectorstore_wrapper


def faiss_create_from_documents(vector_db_persisted_path: str, documents: List[Document],
    embeddings: Embeddings )-> FAISSVectorStore:
    """
    Create or load a FAISS vector database from documents.
    Simple Explanation:
    - This function either creates a new FAISS vector store from documents or loads an existing one.
    Example:
        Input: "/path/to/folder", documents, embeddings
        Output: FAISS vector store object
    Args:
        vector_db_persisted_path (str): Path to the local folder for FAISS storage.
        documents (List[Document]): List of Document objects to add to the vector store.
        embeddings: Embedding model to convert documents into vectors.
    Returns:
        FAISS: A FAISS vector store object.
    """
    try:
         # Try to load an existing FAISS vector store from the specified path
        faiss_vectorstore_wrapper = create_faiss_vectorstore(vector_db_persisted_path, embeddings)
        filtered_docs=  faiss_only_add_documents_not_in_collection(faiss_vectorstore_wrapper, documents)
        
        # add only new documents not already in the collection
        if filtered_docs:
            print(f"➕ Adding {len(filtered_docs)} new documents to FAISS vector store.")
            faiss_vectorstore_wrapper.faiss_vectorstore.add_documents(filtered_docs)    

        faiss_vectorstore_wrapper.faiss_vectorstore.save_local(vector_db_persisted_path)
    except Exception as e:
        print(f"❌ Error creating FAISS vector store: {e}")
         # If loading fails (e.g., store doesn't exist), create a new one from documents
        faiss_vectorstore = FAISS.from_documents(
            documents,
            embeddings
        )
        # Save the newly created vector store to disk for future use
        faiss_vectorstore.save_local(vector_db_persisted_path)
        faiss_vectorstore_wrapper = FAISSVectorStore(faiss_vectorstore, vector_db_persisted_path)
        print(f"✅ FAISS vector store created and saved to {vector_db_persisted_path}")
        return faiss_vectorstore_wrapper.faiss_vectorstore

    return faiss_vectorstore_wrapper
    
    
def faiss_only_add_documents_not_in_collection(
    faiss_vectorstore_ptr: FAISSVectorStore,
    documents: List[Document],
) -> List[Document]:
    """
    Add documents to FAISS vector store only if they are not already present.
    Simple Explanation:
    - This function checks which documents are not already in the FAISS vector store
      and adds only those new documents.
    Example:
        Input: faiss_vectorstore_ptr, documents
        Output: List of newly added Document objects
    Args:
        faiss_vectorstore_ptr (FAISSVectorStore): Pointer to the FAISS vector store         
        documents (List[Document]): List of Document objects to check and add.
    Returns:
        List[Document]: List of Document objects that were newly added.
    """ 

    existing_docs = list(faiss_vectorstore_ptr.faiss_vectorstore.docstore._dict.values())
    existing_doc_ids = {doc.metadata.get("source") for doc in existing_docs}
    new_documents = [doc for doc in documents if doc.metadata.get("source") not in existing_doc_ids]
    if new_documents:
        faiss_vectorstore_ptr.faiss_vectorstore.add_documents(new_documents)
        print(f"➕ Added {len(new_documents)} new documents to FAISS vector store.")
    else:
        print("ℹ️ No new documents to add to FAISS vector store.")

    return new_documents

