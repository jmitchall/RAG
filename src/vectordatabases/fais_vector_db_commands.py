from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from typing import List
from langchain_core.embeddings import Embeddings

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
        print(f"❌ Error loading FAISS vector store: {e}")
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