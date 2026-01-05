import numpy as np
import os
import pickle
import torch
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from typing import List, Optional

from vectordatabases.vector_db_interface import VectorDBInterface

try:
    import chromadb
    from chromadb.config import Settings

    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


class ChromaVectorDB(VectorDBInterface):
    """ChromaDB implementation of vector database with GPU optimizations"""

    def __init__(self, embedding_dim: int =0, persist_path: Optional[str] = None,
                 collection_name: str = None, use_gpu: bool = True, **kwargs):
        if not CHROMA_AVAILABLE:
            raise ImportError("ChromaDB not available. Install with: uv add chromadb")

        super().__init__(embedding_dim, persist_path, **kwargs)

        self.collection_name = collection_name
        self.use_gpu = use_gpu

        self.check_gpu_availability()

        self.init_chroma_db_client(persist_path)

        self._setup_collection()

        # Load existing data if available
        if not self.load():
            optimization_type = "GPU-optimized" if self.gpu_available else "CPU-optimized"
            print(f"✅ Created new {optimization_type} ChromaDB collection: {collection_name}")
        collection_names= self.get_list_of_collections()
        print(f"📂 Available ChromaDB collections: {collection_names}" )

    def get_list_of_collections(self) -> List[str]:
        """Get list of collections in ChromaDB"""
        if not self.client:
            raise RuntimeError("ChromaDB client not initialized")
        return self.client.list_collections()

    def _setup_collection(self):
        collection_name = self.collection_name or "default_collection"

        # Create or get collection with optimized metadata
        collection_metadata = {
            "hnsw:space": "cosine",
            "hnsw:construction_ef": 200 if self.gpu_available else 100,
            "hnsw:M": 48 if self.gpu_available else 16,
            "hnsw:search_ef": 100 if self.gpu_available else 50,
            "gpu_optimized": self.gpu_available
        }

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata=collection_metadata
        )
    
    def get_max_document_length(self) -> int:
        if self.collection.count() == 0:
            return 0
        try:
            results = self.collection.get(include=['documents'])
            max_length = max(len(doc) for doc in results['documents'])
            return max_length
        except Exception as e:
            print(f"⚠️  Could not determine max document length: {e}")
            return 0

    def check_gpu_availability(self):
        # Check GPU availability
        self.gpu_available = torch.cuda.is_available() and self.use_gpu
        if self.gpu_available:
            print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"⚡ GPU acceleration enabled for preprocessing")
        else:
            if self.use_gpu:
                print(f"⚠️  GPU requested but not available, using CPU")
            else:
                print(f"💻 Using CPU (GPU disabled)")

    def init_chroma_db_client(self, persist_path: Optional[str] = None):
        # Initialize ChromaDB client with optimized settings
        if persist_path:
            # Use persistent storage
            chroma_path = f"{persist_path}_chroma"

            # GPU-optimized settings for ChromaDB
            settings = Settings(
                allow_reset=True,
                anonymized_telemetry=False,
                # Optimize for GPU environments
                chroma_server_thread_pool_size=8 if self.gpu_available else 4,
                chroma_server_grpc_port=None,
            )

            self.client = chromadb.PersistentClient(path=chroma_path, settings=settings)
            print(f"💾 Using persistent ChromaDB at {chroma_path}")
        else:
            # Use in-memory storage with optimized settings
            settings = Settings(
                allow_reset=True,
                anonymized_telemetry=False,
                chroma_server_thread_pool_size=8 if self.gpu_available else 4,
            )

            self.client = chromadb.Client(settings=settings)
            print(f"💻 Using in-memory ChromaDB")

    def add_documents(self, docs: List[Document], embeddings: List[np.ndarray]) -> None:
        if len(docs) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        # GPU-accelerated preprocessing if available
        processed_embeddings = embeddings
        if self.gpu_available and len(embeddings) > 100:
            try:
                print(f"🚀 Using GPU for embedding preprocessing...")

                # Convert to GPU tensors for batch processing
                gpu_embeddings = torch.tensor(np.array(embeddings), device='cuda')

                # Normalize embeddings on GPU for better similarity search
                gpu_embeddings = torch.nn.functional.normalize(gpu_embeddings, p=2, dim=1)

                # Convert back to CPU numpy arrays
                processed_embeddings = gpu_embeddings.cpu().numpy()

                print(f"✅ Preprocessed {len(embeddings)} embeddings on GPU")

            except Exception as e:
                print(f"⚠️  GPU preprocessing failed: {e}")
                processed_embeddings = embeddings

        start_idx = len(self.documents)

        # Batch processing for better performance
        batch_size = 1000 if self.gpu_available else 500
        total_docs = len(docs)

        print(f"📄 Adding {total_docs} documents to VectorDB in batches of {batch_size}...")

        for batch_start in range(0, total_docs, batch_size):
            batch_end = min(batch_start + batch_size, total_docs)

            batch_docs = docs[batch_start:batch_end]
            batch_embeddings = processed_embeddings[batch_start:batch_end]
            batch_ids = [f"doc_{start_idx + batch_start + i}" for i in range(len(batch_docs))]

            try:
                self.collection.add(
                    embeddings=[emb.tolist() for emb in batch_embeddings],
                    documents=[doc.page_content for doc in batch_docs],
                    metadatas=[doc.metadata for doc in batch_docs],
                    ids=batch_ids
                )
            except Exception as e:
                print(f"   ❌ Failed to add batch {batch_start + 1}-{batch_end}: {e}")
                raise

        self.documents.extend(docs)
        hardware_type = "GPU-accelerated" if self.gpu_available else "CPU"
        print(f"📄 Added {len(docs)} documents to ChromaDB vector database ({hardware_type})")
        print(f"📊 Total documents: {len(self.documents)}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Document]:
        if len(self.documents) == 0:
            print("⚠️  No documents in vector database to search")
            return []

        if query_embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Query embedding dimension {query_embedding.shape[0]} does not match index dimension {self.embedding_dim}")

        # GPU-accelerated query preprocessing
        processed_query = query_embedding
        if self.gpu_available:
            try:
                # Normalize query embedding on GPU for consistency
                gpu_query = torch.tensor(query_embedding, device='cuda')
                gpu_query = torch.nn.functional.normalize(gpu_query.unsqueeze(0), p=2, dim=1)
                processed_query = gpu_query.squeeze(0).cpu().numpy()

            except Exception as e:
                print(f"⚠️  GPU query preprocessing failed: {e}")
                processed_query = query_embedding

        try:
            results = self.collection.query(
                query_embeddings=[processed_query.tolist()],
                n_results=min(top_k, len(self.documents))
            )

            documents = []
            for i, (doc_text, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
            )):
                similarity_score = 1.0 / (1.0 + distance)
                doc = Document(
                    page_content=doc_text,
                    metadata={
                        **metadata,
                        'similarity_score': similarity_score,
                        'distance': distance,
                        'search_rank': i + 1
                    }
                )
                documents.append(doc)

            if documents:
                search_type = "GPU-accelerated" if self.gpu_available else "CPU"
                print(f"🔍 Found {len(documents)} similar documents ({search_type} search)")

            return documents

        except Exception as e:
            print(f"❌ ChromaDB search failed: {e}")
            return []

    def save(self) -> None:
        """Save ChromaDB collection metadata and configuration"""
        # ChromaDB PersistentClient automatically saves the collection data
        # We just need to save our metadata and configuration
        
        if not self.persist_path:
            print("ℹ️  No persist_path set. ChromaDB data is only in memory or auto-persisted.")
            return
    
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.persist_path) if os.path.dirname(self.persist_path) else '.', exist_ok=True)
        
        docs_file = f"{self.persist_path}.chroma.docs.pkl"
        config_file = f"{self.persist_path}.chroma.config.pkl"
    
        # Save document metadata
        with open(docs_file, 'wb') as f:
            pickle.dump(self.documents, f)
    
        # Save comprehensive configuration including collection info
        config_info = {
            'gpu_available': self.gpu_available,
            'use_gpu': self.use_gpu,
            'embedding_dim': self.embedding_dim,
            'collection_name': self.collection_name,
            'total_documents': len(self.documents),
            'persist_path': self.persist_path,
            'collection_metadata': self.collection.metadata if hasattr(self.collection, 'metadata') else {}
        }
    
        with open(config_file, 'wb') as f:
            pickle.dump(config_info, f)
    
        # Get collection count for verification
        try:
            count = self.collection.count()
            print(f"💾 ChromaDB saved:")
            print(f"   📄 {len(self.documents)} document metadata → {docs_file}")
            print(f"   ⚙️  Configuration → {config_file}")
            print(f"   📊 Collection '{self.collection_name}' has {count} vectors")
            if self.persist_path:
                chroma_path = f"{self.persist_path}_chroma"
                print(f"   💾 Collection data → {chroma_path}")
        except Exception as e:
            print(f"⚠️  Could not verify collection count: {e}")
    
    def load(self) -> bool:
        """Load documents and configuration from persisted files"""
        if not self.persist_path:
            # No persist path, check if in-memory collection has data
            try:
                count = self.collection.count()
                if count > 0:
                    print(f"📂 In-memory collection has {count} documents")
                    return self._reconstruct_documents_from_collection()
                return False
            except:
                return False
    
        docs_file = f"{self.persist_path}.chroma.docs.pkl"
        config_file = f"{self.persist_path}.chroma.config.pkl"
        chroma_path = f"{self.persist_path}_chroma"
    
        # Check if any persisted data exists
        has_config = os.path.exists(config_file)
        has_docs = os.path.exists(docs_file)
        has_chroma = os.path.exists(chroma_path)
    
        if not has_config and not has_chroma:
            print(f"ℹ️  No existing ChromaDB data found at {self.persist_path}")
            return False
    
        try:
            # Load configuration first
            saved_collection_name = self.collection_name
            if has_config:
                with open(config_file, 'rb') as f:
                    config_info = pickle.load(f)
                    
                saved_gpu = config_info.get('gpu_available', False)
                self.embedding_dim = config_info.get('embedding_dim', self.embedding_dim)
                saved_collection_name = config_info.get('collection_name', self.collection_name)
                saved_metadata = config_info.get('collection_metadata', {})
                
                # Check for hardware changes
                if saved_gpu != self.gpu_available:
                    gpu_status = "GPU → CPU" if saved_gpu and not self.gpu_available else "CPU → GPU"
                    print(f"ℹ️  Hardware change detected: {gpu_status}")
                
                print(f"📋 Loaded configuration from {config_file}")
    
            # Reconnect to persisted ChromaDB if path exists
            if has_chroma:
                print(f"🔄 Connecting to persisted ChromaDB at {chroma_path}...")
                
                # Create new client connected to persisted data
                settings = Settings(
                    allow_reset=True,
                    anonymized_telemetry=False,
                    chroma_server_thread_pool_size=8 if self.gpu_available else 4,
                    chroma_server_grpc_port=None,
                )
                
                self.client = chromadb.PersistentClient(path=chroma_path, settings=settings)
                
                # Get the existing collection
                try:
                    self.collection = self.client.get_collection(name=saved_collection_name)
                    count = self.collection.count()
                    print(f"📂 Connected to collection '{self.collection.name}' with {count} vectors")
                    
                    # Verify collection metadata
                    if hasattr(self.collection, 'metadata') and self.collection.metadata:
                        print(f"   HNSW space: {self.collection.metadata.get('hnsw:space', 'N/A')}")
                        print(f"   HNSW M: {self.collection.metadata.get('hnsw:M', 'N/A')}")
                        print(f"   GPU optimized: {self.collection.metadata.get('gpu_optimized', 'N/A')}")
                    
                except Exception as e:
                    print(f"❌ Collection '{saved_collection_name}' not found in persisted database: {e}")
                    # Try to recreate collection
                    print(f"🔄 Recreating collection...")
                    self._setup_collection()
    
            # Load document metadata
            if has_docs:
                with open(docs_file, 'rb') as f:
                    self.documents = pickle.load(f)
                print(f"💾 Loaded {len(self.documents)} document metadata from {docs_file}")
            else:
                # Reconstruct from collection if metadata file missing
                print(f"ℹ️  No metadata file, reconstructing from collection...")
                if not self._reconstruct_documents_from_collection():
                    print(f"⚠️  Could not reconstruct documents from collection")
                    self.documents = []
    
            # Verify integrity
            try:
                count = self.collection.count()
                if count != len(self.documents):
                    print(f"⚠️  Mismatch: Collection has {count} vectors, metadata has {len(self.documents)} documents")
                    if count > len(self.documents):
                        print(f"🔄 Reconstructing full document list from collection...")
                        self._reconstruct_documents_from_collection()
            except Exception as e:
                print(f"⚠️  Could not verify collection count: {e}")
    
            hardware_type = "GPU-enabled" if self.gpu_available else "CPU"
            print(f"✅ ChromaDB loaded successfully ({hardware_type})")
            print(f"   📊 {len(self.documents)} documents ready")
            return True
    
        except Exception as e:
            print(f"❌ Failed to load ChromaDB: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
    def _reconstruct_documents_from_collection(self) -> bool:
        """Attempt to reconstruct documents list from ChromaDB collection"""
        try:
            # Get all documents from collection
            results = self.collection.get(include=['documents', 'metadatas'])

            self.documents = []
            for doc_text, metadata in zip(results['documents'], results['metadatas']):
                doc = Document(
                    page_content=doc_text,
                    metadata=metadata
                )
                self.documents.append(doc)

            print(f"🔄 Reconstructed {len(self.documents)} documents from collection")
            return True

        except Exception as e:
            print(f"⚠️  Could not reconstruct documents: {e}")
            return False

    def get_total_documents(self) -> int:
        return len(self.documents)

    def get_collection_info(self) -> dict:
        """Get detailed information about the ChromaDB collection"""
        try:
            count = self.collection.count()
            metadata = self.collection.metadata or {}

            return {
                'documents_count': count,
                'embedding_dimension': self.embedding_dim,
                'collection_name': self.collection_name,
                'gpu_optimized': self.gpu_available,
                'hnsw_space': metadata.get('hnsw:space', 'cosine'),
                'hnsw_M': metadata.get('hnsw:M', 16),
                'hnsw_construction_ef': metadata.get('hnsw:construction_ef', 100),
                'persistent_storage': self.persist_path is not None
            }
        except Exception as e:
            return {'error': str(e)}

    def optimize_collection(self):
        """Optimize collection settings based on current data size and GPU availability"""
        try:
            count = self.collection.count()

            # Determine optimal settings based on collection size and GPU
            if self.gpu_available and count > 10000:
                # High-performance settings for large GPU collections
                ef_construction = 400
                M = 64
            elif self.gpu_available:
                # Balanced settings for smaller GPU collections  
                ef_construction = 200
                M = 48
            elif count > 10000:
                # CPU settings for large collections
                ef_construction = 200
                M = 32
            else:
                # CPU settings for smaller collections
                ef_construction = 100
                M = 16

            # Update collection metadata (Note: ChromaDB may not support runtime updates)
            print(f"⚙️  Optimal settings for {count} documents:")
            print(f"   M: {M}, ef_construction: {ef_construction}")
            print(f"   GPU optimized: {self.gpu_available}")

        except Exception as e:
            print(f"⚠️  Could not optimize collection: {e}")

    def get_embedding_dim(self) -> int:
        return self.embedding_dim
    
    def delete_collection(self, **kwargs) -> bool:
        """ 
        Delete Table or collection supported by the 
        database and all files associated to
        it from disk.
        """
        collection_name: str = kwargs.get('collection_name', self.collection_name)
        try:
            self.client.delete_collection(name=collection_name)
            print(f"🗑️  Deleted ChromaDB collection: {collection_name}")
            self.documents = []
            return True
        except Exception as e:
            print(f"❌ Failed to delete collection: {e}")
            return False
        
        
    def as_langchain_retriever(self, embedding_function, top_k: int = 5):
        """Convert to LangChain retriever

        Args:
            embedding_function: A callable that takes a string and returns an embedding vector
            top_k: Number of documents to retrieve

        Returns:
            ChromaChainRetriever: A LangChain-compatible retriever
        """
        return ChromaChainRetriever(
            vector_db=self,
            embedding_function=embedding_function,
            top_k=top_k
        )


class ChromaChainRetriever(BaseRetriever):
    """LangChain-compatible retriever for ChromaVectorDB
    
    This class wraps ChromaVectorDB to make it compatible with LangChain's retriever interface.
    Since BaseRetriever is a Pydantic model, it performs strict type validation on all attributes.
    
    The nested Config class with 'arbitrary_types_allowed = True' is required because:
    - vector_db: ChromaVectorDB is a custom class instance (not a standard Pydantic type)
    - embedding_function: callable is a function/callable object (not a standard Pydantic type)
    
    Without this configuration, Pydantic would raise validation errors for these custom types.
    This tells Pydantic to accept these arbitrary Python objects without transformation.
    """
    vector_db: ChromaVectorDB
    embedding_function: callable
    top_k: int = 5

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """Retrieve documents relevant to the query
           1. When LangChain sees retriever in the context
           2. Calls retriever.get_relevant_documents(query)
           3. Which internally calls your _get_relevant_documents()

        Args:
            query: The query string to search for
            run_manager: Optional callback manager

        Returns:
            List of relevant documents with similarity scores
        """
        chunk_size = self.vector_db.get_max_document_length()
        print(f"Max document length in vector DB: {chunk_size} characters")
        query_text_input = query
        if len(query_text_input) > chunk_size:
            print(f"⚠️  Query text length ({len(query)}) exceeds max document length in vector DB ({chunk_size}). Truncating query.")
            query_text_input = query_text_input[:chunk_size]

        # Generate embedding for the query
        query_embedding = self.embedding_function(query_text_input)

        # Ensure it's a numpy array
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)

        # Search the vector database
        results = self.vector_db.search(query_embedding, top_k=self.top_k)
        
        print(f"\n🔍 Retrieved {len(results)} documents for query '{query_text_input}'")
        return results
    
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
            print(self.dict_to_str(doc.metadata))
            context += doc.page_content + "\n\n"
        return context
    
    def dict_to_str(self, d: dict) -> str:
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


