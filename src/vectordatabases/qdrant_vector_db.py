import numpy as np
import os
import pickle
import torch
import uuid
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import List, Optional
from vectordatabases.vector_db_interface import VectorDBInterface

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, OptimizersConfigDiff, HnswConfigDiff

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


class QdrantVectorDB(VectorDBInterface):
    """Qdrant implementation of vector database with GPU optimizations"""

    def __init__(self, embedding_dim: int = 0, persist_path: Optional[str] = None,
                 collection_name: str = None, host: str = None, port: int = None,
                 use_gpu: bool = True, prefer_server: bool = False, **kwargs):
        if not QDRANT_AVAILABLE:
            raise ImportError("Qdrant not available. Install with: uv add qdrant-client")

        super().__init__(embedding_dim, persist_path, **kwargs)
        self.use_gpu = use_gpu
        self.host = host
        self.port = port
        self.prefer_server = prefer_server
        self.client = None
        self.gpu_available = None
        self.server_mode = False

        # Set collection name with fallback
        if collection_name:
            self.collection_name = collection_name
        elif persist_path:
            # Generate collection name from persist_path
            self.collection_name = os.path.basename(persist_path).replace('.', '_')
        else:
            self.collection_name = "default_collection"

        # Scenario 1: Load from existing persist_path only
        if persist_path and not embedding_dim and not host and not port:
            print(f"🔄 Loading from persist_path: {persist_path}")
            if not self.load():
                print(f"❌ Failed to load from {persist_path}. Need embedding_dim to create new collection.")
                return
        # Scenario 2: Create new or connect to existing
        elif embedding_dim:
            self.embedding_dim = embedding_dim
            self.prepare(host, port, prefer_server)
        else:
            raise ValueError("Must provide either persist_path (to load) or embedding_dim (to create new)")

        # print list of lloaded Collections
        collections = self.get_list_of_collections()
        print(f"📂 Available Qdrant collections: {collections}")

    def get_list_of_collections(self) -> List[str]:
        """Get list of collections in Qdrant"""
        if not self.client:
            raise RuntimeError("Qdrant client not initialized")

        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            return collection_names
        except Exception as e:
            print(f"❌ Failed to get collections: {e}")
            return []

    def prepare(self, host, port, prefer_server):
        # Check GPU availability
        self.gpu_available = torch.cuda.is_available() and self.use_gpu
        if self.gpu_available:
            print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            if self.use_gpu:
                print(f"⚠️  GPU requested but not available, using CPU")
            else:
                print(f"💻 Using CPU (GPU disabled)")

        self.init_client(host, port, prefer_server)

        # Try to load existing data first
        loaded = False
        if self.persist_path:
            loaded = self.load()

        # Only setup collection if we didn't load successfully
        if not loaded:
            self._setup_collection()
            print(f"✅ Created new Qdrant collection: {self.collection_name}")

    def init_client(self, host: str = None, port: int = None,
                    prefer_server: bool = False) -> None:
        # Initialize client
        self.server_mode = False
        self.client = None
        self.host = host
        self.port = port
        self.prefer_server = prefer_server
        # Only try server connection if explicitly requested
        if prefer_server and host and port:
            try:
                print(f"🔍 Attempting to connect to Qdrant server at {host}:{port}...")
                test_client = QdrantClient(host=host, port=port, timeout=5)
                # Test the connection with a simple operation
                test_client.get_collections()
                self.client = test_client
                print(f"✅ Connected to Qdrant server at {host}:{port}")
                self.server_mode = True
            except Exception as e:
                print(f"ℹ️  Qdrant server not available ({e})")
                print(f"🔄 Using in-memory Qdrant database instead...")
        else:
            if not prefer_server:
                print(f"💻 Using in-memory Qdrant database (server not requested)")
            else:
                print(f"💻 Using in-memory Qdrant database (no server configured)")

        # Use in-memory if server connection failed or wasn't attempted
        if self.client is None:
            try:
                self.client = QdrantClient(":memory:")
                if not prefer_server:
                    print(f"✅ In-memory Qdrant database initialized")
                self.server_mode = False
            except Exception as e:
                print(f"❌ Failed to create in-memory Qdrant client: {e}")
                raise RuntimeError(f"Could not initialize Qdrant client: {e}")

    def _get_optimized_config(self):
        """Get optimized configuration based on GPU availability and data size"""
        if self.gpu_available:
            # GPU-optimized configuration
            return {
                "hnsw_config": HnswConfigDiff(
                    m=48,  # Higher M for better recall with GPU memory
                    ef_construct=200,  # Higher ef_construct for better indexing
                    full_scan_threshold=10000,  # Use full scan for smaller datasets
                    max_indexing_threads=0,  # Auto-detect threads
                ),
                "optimizer_config": OptimizersConfigDiff(
                    deleted_threshold=0.2,
                    vacuum_min_vector_number=1000,
                    default_segment_number=2,  # Fewer segments for GPU
                    max_segment_size=None,
                    memmap_threshold=None,
                    indexing_threshold=20000,
                    flush_interval_sec=5,
                    max_optimization_threads=None,
                )
            }
        else:
            # CPU-optimized configuration
            return {
                "hnsw_config": HnswConfigDiff(
                    m=16,  # Lower M for CPU memory efficiency
                    ef_construct=100,  # Lower ef_construct for CPU
                    full_scan_threshold=5000,
                    max_indexing_threads=0,
                ),
                "optimizer_config": OptimizersConfigDiff(
                    deleted_threshold=0.2,
                    vacuum_min_vector_number=1000,
                    default_segment_number=4,  # More segments for CPU
                    max_segment_size=None,
                    memmap_threshold=None,
                    indexing_threshold=10000,
                    flush_interval_sec=10,
                    max_optimization_threads=None,
                )
            }

    def _setup_collection(self):
        """Create or get the collection with optimized configuration"""
        if not self.client:
            raise RuntimeError("Qdrant client not initialized")

        try:
            # Try to get existing collection
            collection_info = self.client.get_collection(self.collection_name)
            print(f"📂 Found existing Qdrant collection: {self.collection_name}")
            print(f"📊 Points in collection: {collection_info.points_count}")

            # Update collection with optimized config if it's a server
            if self.server_mode:
                try:
                    config = self._get_optimized_config()
                    self.client.update_collection(
                        collection_name=self.collection_name,
                        optimizer_config=config["optimizer_config"],
                        hnsw_config=config["hnsw_config"]
                    )
                    optimization_type = "GPU-optimized" if self.gpu_available else "CPU-optimized"
                    print(f"⚙️  Updated collection with {optimization_type} configuration")
                except Exception as e:
                    print(f"ℹ️  Could not update collection config: {e}")

        except Exception as excep:
            print(f"❌ Failed to get collection : {excep}")
            # Create new collection with optimized configuration
            try:
                config = self._get_optimized_config()

                vector_config = VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                    hnsw_config=config["hnsw_config"]
                )

                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=vector_config,
                    optimizers_config=config["optimizer_config"]
                )

                optimization_type = "GPU-optimized" if self.gpu_available else "CPU-optimized"
                mode = "server" if self.server_mode else "in-memory"
                print(f"✅ Created new {optimization_type} Qdrant collection ({mode}): {self.collection_name}")

            except Exception as create_e:
                print(f"⚠️  Failed to create collection with optimized config: {create_e}")
                # Try creating with minimal configuration as fallback
                try:
                    vector_config = VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )

                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=vector_config
                    )
                    print(f"✅ Created Qdrant collection with basic configuration: {self.collection_name}")

                except Exception as basic_e:
                    print(f"❌ Failed to create collection with basic config: {basic_e}")
                    raise RuntimeError(f"Could not create Qdrant collection: {basic_e}")

    def get_max_document_length(self) -> int:
        if self.documents:
            return max(len(doc.page_content) for doc in self.documents)
        return 0

    def add_documents(self, docs: List[Document], embeddings: List[np.ndarray]) -> None:
        if len(docs) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        # Convert embeddings to GPU tensors if available for faster processing
        if self.gpu_available and len(embeddings) > 100:
            print(f"🚀 Using GPU for embedding preprocessing...")
            try:
                # Convert to GPU tensors for faster batch processing
                gpu_embeddings = torch.tensor(np.array(embeddings), device='cuda')
                # Normalize on GPU if needed
                gpu_embeddings = torch.nn.functional.normalize(gpu_embeddings, p=2, dim=1)
                # Convert back to CPU numpy arrays for Qdrant
                embeddings = gpu_embeddings.cpu().numpy()
                print(f"✅ Processed {len(embeddings)} embeddings on GPU")
            except Exception as e:
                print(f"⚠️  GPU processing failed, using CPU: {e}")

        points = []
        start_idx = len(self.documents)

        # Batch processing for better performance
        batch_size = 1000 if self.gpu_available else 500
        total_points = len(docs)

        print(f"📄 Adding {total_points} documents to vector database in batches of {batch_size}...")

        for batch_start in range(0, total_points, batch_size):
            batch_end = min(batch_start + batch_size, total_points)
            batch_points = []

            for i in range(batch_start, batch_end):
                doc = docs[i]
                embedding = embeddings[i]
                point_id = str(uuid.uuid4())

                batch_points.append(PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "doc_index": start_idx + i
                    }
                ))

            # Upsert batch
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch_points
                )
                print(f"   ✅ Processed batch {batch_start + 1}-{batch_end} of {total_points}")
            except Exception as e:
                print(f"   ❌ Failed to upsert batch {batch_start + 1}-{batch_end}: {e}")
                raise

        self.documents.extend(docs)
        print(f"📄 Added {len(docs)} documents to Qdrant vector database")
        print(f"📊 Total documents: {len(self.documents)}")

        # Optimize collection after adding documents if using server
        if self.server_mode and len(docs) > 1000:
            try:
                print(f"⚙️  Optimizing collection for better search performance...")
                # This will trigger index optimization
                self.client.update_collection(
                    collection_name=self.collection_name,
                    optimizer_config=OptimizersConfigDiff(
                        indexing_threshold=max(1000, len(self.documents) // 10)
                    )
                )
                print(f"✅ Collection optimization triggered")
            except Exception as e:
                print(f"ℹ️  Could not trigger optimization: {e}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5,
               search_params: Optional[dict] = None) -> List[Document]:
        if len(self.documents) == 0:
            print("⚠️  No documents in vector database to search")
            return []

        if query_embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Query embedding dimension {query_embedding.shape[0]} does not match index dimension {self.embedding_dim}")

        # GPU-accelerated query preprocessing if available
        if self.gpu_available:
            try:
                # Normalize query embedding on GPU
                gpu_query = torch.tensor(query_embedding, device='cuda')
                gpu_query = torch.nn.functional.normalize(gpu_query.unsqueeze(0), p=2, dim=1)
                query_embedding = gpu_query.squeeze(0).cpu().numpy()
            except Exception as e:
                print(f"⚠️  GPU query preprocessing failed: {e}")

        # Set search parameters based on GPU availability
        if search_params is None:
            if self.gpu_available:
                # More aggressive search with GPU
                search_params = {
                    "hnsw_ef": min(200, max(top_k * 4, 64)),
                    "exact": False
                }
            else:
                # Conservative search with CPU
                search_params = {
                    "hnsw_ef": min(100, max(top_k * 2, 32)),
                    "exact": False
                }

        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding.tolist(),
                limit=min(top_k, len(self.documents)),
                search_params=search_params
            )

            documents = []
            for i, result in enumerate(results.points):
                doc = Document(
                    page_content=result.payload["content"],
                    metadata={
                        **result.payload["metadata"],
                        'similarity_score': result.score,
                        'search_rank': i + 1
                    }
                )
                documents.append(doc)

            if documents:
                search_type = "GPU-accelerated" if self.gpu_available else "CPU"
                print(f"🔍 Found {len(documents)} similar documents ({search_type} search)")

            return documents

        except Exception as e:
            print(f"❌ Qdrant search failed: {e}")
            return []

    def save(self) -> None:
        if not self.persist_path:
            mode = "server" if self.server_mode else "in-memory"
            print(f"ℹ️  No persist_path set, Qdrant data persisted in {mode} collection")
            return

        # Save document metadata for consistency
        os.makedirs(os.path.dirname(self.persist_path) if os.path.dirname(self.persist_path) else '.', exist_ok=True)
        docs_file = f"{self.persist_path}.qdrant.docs.pkl"

        with open(docs_file, 'wb') as f:
            pickle.dump(self.documents, f)

        print(f"💾 Qdrant metadata saved to {docs_file}")

        # Save GPU configuration info
        config_file = f"{self.persist_path}.qdrant.config.pkl"
        config_info = {
            'host': self.host,
            'port': self.port,
            'embedding_dim': self.embedding_dim,
            'collection_name': self.collection_name,
            'prefer_server': self.prefer_server,
            'use_gpu': self.use_gpu,
            'gpu_available': self.gpu_available
        }

        with open(config_file, 'wb') as f:
            pickle.dump(config_info, f)

        # Save collection snapshot (vectors and payloads) for in-memory mode
        if not self.server_mode:
            try:
                print(f"💾 Saving collection vectors and payloads...")
                # Get all points from collection
                collection_info = self.client.get_collection(self.collection_name)
                total_points = collection_info.points_count

                if total_points > 0:
                    # Scroll through all points
                    all_points = []
                    offset = None
                    batch_size = 1000

                    while True:
                        result = self.client.scroll(
                            collection_name=self.collection_name,
                            limit=batch_size,
                            offset=offset,
                            with_payload=True,
                            with_vectors=True
                        )

                        points, next_offset = result

                        if not points:
                            break

                        all_points.extend(points)

                        if next_offset is None:
                            break

                        offset = next_offset

                    # Save all points data
                    points_file = f"{self.persist_path}.qdrant.points.pkl"
                    points_data = [
                        {
                            'id': point.id,
                            'vector': point.vector,
                            'payload': point.payload
                        }
                        for point in all_points
                    ]

                    with open(points_file, 'wb') as f:
                        pickle.dump(points_data, f)

                    print(f"💾 Saved {len(points_data)} collection points to {points_file}")
            except Exception as e:
                print(f"⚠️  Could not save collection points: {e}")
        else:
            print(f"ℹ️  Server mode: collection data persisted on Qdrant server")

    def load(self) -> bool:
        """Load documents and configuration from persisted files"""
        if not self.persist_path:
            return False

        docs_file = f"{self.persist_path}.qdrant.docs.pkl"
        config_file = f"{self.persist_path}.qdrant.config.pkl"
        points_file = f"{self.persist_path}.qdrant.points.pkl"

        # Check if files exist
        if not os.path.exists(docs_file) or not os.path.exists(config_file):
            print(f"ℹ️  No existing Qdrant data found at {self.persist_path}")
            return False

        try:
            # Load configuration
            with open(config_file, 'rb') as f:
                config_info = pickle.load(f)

            # Restore configuration
            self.host = config_info.get('host', self.host)
            self.port = config_info.get('port', self.port)
            self.embedding_dim = config_info.get('embedding_dim', self.embedding_dim)
            self.collection_name = config_info.get('collection_name', self.collection_name)
            self.prefer_server = config_info.get('prefer_server', self.prefer_server)
            self.use_gpu = config_info.get('use_gpu', self.use_gpu)

            # Check GPU availability (may have changed since save)
            self.gpu_available = torch.cuda.is_available() and self.use_gpu

            # Initialize client if not already done
            if self.client is None:
                print(f"🔄 Initializing Qdrant client from saved configuration...")
                self.init_client(self.host, self.port, self.prefer_server)

            # Load documents
            with open(docs_file, 'rb') as f:
                self.documents = pickle.load(f)

            print(f"💾 Loaded {len(self.documents)} documents from {docs_file}")

            # Try to connect to existing collection
            collection_exists = False
            try:
                collection_info = self.client.get_collection(self.collection_name)
                print(f"📂 Connected to existing collection: {self.collection_name}")
                print(f"📊 Points in collection: {collection_info.points_count}")
                collection_exists = True

                # Verify point count matches
                if collection_info.points_count != len(self.documents):
                    print(
                        f"⚠️  Warning: Collection has {collection_info.points_count} points but loaded {len(self.documents)} documents")

            except Exception as e:
                print(f"ℹ️  Collection '{self.collection_name}' not found: {e}")

            # If collection doesn't exist and we have saved points, restore them
            if not collection_exists and os.path.exists(points_file):
                print(f"🔄 Restoring collection from saved points...")

                # Create collection
                self._setup_collection()

                # Load and restore points
                try:
                    with open(points_file, 'rb') as f:
                        points_data = pickle.load(f)

                    print(f"� Restoring {len(points_data)} points to collection...")

                    # Batch upsert for better performance
                    batch_size = 1000 if self.gpu_available else 500
                    for batch_start in range(0, len(points_data), batch_size):
                        batch_end = min(batch_start + batch_size, len(points_data))
                        batch_points = []

                        for point_data in points_data[batch_start:batch_end]:
                            batch_points.append(PointStruct(
                                id=point_data['id'],
                                vector=point_data['vector'],
                                payload=point_data['payload']
                            ))

                        self.client.upsert(
                            collection_name=self.collection_name,
                            points=batch_points
                        )
                        print(f"   ✅ Restored batch {batch_start + 1}-{batch_end} of {len(points_data)}")

                    print(f"✅ Successfully restored {len(points_data)} points to collection")

                except Exception as e:
                    print(f"❌ Failed to restore points: {e}")
                    return False

            elif not collection_exists:
                print(f"⚠️  No collection or saved points found. Collection needs to be recreated with add_documents()")
                # Create empty collection
                self._setup_collection()

            mode = "server" if self.server_mode else "in-memory"
            print(f"✅ Qdrant data loaded from {self.persist_path} ({mode} mode)")
            return True

        except Exception as e:
            print(f"❌ Failed to load Qdrant data: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_total_documents(self) -> int:
        return len(self.documents)

    def get_collection_info(self) -> dict:
        """Get detailed information about the collection"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                'points_count': collection_info.points_count,
                'segments_count': len(collection_info.segments) if hasattr(collection_info, 'segments') else 'N/A',
                'vector_size': collection_info.config.params.vectors.size,
                'distance': collection_info.config.params.vectors.distance,
                'gpu_optimized': self.gpu_available,
                'server_mode': self.server_mode
            }
        except Exception as e:
            return {'error': str(e)}

    def get_embedding_dim(self) -> int:
        return self.embedding_dim

    def delete_collection(self, **kwargs) -> bool:
        """ 
        Delete Table or collection supported by the 
        database and all files associated to
        it in the persist_path.
        """
        collection_name: str = kwargs.get('collection_name', self.collection_name)
        persist_path: str = kwargs.get('persist_path', self.persist_path)
        try:
            self.client.delete_collection(collection_name=collection_name)
            print(f"🗑️  Deleted Qdrant collection: {collection_name}")

            # Remove persisted files if they exist
            if self.persist_path:
                docs_file = f"{persist_path}.qdrant.docs.pkl"
                config_file = f"{persist_path}.qdrant.config.pkl"
                points_file = f"{persist_path}.qdrant.points.pkl"

                for file in [docs_file, config_file, points_file]:
                    if os.path.exists(file):
                        os.remove(file)
                        print(f"🗑️  Removed persisted file: {file}")

            # Clear in-memory documents
            self.documents = []

            return True
        except Exception as e:
            print(f"❌ Failed to delete collection: {e}")
            return False

    def as_langchain_retriever(self, embedding_function, top_k: int = 5):
        """Convert to LangChain retriever interface"""
        return QdrantLangChainRetriever(vector_db=self,
                                        embedding_function=embedding_function,
                                        top_k=top_k)


class QdrantLangChainRetriever(BaseRetriever):
    """LangChain Retriever wrapper for QdrantVectorDB"""

    vector_db: QdrantVectorDB
    embedding_function: callable
    top_k: int = 5

    def get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        # Generate embedding for the query
        query_embedding = self.embedding_function(query)

        # Search in the vector database
        results = self.vector_db.search(
            query_embedding=query_embedding,
            top_k=self.top_k
        )

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
        context = ""
        for i, doc in enumerate(results, 1):
            context += doc.page_content + "\n\n"
        return context
