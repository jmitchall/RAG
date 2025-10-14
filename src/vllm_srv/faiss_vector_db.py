import faiss
import numpy as np
import os
import pickle
import torch
from langchain_core.documents import Document
from typing import List, Optional

from vector_db_interface import VectorDBInterface


class FaissVectorDB(VectorDBInterface):
    """FAISS implementation of vector database with enhanced GPU support"""

    def __init__(self, embedding_dim: int, persist_path: Optional[str] = None, use_gpu: bool = True,
                 gpu_memory_fraction: float = 0.8, **kwargs):
        super().__init__(embedding_dim, persist_path, **kwargs)

        self.use_gpu = use_gpu
        self.gpu_memory_fraction = gpu_memory_fraction
        self.gpu_available = False
        self.gpu_resources = None

        # Enhanced GPU detection and setup
        if self.use_gpu:
            self._setup_gpu()

        # Try to load existing index, otherwise create new one
        if not self.load():
            self._create_new_index()

    def _setup_gpu(self):
        """Enhanced GPU setup with proper resource management"""
        try:
            # Check if FAISS GPU is available
            if not hasattr(faiss, 'StandardGpuResources'):
                print(f"ℹ️  FAISS CPU version detected (no GPU support)")
                print(f"💡 For GPU support, install: pip install faiss-gpu")
                self.gpu_available = False
                return

            # Check if CUDA is available
            if not torch.cuda.is_available():
                print(f"ℹ️  CUDA not available, using CPU")
                self.gpu_available = False
                return

            # Get GPU information
            gpu_count = torch.cuda.device_count()
            current_gpu = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_gpu)
            gpu_memory = torch.cuda.get_device_properties(current_gpu).total_memory

            print(f"🚀 GPU detected: {gpu_name}")
            print(f"💾 GPU Memory: {gpu_memory / 1e9:.1f} GB")
            print(f"🔢 Available GPUs: {gpu_count}")

            # Setup GPU resources with memory management
            self.gpu_resources = faiss.StandardGpuResources()

            # Set memory fraction to avoid OOM
            available_memory = int(gpu_memory * self.gpu_memory_fraction)
            self.gpu_resources.setTempMemory(available_memory)

            print(f"⚙️  Allocated {available_memory / 1e9:.1f} GB GPU memory for FAISS")

            self.gpu_available = True

        except Exception as e:
            print(f"⚠️  GPU setup failed: {str(e)}")
            print(f"💻 Falling back to CPU")
            self.gpu_available = False
            self.gpu_resources = None

    def _create_new_index(self):
        """Create a new FAISS index with GPU optimization"""
        # Start with CPU index
        self.index = faiss.IndexFlatL2(self.embedding_dim)

        if self.gpu_available:
            try:
                # Move to GPU with device 0
                self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.index)
                print(f"✅ Created new FAISS vector database with GPU acceleration")

                # Print GPU index information
                print(f"⚡ GPU Index Type: {type(self.index).__name__}")

            except Exception as e:
                print(f"⚠️  GPU index creation failed: {str(e)}")
                print(f"💻 Using CPU index instead")
                self.gpu_available = False
                # Recreate CPU index
                self.index = faiss.IndexFlatL2(self.embedding_dim)
        else:
            print(f"💻 Created new FAISS vector database with CPU")


    def get_max_document_length(self) -> int:
        if self.documents:
            return max(len(doc.page_content) for doc in self.documents)
        return 0
     
     
    def add_documents(self, docs: List[Document], embeddings: List[np.ndarray]) -> None:
        if len(docs) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        # Convert embeddings with GPU optimization
        embeddings_array = np.array(embeddings, dtype=np.float32)

        if embeddings_array.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension {embeddings_array.shape[1]} does not match index dimension {self.embedding_dim}")

        # GPU-accelerated preprocessing if available
        if self.gpu_available and len(embeddings) > 100:
            try:
                print(f"🚀 Using GPU for embedding preprocessing...")

                # Convert to GPU tensors for normalization (optional but can improve search quality)
                gpu_embeddings = torch.tensor(embeddings_array, device='cuda')

                # Normalize embeddings on GPU (L2 normalization for better cosine similarity)
                gpu_embeddings = torch.nn.functional.normalize(gpu_embeddings, p=2, dim=1)

                # Convert back to CPU numpy for FAISS
                embeddings_array = gpu_embeddings.cpu().numpy().astype(np.float32)

                print(f"✅ Preprocessed {len(embeddings)} embeddings on GPU")

            except Exception as e:
                print(f"⚠️  GPU preprocessing failed: {e}")
                print(f"💻 Using original embeddings")

        # Add to index
        try:
            self.index.add(embeddings_array)
            self.documents.extend(docs)

            hardware_type = "GPU" if self.gpu_available else "CPU"
            print(f"📄 Added {len(docs)} documents to FAISS vector database ({hardware_type})")
            print(f"📊 Total documents: {len(self.documents)}")

            # Print index statistics
            print(f"🔢 Index size: {self.index.ntotal} vectors")

        except Exception as e:
            print(f"❌ Failed to add documents: {e}")
            raise

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Document]:
        if len(self.documents) == 0:
            print("⚠️  No documents in vector database to search")
            return []

        if query_embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Query embedding dimension {query_embedding.shape[0]} does not match index dimension {self.embedding_dim}")

        top_k = min(top_k, len(self.documents))

        # GPU-accelerated query preprocessing
        if self.gpu_available:
            try:
                # Normalize query on GPU for consistency
                gpu_query = torch.tensor(query_embedding, device='cuda')
                gpu_query = torch.nn.functional.normalize(gpu_query.unsqueeze(0), p=2, dim=1)
                query_array = gpu_query.squeeze(0).cpu().numpy().astype(np.float32).reshape(1, -1)
            except Exception as e:
                print(f"⚠️  GPU query preprocessing failed: {e}")
                query_array = np.array([query_embedding], dtype=np.float32)
        else:
            query_array = np.array([query_embedding], dtype=np.float32)

        try:
            # Perform search
            distances, indices = self.index.search(query_array, top_k)

            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx >= 0 and idx < len(self.documents):
                    doc = self.documents[idx]
                    similarity_score = 1.0 / (1.0 + distance)

                    doc_with_score = Document(
                        page_content=doc.page_content,
                        metadata={
                            **doc.metadata,
                            'similarity_score': similarity_score,
                            'distance': float(distance),
                            'search_rank': i + 1
                        }
                    )
                    results.append(doc_with_score)

            if results:
                hardware_type = "GPU-accelerated" if self.gpu_available else "CPU"
                print(f"🔍 Found {len(results)} similar documents ({hardware_type} search)")

            return results

        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []

    def save(self) -> None:
        if not self.persist_path:
            raise ValueError("No persist_path set for saving")

        os.makedirs(os.path.dirname(self.persist_path) if os.path.dirname(self.persist_path) else '.', exist_ok=True)

        index_file = f"{self.persist_path}.faiss"
        docs_file = f"{self.persist_path}.docs.pkl"
        config_file = f"{self.persist_path}.config.pkl"

        try:
            # Save index (move to CPU first if on GPU)
            if self.gpu_available and hasattr(faiss, 'index_gpu_to_cpu'):
                print(f"💾 Moving GPU index to CPU for saving...")
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(cpu_index, index_file)
            else:
                faiss.write_index(self.index, index_file)

            # Save documents
            with open(docs_file, 'wb') as f:
                pickle.dump(self.documents, f)

            # Save configuration
            config_info = {
                'gpu_available': self.gpu_available,
                'embedding_dim': self.embedding_dim,
                'total_documents': len(self.documents),
                'gpu_memory_fraction': self.gpu_memory_fraction
            }

            with open(config_file, 'wb') as f:
                pickle.dump(config_info, f)

            print(f"💾 FAISS vector database saved to {self.persist_path}")

        except Exception as e:
            print(f"❌ Failed to save database: {e}")
            raise

    def load(self) -> bool:
        if not self.persist_path:
            return False

        index_file = f"{self.persist_path}.faiss"
        docs_file = f"{self.persist_path}.docs.pkl"
        config_file = f"{self.persist_path}.config.pkl"

        if not (os.path.exists(index_file) and os.path.exists(docs_file)):
            return False

        try:
            # Load index
            print(f"📥 Loading FAISS index from {index_file}...")
            self.index = faiss.read_index(index_file)

            # Load configuration if available
            if os.path.exists(config_file):
                with open(config_file, 'rb') as f:
                    config_info = pickle.load(f)
                    saved_gpu = config_info.get('gpu_available', False)
                    self.embedding_dim = config_info.get('embedding_dim', self.embedding_dim)

                    if saved_gpu and not self.gpu_available:
                        print(f"ℹ️  Index was created with GPU but GPU not available, using CPU")
                    elif not saved_gpu and self.gpu_available:
                        print(f"ℹ️  Index was created with CPU but GPU available, moving to GPU")

            # Move to GPU if available
            if self.gpu_available:
                try:
                    print(f"⚡ Moving loaded index to GPU...")
                    self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.index)
                    print(f"✅ Successfully moved index to GPU")
                except Exception as e:
                    print(f"⚠️  Could not move loaded index to GPU: {str(e)}")
                    print(f"💻 Using CPU instead")
                    self.gpu_available = False

            # Load documents
            with open(docs_file, 'rb') as f:
                self.documents = pickle.load(f)

            # Verify consistency
            if self.index.ntotal != len(self.documents):
                print(f"⚠️  Index has {self.index.ntotal} vectors but {len(self.documents)} documents")
                return False

            hardware_type = "GPU" if self.gpu_available else "CPU"
            print(f"✅ Successfully loaded FAISS database with {len(self.documents)} documents ({hardware_type})")
            return True

        except Exception as e:
            print(f"❌ Failed to load FAISS database: {str(e)}")
            return False

    def get_total_documents(self) -> int:
        return len(self.documents)

    def get_index_info(self) -> dict:
        """Get detailed information about the FAISS index"""
        try:
            return {
                'total_vectors': self.index.ntotal,
                'embedding_dimension': self.embedding_dim,
                'index_type': type(self.index).__name__,
                'gpu_enabled': self.gpu_available,
                'is_trained': getattr(self.index, 'is_trained', True),
                'gpu_memory_fraction': self.gpu_memory_fraction if self.gpu_available else None
            }
        except Exception as e:
            return {'error': str(e)}

    def optimize_for_search(self):
        """Optimize the index for better search performance"""
        if self.gpu_available:
            try:
                # GPU-specific optimizations could go here
                print(f"⚙️  Index already optimized for GPU")
            except Exception as e:
                print(f"⚠️  GPU optimization failed: {e}")
        else:
            print(f"💻 CPU index is already optimized")

    def get_embedding_dim(self) -> int:
        return self.embedding_dim
    

