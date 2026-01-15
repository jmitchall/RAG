#!/usr/bin/env python3
"""
FAISS Vector Database Implementation

Author: Jonathan A. Mitchall
Version: 1.0
Last Updated: January 10, 2026

License: MIT License

Copyright (c) 2026 Jonathan A. Mitchall

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Revision History:
    2026-01-10 (v1.0): Initial comprehensive documentation
"""

import faiss
import numpy as np
import os
import pickle
import torch
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import List, Optional
from vectordatabases.vector_db_interface import VectorDBInterface
from refection_logger import logger

class FaissVectorDB(VectorDBInterface):
    """FAISS implementation of vector database with enhanced GPU support
    The Main Components Are:
    - self.index: The FAISS index object for vector storage and search
    - self.documents: List of Document objects corresponding to indexed vectors
    - self.embedding_dim : Dimension of the embeddings used in the index
    - self.persist_path: Optional path for saving/loading the index and documents
    - self.gpu_available: Flag indicating if GPU acceleration is enabled
    - self.gpu_resources: FAISS GPU resources object for managing GPU memory
    - self.softmax_temperature: Temperature parameter for confidence score softmax normalization
    """

    def __init__(self, embedding_dim: int = 0, persist_path: Optional[str] = None, use_gpu: bool = True,
                 gpu_memory_fraction: float = 0.8, **kwargs):
        super().__init__(embedding_dim, persist_path, **kwargs)

        self.use_gpu = use_gpu
        self.gpu_memory_fraction = gpu_memory_fraction
        self.gpu_available = False
        self.gpu_resources = None
        self.softmax_temperature = kwargs.get('softmax_temperature', 1.0)

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
                logger.info(f"ℹ️  FAISS CPU version detected (no GPU support)")
                logger.info(f"💡 For GPU support, install: pip install faiss-gpu")
                self.gpu_available = False
                return

            # Check if CUDA is available
            if not torch.cuda.is_available():
                logger.info(f"ℹ️  CUDA not available, using CPU")
                self.gpu_available = False
                return

            # Get GPU information
            gpu_count = torch.cuda.device_count()
            current_gpu = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_gpu)
            gpu_memory = torch.cuda.get_device_properties(current_gpu).total_memory

            logger.info(f"🚀 GPU detected: {gpu_name}")
            logger.info(f"💾 GPU Memory: {gpu_memory / 1e9:.1f} GB")
            logger.info(f"🔢 Available GPUs: {gpu_count}")

            # Setup GPU resources with memory management
            self.gpu_resources = faiss.StandardGpuResources()

            # Set memory fraction to avoid OOM
            available_memory = int(gpu_memory * self.gpu_memory_fraction)
            self.gpu_resources.setTempMemory(available_memory)

            logger.info(f"⚙️  Allocated {available_memory / 1e9:.1f} GB GPU memory for FAISS")

            self.gpu_available = True

        except Exception as e:
            logger.info(f"⚠️  GPU setup failed: {str(e)}")
            logger.info(f"💻 Falling back to CPU")
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
                logger.info(f"✅ Created new FAISS vector database with GPU acceleration")

                # logger.info GPU index information
                logger.info(f"⚡ GPU Index Type: {type(self.index).__name__}")

            except Exception as e:
                logger.info(f"⚠️  GPU index creation failed: {str(e)}")
                logger.info(f"💻 Using CPU index instead")
                self.gpu_available = False
                # Recreate CPU index
                self.index = faiss.IndexFlatL2(self.embedding_dim)
        else:
            logger.info(f"💻 Created new FAISS vector database with CPU")

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
                logger.info(f"🚀 Using GPU for embedding preprocessing...")

                # Convert to GPU tensors for normalization (optional but can improve search quality)
                gpu_embeddings = torch.tensor(embeddings_array, device='cuda')

                # Normalize embeddings on GPU (L2 normalization for better cosine similarity)
                gpu_embeddings = torch.nn.functional.normalize(gpu_embeddings, p=2, dim=1)

                # Convert back to CPU numpy for FAISS
                embeddings_array = gpu_embeddings.cpu().numpy().astype(np.float32)

                logger.info(f"✅ Preprocessed {len(embeddings)} embeddings on GPU")

            except Exception as e:
                logger.info(f"⚠️  GPU preprocessing failed: {e}")
                logger.info(f"💻 Using original embeddings")

        # Add to index
        try:
            self.index.add(embeddings_array)
            self.documents.extend(docs)

            hardware_type = "GPU" if self.gpu_available else "CPU"
            logger.info(f"📄 Added {len(docs)} documents to FAISS vector database ({hardware_type})")
            logger.info(f"📊 Total documents: {len(self.documents)}")

            # logger.info index statistics
            logger.info(f"🔢 Index size: {self.index.ntotal} vectors")

        except Exception as e:
            logger.info(f"❌ Failed to add documents: {e}")
            raise

    def _confidence_softmax(self, distances: np.ndarray) -> np.ndarray:
        """Softmax normalization - probabilities sum to 100%"""
        negative_distances = -distances / self.softmax_temperature
        exp_scores = np.exp(negative_distances - np.max(negative_distances))
        probabilities = exp_scores / np.sum(exp_scores)
        return probabilities

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Document]:
        if len(self.documents) == 0:
            logger.info("⚠️  No documents in vector database to search")
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
                logger.info(f"⚠️  GPU query preprocessing failed: {e}")
                query_array = np.array([query_embedding], dtype=np.float32)
        else:
            query_array = np.array([query_embedding], dtype=np.float32)

        try:
            # Perform search
            distances, indices = self.index.search(query_array, top_k)

            # Convert distances to confidence percentages
            confidence_percentages = self._confidence_softmax(distances[0])

            results = []
            for i, (distance, idx, confidence) in enumerate(zip(distances[0], indices[0], confidence_percentages)):
                if idx >= 0 and idx < len(self.documents):
                    doc = self.documents[idx]
                    similarity_score = float(confidence)

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
                logger.info(f"🔍 Found {len(results)} similar documents ({hardware_type} search)")

            # L2 (Euclidean) distances between the query embedding and the matched document embedding
            # Shape: (1, top_k) - the first dimension is for the query batch (always 1 in this code), second is the number of results
            # Lower distances = more similar documents (since it's measuring distance, not similarity)
            # Sort in increasing order on distance (most similar first)
            results.sort(key=lambda doc: doc.metadata.get('distance', 10000.0))

            return results

        except Exception as e:
            logger.info(f"❌ Search failed: {e}")
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
                logger.info(f"💾 Moving GPU index to CPU for saving...")
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

            logger.info(f"💾 FAISS vector database saved to {self.persist_path}")

        except Exception as e:
            logger.info(f"❌ Failed to save database: {e}")
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
            logger.info(f"📥 Loading FAISS index from {index_file}...")
            self.index = faiss.read_index(index_file)

            # Load configuration if available
            if os.path.exists(config_file):
                with open(config_file, 'rb') as f:
                    config_info = pickle.load(f)
                    saved_gpu = config_info.get('gpu_available', False)
                    self.embedding_dim = config_info.get('embedding_dim', self.embedding_dim)

                    if saved_gpu and not self.gpu_available:
                        logger.info(f"ℹ️  Index was created with GPU but GPU not available, using CPU")
                    elif not saved_gpu and self.gpu_available:
                        logger.info(f"ℹ️  Index was created with CPU but GPU available, moving to GPU")

            # Move to GPU if available
            if self.gpu_available:
                try:
                    logger.info(f"⚡ Moving loaded index to GPU...")
                    self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.index)
                    logger.info(f"✅ Successfully moved index to GPU")
                except Exception as e:
                    logger.info(f"⚠️  Could not move loaded index to GPU: {str(e)}")
                    logger.info(f"💻 Using CPU instead")
                    self.gpu_available = False

            # Load documents
            with open(docs_file, 'rb') as f:
                self.documents = pickle.load(f)

            # Verify consistency
            if self.index.ntotal != len(self.documents):
                logger.info(f"⚠️  Index has {self.index.ntotal} vectors but {len(self.documents)} documents")
                return False

            hardware_type = "GPU" if self.gpu_available else "CPU"
            logger.info(f"✅ Successfully loaded FAISS database with {len(self.documents)} documents ({hardware_type})")
            return True

        except Exception as e:
            logger.info(f"❌ Failed to load FAISS database: {str(e)}")
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
                logger.info(f"⚙️  Index already optimized for GPU")
            except Exception as e:
                logger.info(f"⚠️  GPU optimization failed: {e}")
        else:
            logger.info(f"💻 CPU index is already optimized")

    def get_embedding_dim(self) -> int:
        return self.embedding_dim

    def delete_collection(self, **kwargs) -> bool:
        """ 
        Delete Table or collection supported by the 
        database and all files associated to
        it from disk.
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        persist_path: str = kwargs.get('persist_path', self.persist_path)
        try:
            # Clear in-memory data
            self.index.reset()
            self.documents = []

            # Delete persisted files if they exist
            if self.persist_path:
                index_file = f"{persist_path}.faiss"
                docs_file = f"{persist_path}.docs.pkl"
                config_file = f"{persist_path}.config.pkl"

                for file in [index_file, docs_file, config_file]:
                    if os.path.exists(file):
                        os.remove(file)
                        logger.info(f"🗑️  Deleted file: {file}")

            logger.info(f"✅ Successfully deleted FAISS collection and associated files")
            return True

        except Exception as e:
            logger.info(f"❌ Failed to delete collection: {e}")
            return False

    def as_langchain_retriever(self, embedding_function, top_k: int = 5):
        """Convert to LangChain retriever

        Args:
            embedding_function: A callable that takes a string and returns an embedding vector
            top_k: Number of documents to retrieve

        Returns:
            FaissLangChainRetriever: A LangChain-compatible retriever
        """
        return FaissLangChainRetriever(
            vector_db=self,
            embedding_function=embedding_function,
            top_k=top_k
        )


class FaissLangChainRetriever(BaseRetriever):
    """LangChain-compatible retriever for FaissVectorDB
    
    This class wraps FaissVectorDB to make it compatible with LangChain's retriever interface.
    Since BaseRetriever is a Pydantic model, it performs strict type validation on all attributes.
    
    The nested Config class with 'arbitrary_types_allowed = True' is required because:
    - vector_db: FaissVectorDB is a custom class instance (not a standard Pydantic type)
    - embedding_function: callable is a function/callable object (not a standard Pydantic type)
    
    Without this configuration, Pydantic would raise validation errors for these custom types.
    This tells Pydantic to accept these arbitrary Python objects without transformation.
    """
    vector_db: FaissVectorDB
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
        logger.info(f"Max document length in vector DB: {chunk_size} characters")
        query_text_input = query
        if len(query_text_input) > chunk_size:
            logger.info(
                f"⚠️  Query text length ({len(query)}) exceeds max document length in vector DB ({chunk_size}). Truncating query.")
            query_text_input = query_text_input[:chunk_size]

        # Generate embedding for the query
        query_embedding = self.embedding_function(query_text_input)

        # Ensure it's a numpy array
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)

        # Search the vector database
        results = self.vector_db.search(query_embedding, top_k=self.top_k)

        logger.info(f"\n🔍 Retrieved {len(results)} documents for query '{query_text_input}'")
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
            logger.info(f"\n--- Result {i} ---")
            logger.info(f"Content: {doc.page_content}")
            logger.info(f"Source: {doc.metadata.get('source', 'unknown')}")
            logger.info(f"Similarity Score: {doc.metadata.get('similarity_score', 'N/A'):.4f}")
            logger.info(f"distance Score: {doc.metadata.get('distance', 'N/A'):.4f}")
            logger.info(f"Search Rank: {doc.metadata.get('search_rank', 'N/A')}")
            context += doc.page_content + "\n\n"
        return context
