from typing import Optional, Type

from chroma_vector_db import ChromaVectorDB
from faiss_vector_db import FaissVectorDB
from qdrant_vector_db import QdrantVectorDB
from vector_db_interface import VectorDBInterface


class VectorDBFactory:
    """Factory class to create vector database instances"""

    @staticmethod
    def create_vector_db(
            db_type: str,
            embedding_dim: int,
            persist_path: Optional[str] = None,
            **kwargs
    ) -> VectorDBInterface:
        """
        Create a vector database instance
        
        Args:
            db_type: Type of database ('faiss', 'qdrant', 'chroma')
            embedding_dim: Dimension of embeddings
            persist_path: Path for persistence
            **kwargs: Additional arguments specific to each database
        
        Returns:
            VectorDBInterface: Vector database instance
        """
        db_type = db_type.lower()

        if db_type == 'faiss':
            return FaissVectorDB(embedding_dim, persist_path, **kwargs)
        elif db_type == 'qdrant':
            return QdrantVectorDB(embedding_dim, persist_path, **kwargs)
        elif db_type == 'chroma':
            return ChromaVectorDB(embedding_dim, persist_path, **kwargs)
        else:
            raise ValueError(f"Unsupported database type: {db_type}. Choose from 'faiss', 'qdrant', 'chroma'")

    @staticmethod
    def get_available_databases() -> dict:
        """Get information about available database types"""
        available = {}

        # Check FAISS
        try:
            import faiss
            available['faiss'] = {
                'available': True,
                'gpu_support': hasattr(faiss, 'StandardGpuResources'),
                'description': 'Facebook AI Similarity Search - Fast CPU/GPU vector search'
            }
        except ImportError:
            available['faiss'] = {
                'available': False,
                'gpu_support': False,
                'description': 'Install with: uv add faiss-cpu or faiss-gpu'
            }

        # Check Qdrant
        try:
            import qdrant_client
            available['qdrant'] = {
                'available': True,
                'gpu_support': True,
                'description': 'Modern vector database with advanced filtering'
            }
        except ImportError:
            available['qdrant'] = {
                'available': False,
                'gpu_support': True,
                'description': 'Install with: uv add qdrant-client'
            }

        # Check ChromaDB
        try:
            import chromadb
            available['chroma'] = {
                'available': True,
                'gpu_support': False,
                'description': 'Simple and efficient vector database'
            }
        except ImportError:
            available['chroma'] = {
                'available': False,
                'gpu_support': False,
                'description': 'Install with: uv add chromadb'
            }

        return available

    @staticmethod
    def get_vector_db(db_path: str) -> VectorDBInterface:
        """
        Retrieve a persisted vector database from the specified path.
        
        Args:
            db_path: Path to the persisted database
        Returns:
            VectorDBInterface: Vector database instance
        """
        # For simplicity, assume the db_path contains the type as a prefix
        if 'vector_db_faiss' in db_path:
            return FaissVectorDB(0,persist_path=db_path)
        elif 'vector_db_qdrant' in db_path:
            return QdrantVectorDB(0,persist_path=db_path)
        elif 'vector_db_chroma' in db_path:
            return ChromaVectorDB(0,persist_path=db_path)
        else:
            raise ValueError(f"Cannot determine database type from path: {db_path}")

    @staticmethod
    def get_actual_db_embedding_dim(vector_db: VectorDBInterface) -> int:
        """
        Get the actual embedding dimension used by the vector database.
        
        Args:
            vector_db: Instance of VectorDBInterface
        Returns:
            int: Actual embedding dimension
        """
        return vector_db.get_embedding_dim()
