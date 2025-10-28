import numpy as np
from abc import ABC, abstractmethod
from langchain_core.documents import Document
from typing import List, Optional


class VectorDBInterface(ABC):
    """Abstract interface for vector databases"""

    def __init__(self, embedding_dim: int, persist_path: Optional[str] = None, **kwargs):
        self.embedding_dim = embedding_dim
        self.persist_path = persist_path
        self.documents: List[Document] = []

    @abstractmethod
    def add_documents(self, docs: List[Document], embeddings: List[np.ndarray]) -> None:
        """Add documents and their embeddings to the vector database"""
        pass

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Document]:
        """Search for similar documents given a query embedding"""
        pass

    @abstractmethod
    def save(self) -> None:
        """Save the vector database to disk"""
        pass

    @abstractmethod
    def load(self) -> bool:
        """Load the vector database from disk. Returns True if successful."""
        pass

    @abstractmethod
    def get_total_documents(self) -> int:
        """Get the total number of documents in the database"""
        pass

    @abstractmethod
    def get_embedding_dim() -> int:
        """Get the dimension of the embeddings used in the database"""
        pass

    @abstractmethod
    def get_max_document_length(self) -> int:
        """Get the maximum document length supported by the database"""
        pass