#!/usr/bin/env python3
"""
Vector Database Interface - Abstract Base Class for Vector Databases

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
    def get_embedding_dim(self) -> int:
        """Get the dimension of the embeddings used in the database"""
        pass

    @abstractmethod
    def get_max_document_length(self) -> int:
        """Get the maximum document length supported by the database"""
        pass

    @abstractmethod
    def delete_collection(self, **kwargs) -> bool:
        """Delete Table or collection supported by the database and all files associated to **kwargs"""
        pass
