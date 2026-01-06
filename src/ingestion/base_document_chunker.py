from abc import ABC, abstractmethod
from langchain.schema import Document
from typing import List, Tuple


class BaseDocumentChunker(ABC):
    """
    Abstract base class for document chunkers.
    
    Defines the common interface that all document chunker implementations
    must follow, regardless of the underlying library (LangChain, LlamaIndex, etc.).
    """

    def __init__(self, directory_path: str):
        self.directory_path = directory_path
        self._chunk_size = None
        self._chunk_overlap = None
        self._documents = []

    @property
    @abstractmethod
    def chunk_size(self):
        """Get the chunk size."""
        pass

    @chunk_size.setter
    @abstractmethod
    def chunk_size(self, value: int):
        """Set the chunk size."""
        pass

    @property
    @abstractmethod
    def chunk_overlap(self):
        """Get the chunk overlap."""
        pass

    @chunk_overlap.setter
    @abstractmethod
    def chunk_overlap(self, value: int):
        """Set the chunk overlap."""
        pass

    @abstractmethod
    def directory_to_documents(self) -> List:
        """
        Load files from a directory using different loaders based on file extensions.
        
        Returns:
            List of Document objects from all loaded files.
        """
        pass

    @abstractmethod
    def chunk_documents(self, documents: List = None) -> List[Document]:
        """
        Chunk a list of documents.
        
        Args:
            documents: List of documents to chunk. If None, uses self._documents.
            
        Returns:
            List of chunked documents.
        """
        pass

    def get_chunked_texts_list(self, documents: List[Document] = None) -> Tuple[List[str], List[Document]]:
        if documents is None:
            documents = self._documents
        document_chunks = self.chunk_documents(documents)

        print(f"📊 Loaded {len(documents)} documents and created {len(document_chunks)} chunks")
        return [doc.page_content for doc in document_chunks], document_chunks

    @abstractmethod
    def calculate_optimal_chunk_parameters_given_max_tokens(self, max_tokens: int, avg_words_per_token: float) -> Tuple[
        int, int]:
        """
        Calculate optimal chunk size and overlap based on max tokens.
        
        Args:
            max_tokens: Maximum tokens allowed by the embedding model.
            
        Returns:
            Tuple of (chunk_size, chunk_overlap) in characters.
        """
        pass

    @abstractmethod
    def validate_and_fix_chunks(self, chunk_texts: List[str], max_tokens: int) -> List[str]:
        """
        Additional validation and fixing of chunks that might still be too long.
        
        Args:
            chunk_texts: List of text chunks to validate.
            max_tokens: Maximum tokens allowed per chunk.
            
        Returns:
            List of validated and potentially fixed text chunks.
        """
        pass

    def recommend_embedding_dimension(self) -> int:
        """
        Recommend embedding dimension based on chunk size.
        
        Returns:
            int: Recommended embedding dimension.
        """
        if self.chunk_size <= 512:
            return 384  # Suitable for small chunks
        elif self.chunk_size <= 1024:
            return 768  # Suitable for medium chunks
        else:
            return 1024  # Suitable for large chunks

    def calculate_max_char_sub_tokens_per_word(self, avg_words_per_token) -> int:
        """
        Estimate maximum character sub-tokens per word based on chunk size.
        
        Returns:
            int: Estimated maximum character sub-tokens per word.
        """
        max_doc_words = self._get_max_document_words()
        return int(max_doc_words / avg_words_per_token if avg_words_per_token > 0 else 0)

    @abstractmethod
    def _get_max_document_words(self) -> int:
        """
        Get the maximum word count from all documents.
        Implementation depends on document format.
        
        Returns:
            int: Maximum word count.
        """
        pass

    def calculate_optimal_chunk_size(self, max_tokens: int, words_per_token: float = 0.75) -> Tuple[int, int]:
        """
        Calculate optimal chunk_size and chunk_overlap based on max_tokens.
        
        Args:
            max_tokens: Maximum tokens per chunk
            words_per_token: Average words per token (varies by language/model)
        
        Returns:
            Tuple[int, int]: (chunk_size_chars, chunk_overlap_chars)
        """
        # Convert tokens to approximate word count
        max_words = int(max_tokens * words_per_token)

        # Average characters per word (including spaces) is ~5-6
        avg_chars_per_word = 5.5
        chunk_size_chars = int(max_words * avg_chars_per_word)

        # Overlap should be 10-20% of chunk size
        chunk_overlap_chars = int(chunk_size_chars * 0.20)

        chunk_size_chars = min(max_tokens, chunk_size_chars)  # Ensure chunk size does not exceed max_tokens

        print(f"📐 Calculated chunk parameters:")
        print(f"   Max tokens: {max_tokens}")
        print(f"   Max words: {max_words}")
        print(f"   Chunk size: {chunk_size_chars} characters")
        print(f"   Chunk overlap: {chunk_overlap_chars} characters")

        return chunk_size_chars, chunk_overlap_chars
