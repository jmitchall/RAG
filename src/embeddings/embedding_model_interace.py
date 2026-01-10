import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict
from refection_logger import logger
from vllm import LLM


class EmbeddingModelInterface(ABC):
    """
    Abstract base class for embedding models.
    Defines the interface for text-to-vector embedding models.
    """

    def __init__(self, max_tokens: int):
        """
        Initialize the embedding model interface.

        Args:
            max_tokens: Maximum tokens per text chunk
        """
        # Store max tokens to prevent context length errors
        self._max_tokens = max_tokens
        self._safe_embedding_dim = 0
        logger.info(f"📏 Text will be truncated to {self.max_tokens} tokens to avoid context length errors")

    @property
    def safe_embedding_dim(self) -> int:
        return self._safe_embedding_dim

        # get property max_tokens

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @max_tokens.setter
    def max_tokens(self, value: int):
        self._max_tokens = value

    @staticmethod
    def createte_instance(**kwargs):
        """
        Create an instance of the embedding model.

        Args:
            kwargs: Additional parameters for model creation
        """
        raise NotImplementedError

    def truncate_text(self, text: str) -> str:
        """
        Truncate text to fit within token limits.
        Simple word-based approximation: ~0.75 tokens per word.
        """
        words = text.split()
        # Estimate max words based on token limit (conservative estimate)
        max_words = int(self.max_tokens * 0.75)

        if len(words) > max_words:
            truncated = ' '.join(words[:max_words])
            return truncated
        return text

    def get_embedding(self, text: str):
        """
        Get embedding vector for a single text string.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector
        """
        raise NotImplementedError

    def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Get embeddings for multiple texts (batch processing).

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        raise NotImplementedError

    def get_supported_model_token_limits(model_name: str) -> Dict[str, int]:
        """
        Get supported model token limits.

        Args:
            model_name: Name of the embedding model

        Returns:
            Dict[str, int]: Mapping of model names to their token limits
        """
        raise NotImplementedError

    def calculate_avg_words_per_token(self, documents: List[str]) -> float:
        """
        Calculate average words per token using actual tokenizer.
        
        Returns:
            float: Average words per token (typically ~0.75 for English)
        """
        return 0.75  # Default fallback
