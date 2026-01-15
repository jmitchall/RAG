#!/usr/bin/env python3
"""
Embedding Manager - High-level Text Embedding Generation

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

Embedding Manager -  Embedding Generation
==============================================

Converts text into numerical vectors (embeddings) for semantic search.
Uses local transformer models via HuggingFace - no server needed.
"""
# Import torch - this is PyTorch, a library for running AI models
# Import numpy library - this handles mathematical arrays (lists of numbers) very efficiently
# It's like a calculator that can work with thousands of numbers at once
import numpy as np
from abc import ABC
from embeddings.embedding_model_interace import EmbeddingModelInterface
# Import transformers library - this loads AI models directly onto our computer
# Import List type hint - this tells Python we're working with lists (like [1, 2, 3])
from typing import List


# Define a new class called EmbeddingManager - this handles converting text into mathematical vectors
# Think of it like a translator that converts human language into numbers that computers can understand
class EmbeddingManager(ABC):
    """
    Manages text-to-vector embeddings using local transformer models.
    Optimized for RTX GPUs with automatic GPU detection.
    """

    # Constructor function that sets up our embedding manager when we create it
    # model_name: which AI model to use for creating embeddings (default is a good general-purpose one)
    # embedding_dim: how many numbers each text gets converted into (like 384 or 1024 numbers per sentence)
    # max_tokens: maximum number of tokens per text chunk (default 400 to stay under 512 limit)
    # dtype: Model precision ("float16", "bfloat16", "float32")
    def __init__(
            self,
            embedding_model_instance: EmbeddingModelInterface,
            embedding_dim: int = 1024,
    ):
        """
        Initialize the embedding manager with a local model.

        Args:
            model_name: HuggingFace model identifier
            embedding_dim: Expected embedding dimension (for validation)
        """
        # If an embedding model instance is provided, use it directly
        if embedding_model_instance is not None:
            self.embedding_model = embedding_model_instance
        else:
            # throw error if no instance provided
            raise ValueError("An embedding_model_instance must be provided to EmbeddingManager")

        # Store how many numbers each embedding should have (all embeddings must have same size)
        self.embedding_dim = embedding_dim

    # Method to get an embedding (vector) for a single piece of text
    # text: the string we want to convert to numbers, returns: array of numbers representing the text
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding vector for a single text string.

        Args:
            text: Input text to embed

        Returns:
            np.ndarray: Embedding vector
        """
        # Return the embedding (array of numbers representing our text)
        return self.embedding_model.get_embedding(text)

    # Method to get embeddings for multiple pieces of text at once (more efficient than one-by-one)
    # texts: list of strings to convert, returns: list of number arrays
    def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Get embeddings for multiple texts (batch processing).

        Args:
            texts: List of input texts

        Returns:
            List[np.ndarray]: List of embedding vectors
        """
        # Convert all texts to numbers at once using the tokenizer
        return self.embedding_model.get_embeddings(texts)

    def get_actual_embedding_dimension(self) -> int:
        """
        Get the actual embedding dimension by testing with a sample text.
        This is used as a verification step after creating the manager.
        """
        return self.embedding_model.get_actual_embedding_dimension()
