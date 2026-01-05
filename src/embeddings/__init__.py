#!/usr/bin/env python3
"""
Embedding Manager -  Embedding Generation
==============================================

Converts text into numerical vectors (embeddings) for semantic search.
Uses local transformer models via HuggingFace - no server needed.
"""
# Import torch - this is PyTorch, a library for running AI models
# Import numpy library - this handles mathematical arrays (lists of numbers) very efficiently
# It's like a calculator that can work with thousands of numbers at once
import numpy as np
import torch
# Import transformers library - this loads AI models directly onto our computer
from transformers import AutoModel, AutoTokenizer
# Import List type hint - this tells Python we're working with lists (like [1, 2, 3])
from typing import List, Dict, Tuple

# Define a new class called EmbeddingManager - this handles converting text into mathematical vectors
# Think of it like a translator that converts human language into numbers that computers can understand
class EmbeddingManager:
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
            model_name: str = "BAAI/bge-base-en-v1.5",
            embedding_dim: int = 1024,
            max_tokens: int = 400,
            dtype: str = "float16"
    ):
        """
        Initialize the embedding manager with a local model.

        Args:
            model_name: HuggingFace model identifier
            embedding_dim: Expected embedding dimension (for validation)
            max_tokens: Maximum tokens per text chunk
            dtype: Model precision ("float16", "bfloat16", "float32")
        """

        # Store the name of the AI model we want to use (like "BAAI/bge-base-en-v1.5")
        self.model_name = model_name

        # Store how many numbers each embedding should have (all embeddings must have same size)
        self.embedding_dim = embedding_dim

        # Store max tokens to prevent context length errors
        self.max_tokens = max_tokens
        self.dtype = dtype

        # Map dtype string to torch dtype
        self.dtype_map = {
            "float16": torch.float16, # Half precision for efficiency
            "bfloat16": torch.bfloat16,
            "float32": torch.float32
        }
        self.torch_dtype = self.dtype_map.get(dtype, torch.float16)