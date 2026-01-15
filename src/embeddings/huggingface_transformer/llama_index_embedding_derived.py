#!/usr/bin/env python3
"""
LlamaIndex HuggingFace Embeddings - LlamaIndex Compatible Wrapper

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

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import Optional
from refection_logger import logger

class LlamaIndexHuggingFaceOfflineEmbeddings(HuggingFaceEmbedding):
    """
    derived class for HuggingFaceEmbedding to be used with LlamaIndex in offline mode.
    Uses a local cache folder to store the model for offline use.
    """

    def __init__(
            self,
            model_name: str = "BAAI/bge-small-en-v1.5",
            cache_folder: Optional[str] = "./models",
            **kwargs
    ):
        """
        Initialize HuggingFaceOfflineEmbeddings.
        
        Args:
            model_name: HuggingFace model identifier
            cache_folder: Folder to cache the model (default: ./models for offline use)
            **kwargs: Additional arguments passed to HuggingFaceEmbedding base class
                     (e.g., device, max_length, normalize, embed_batch_size, etc.)
        """
        # Pass all parameters to the base class constructor
        super().__init__(
            model_name=model_name,
            cache_folder=cache_folder,
            **kwargs
        )

    @property
    def get_tokenizer(self):
        """
        Return the tokenizer used by the embedding model.
        Tries multiple access patterns for compatibility.
        """
        # Try public attribute first
        if hasattr(self, 'tokenizer'):
            return self.tokenizer

        # Try private attribute
        if hasattr(self, '_tokenizer'):
            return self._tokenizer

        # Try through model
        if hasattr(self, '_model') and hasattr(self._model, 'tokenizer'):
            return self._model.tokenizer

        # Last resort: load tokenizer from model_name
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_folder
        )


# Example usage
if __name__ == "__main__":
    # Example 1: Basic initialization with defaults
    embeddings1 = LlamaIndexHuggingFaceOfflineEmbeddings()

    # Example 2: Custom model with device specification
    embeddings2 = LlamaIndexHuggingFaceOfflineEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder="./custom_models",
        device="cpu",
        max_length=512
    )

    # Example 3: Full configuration with all kwargs
    embeddings3 = LlamaIndexHuggingFaceOfflineEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        cache_folder="./models",
        device="cuda",  # or "cpu"
        max_length=512,
        normalize=True,
        embed_batch_size=10
    )

    # Test embedding generation
    text = "This is a sample text for embedding."
    embedding = embeddings2.get_text_embedding(text)
    logger.info(f"Generated embedding dimension: {len(embedding)}")
    logger.info(f"First 5 values: {embedding[:5]}")

    # Test query embedding
    query = "What is machine learning?"
    query_embedding = embeddings2.get_query_embedding(query)
    logger.info(f"\nQuery embedding dimension: {len(query_embedding)}")

    # Test batch embeddings
    texts = [
        "First document",
        "Second document",
        "Third document"
    ]
    batch_embeddings = [embeddings2.get_text_embedding(t) for t in texts]
    logger.info(f"\nGenerated {len(batch_embeddings)} batch embeddings")
