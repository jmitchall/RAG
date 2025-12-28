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
from abc import ABC
# Import transformers library - this loads AI models directly onto our computer
from transformers import AutoModel, AutoTokenizer
# Import List type hint - this tells Python we're working with lists (like [1, 2, 3])
from typing import List, Dict, Tuple


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

        # Map dtype string to torch dtype
        self.dtype_map = {
            "float16": torch.float16,  # Half precision for efficiency
            "bfloat16": torch.bfloat16,
            "float32": torch.float32
        }
        self.torch_dtype = self.dtype_map.get(dtype, torch.float16)
        self.load_model()

    def load_model(self):
        self.load_local_model()

    def _truncate_text(self, text: str) -> str:
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
        # Truncate text to prevent token limit errors
        text = self._truncate_text(text)

        embedding = self.get_embedding_local(text)

        # Return the embedding (array of numbers representing our text)
        return embedding

    def get_embedding_local(self, text: str) -> np.ndarray:
        """
        Generate embeddings for the given text using a local transformer model.
        Args:
            text: Input text string
        Returns:
            np.ndarray: Embedding vector as a numpy array
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_tokens
        )

        # If we have a GPU available, move our tokenized text to GPU memory
        if torch.cuda.is_available():
            inputs = {key: val.to('cuda') for key, val in inputs.items()}

        # Run the AI model on our tokenized text to get an embedding
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Convert the model's output into a single vector representing our text
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        # Return the embedding (array of numbers representing our text)
        return embedding

    def get_embeddings_local(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for a batch of texts using a local transformer model.
        Args:
            texts: List of input text strings
        Returns:
            List[np.ndarray]: List of embedding vectors as numpy arrays
        """
        # Tokenize batch
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_tokens
        )

        # Move tokenized texts to GPU if available
        if torch.cuda.is_available():
            inputs = {key: val.to('cuda') for key, val in inputs.items()}

        # Process all texts through the AI model at once (batch processing)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Convert model outputs to final embeddings
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        # Convert to list of arrays and return
        return [emb for emb in embeddings]

    def load_local_model(self):
        self.use_server = False
        """Load model locally using transformers library instead of vLLM server."""

        print(f"💻 Loading model locally: {self.model_name}")
        print(f"📏 Text will be truncated to {self.max_tokens} tokens to avoid context length errors")

        # Load a tokenizer - this converts text into numbers that the AI model can understand
        # Think of it like a dictionary that maps words to numbers
        # This will:
        # 1. First Download the tokenizer files from Hugging Face
        #
        # * Download the tokenizer files from Hugging Face
        # * Store them in a local cache directory
        # * Print messages like: Downloading tokenizer_config.json...
        #
        # 2. Cache Location:
        # Linux/Mac
        # ~/.cache/huggingface/hub/
        # Windows
        # C:\Users\<username>\.cache\huggingface\hub\
        # Or use environment variable
        # $HF_HOME/hub/
        #
        # 3. Subsequent Loads:
        # The next time you load the same model, it will use the cached files
        # and print messages like: Already cached, loading...
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
            # , cache_dir="./model_cache"  # Optional: specify custom cache directory
        )
        # Check where it's cached
        print(f"Cache directory: {self.tokenizer.name_or_path}")
        print(f"Full cache path: {self.tokenizer.init_kwargs.get('name_or_path')}")
        print(f"✅ Tokenizer loaded")

        # ~/.cache/huggingface/hub/
        # └── models--BAAI--bge-base-en-v1.5/
        #    ├── snapshots/
        #    │   └── a94b870e7f/
        #    │       ├── config.json
        #    │       ├── tokenizer_config.json
        #    │       ├── vocab.txt
        #    │       ├── special_tokens_map.json
        #    │       └── tokenizer.json
        #    └── refs/
        #        └── main

        # Load the actual AI model that creates embeddings
        # This downloads and loads the model weights (the AI's "brain")
        # This will:
        # 1. Download the model files from Hugging Face
        # * Download the model files (BIG files, can be hundreds of MB)
        # * Store them in the local cache directory
        # * Print messages like: Downloading pytorch_model.bin...
        #
        # # 2. Cache Location:
        # Linux/Mac
        # ~/.cache/huggingface/hub/
        # Windows
        # C:\Users\<username>\.cache\huggingface\hub\
        # Or use environment variable
        # $HF_HOME/hub/
        #
        # 3. Subsequent Loads:
        # The next time you load the same model, it will use the cached files
        # and print messages like: Already cached, loading...
        #
        # ~/.cache/huggingface/hub/models--BAAI--bge-base-en-v1.5/
        # ├── config.json              # Model architecture config
        # ├── pytorch_model.bin        # The actual neural network weights (BIG file!)
        # └── model.safetensors        # Alternative format for weights
        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype  # Apply dtype from server config
            # , cache_dir="./model_cache"  # Optional: specify custom cache directory
        )

        # Set the model to evaluation mode - this makes it ready for making predictions (not training)
        self.model.eval()

        # Check if we have a CUDA-compatible GPU available for faster processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Move the model to GPU memory for much faster processing
            # GPU can do thousands of calculations in parallel, CPU does them one by one
            self.model.to('cuda')
            print(f"⚡ Model loaded on GPU with dtype={self.torch_dtype}")
        else:
            print(f"💻 Model loaded on CPU")
        print(f"✅ Local model is ready to get embeddings")

    def get_actual_embedding_dimension(self) -> int:
        """
        Get the actual embedding dimension by testing with a sample text.
        This is used as a verification step after creating the manager.
        """
        try:
            sample_embedding = self.get_embedding("test")
            actual_dim = len(sample_embedding)
            print(f"✅ Verified embedding dimension: {actual_dim}")
            return actual_dim
        except Exception as e:
            print(f"⚠️  Could not verify embedding dimension: {e}")
            return 768

    @staticmethod
    # Additional methods will be defined here as needed
    def get_supported_model_token_limits(model_name: str) -> Dict[str, int]:
        """
        Get token limits for different embedding models.
        Returns dict with 'max_context', 'recommended_max', and 'safe_max' tokens.
        """
        model_limits = {
            # BAAI BGE Models - Reduced limits for better safety
            "BAAI/bge-base-en-v1.5": {
                "max_context": 512,
                "recommended_max": 350,  # Reduced from 400
                "safe_max": 250  # Reduced from 300
            },
            "BAAI/bge-small-en-v1.5": {
                "max_context": 512,
                "recommended_max": 350,
                "safe_max": 250
            },
            "BAAI/bge-large-en-v1.5": {
                "max_context": 608,
                "recommended_max": 350,
                "safe_max": 250
            },

            # Sentence Transformers
            "sentence-transformers/all-MiniLM-L6-v2": {
                "max_context": 256,
                "recommended_max": 180,  # Reduced
                "safe_max": 120  # Reduced
            },
            "sentence-transformers/all-mpnet-base-v2": {
                "max_context": 384,
                "recommended_max": 280,  # Reduced
                "safe_max": 200  # Reduced
            },
            "sentence-transformers/all-MiniLM-L12-v2": {
                "max_context": 256,
                "recommended_max": 180,
                "safe_max": 120
            },

            # OpenAI Models (these have large contexts so less reduction needed)
            "text-embedding-ada-002": {
                "max_context": 8191,
                "recommended_max": 7000,
                "safe_max": 6000
            },
            "text-embedding-3-small": {
                "max_context": 8191,
                "recommended_max": 7000,
                "safe_max": 6000
            },
            "text-embedding-3-large": {
                "max_context": 8191,
                "recommended_max": 7000,
                "safe_max": 6000
            },

            # Cohere Models
            "embed-english-v3.0": {
                "max_context": 512,
                "recommended_max": 350,
                "safe_max": 250
            },

            # E5 Models
            "intfloat/e5-base-v2": {
                "max_context": 512,
                "recommended_max": 350,
                "safe_max": 250
            },
            "intfloat/e5-large-v2": {
                "max_context": 512,
                "recommended_max": 350,
                "safe_max": 250
            },
        }

        # Default fallback for unknown models
        default_limits = {
            "max_context": 512,
            "recommended_max": 350,
            "safe_max": 250
        }

        return model_limits.get(model_name, default_limits)

    @staticmethod
    def estimate_optimal_max_token_actual_embeddings(embedding_model: str, use_embedding_server: bool = True
                                                     , batch_processing: bool = True
                                                     , safety_level: str = "recommended"
                                                     , max_tokens: int = None) -> Tuple[int, int]:
        """ Estimate optimal max_tokens and actual embedding dimension for a given model.
        Args:
            embedding_model: Name of the embedding model
            use_embedding_server: Whether to use an external embedding server
            batch_processing: Whether processing many texts at once
            safety_level: "safe", "recommended", or "max" (how conservative to be)
            max_tokens: If provided, skip estimation and use this value directly
        Returns:
            Tuple[int, int]: (optimal max_tokens, actual embedding dimension)
        """

        # Step 1: Choose optimal max_tokens based on the model
        if max_tokens is None:
            max_tokens = EmbeddingManager.choose_optimal_max_tokens(
                model_name=embedding_model,
                batch_processing=batch_processing,
                safety_level=safety_level
            )
            # Step 2: Detect the correct embedding dimension
            actual_embedding_dim = EmbeddingManager.detect_embedding_dimension(
                model_name=embedding_model,
                max_tokens=max_tokens
            )
            return max_tokens, actual_embedding_dim

    @staticmethod
    def choose_optimal_max_tokens(model_name: str,
                                  batch_processing: bool = True, safety_level: str = "recommended") -> int:
        """
        Choose optimal max_tokens based on model, usage pattern, and safety requirements.

        Args:
            model_name: Name of the embedding model
            use_server: Whether using a server (may need more conservative limits)
            batch_processing: Whether processing many texts at once
            safety_level: "safe", "recommended", or "max" (how conservative to be)

        Returns:
            int: Optimal max_tokens value
        """
        limits = EmbeddingManager.get_supported_model_token_limits(model_name)

        print(f"📏 Token limits for {model_name}:")
        print(f"   Max context: {limits['max_context']} tokens")
        print(f"   Recommended: {limits['recommended_max']} tokens")
        print(f"   Safe: {limits['safe_max']} tokens")

        # Choose based on safety level
        if safety_level == "max":
            base_tokens = limits['max_context']
        elif safety_level == "safe":
            base_tokens = limits['safe_max']
        else:  # recommended
            base_tokens = limits['recommended_max']

        if batch_processing:
            # Batch processing might need extra buffer
            base_tokens = int(base_tokens * 0.8)  # Reduced from 0.85 to 0.8
            print(f"   Reduced by 20% for batch processing: {base_tokens}")

        print(f"🎯 Selected max_tokens: {base_tokens}")
        return base_tokens

    @staticmethod
    def detect_embedding_dimension(model_name: str, max_tokens: int = 250) -> int:
        """
        Detect the actual embedding dimension from the model by making a test call.
        This is the most reliable way to get the correct dimension.
        """
        print(f"🔍 Detecting embedding dimension for model: {model_name}")
        limits = EmbeddingManager.get_supported_model_token_limits(model_name)
        try:

            # Use a very short test string to avoid token limit issues
            test_text = "test"

            # Create a temporary embedding manager with a placeholder dimension
            temp_manager = EmbeddingManager(
                model_name=model_name,
                embedding_dim=limits['max_context'],  # Placeholder, will be overridden
                max_tokens=max_tokens
            )

            # Get a test embedding to determine actual dimension
            test_embedding = temp_manager.get_embedding(test_text)
            actual_dim = len(test_embedding)

            print(f"✅ Detected embedding dimension: {actual_dim}")
            return actual_dim

        except Exception as e:
            print(f"⚠️  Could not detect embedding dimension: {e}")

            # Fallback to known dimensions for common models
            model_dimensions = {
                "BAAI/bge-base-en-v1.5": 768,
                "BAAI/bge-small-en-v1.5": 384,
                "BAAI/bge-large-en-v1.5": 1024,
                "sentence-transformers/all-MiniLM-L6-v2": 384,
                "sentence-transformers/all-mpnet-base-v2": 768,
                "text-embedding-ada-002": 1536,
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
            }

            fallback_dim = model_dimensions.get(model_name, 768)
            print(f"📚 Using known dimension for {model_name}: {fallback_dim}")
            return fallback_dim