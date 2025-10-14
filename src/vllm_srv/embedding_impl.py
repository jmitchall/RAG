# Import torch - this is PyTorch, a library for running AI models
# Import numpy library - this handles mathematical arrays (lists of numbers) very efficiently
# It's like a calculator that can work with thousands of numbers at once
import numpy as np
import requests
import torch
# Import List type hint - this tells Python we're working with lists (like [1, 2, 3])
from typing import List, Dict, Tuple


# Define a new class called EmbeddingManager - this handles converting text into mathematical vectors
# Think of it like a translator that converts human language into numbers that computers can understand
class EmbeddingManager:

    # Constructor function that sets up our embedding manager when we create it
    # model_name: which AI model to use for creating embeddings (default is a good general-purpose one)
    # embedding_dim: how many numbers each text gets converted into (like 384 or 1024 numbers per sentence)
    # model_host: the web address where our embedding server is running (like a website URL)
    # use_server: whether to use a remote server (True) or run the model locally on this computer (False)
    # max_tokens: maximum number of tokens per text chunk (default 400 to stay under 512 limit)
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"
                 , embedding_dim: int = 1024
                 , model_host: str = "http://localhost:8001", use_server: bool = True
                 , max_tokens: int = 400):

        # Store the name of the AI model we want to use (like "BAAI/bge-base-en-v1.5")
        self.model_name = model_name

        # Store how many numbers each embedding should have (all embeddings must have same size)
        self.embedding_dim = embedding_dim

        # Store the web address where our embedding server is running
        self.model_host = model_host.rstrip('/')  # Remove trailing slash if present

        # Store whether we're using a server (True) or local model (False)
        self.use_server = use_server

        # Store max tokens to prevent context length errors
        self.max_tokens = max_tokens

        # Call our private method to load the model (the __ makes it private)
        self.__load_model()

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

    def check_if_server_is_running(self) -> bool:
        """Check if the vLLM server is running and reachable."""
        # Test server connection
        try:
            response = requests.get(f"{self.model_host}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ vLLM server is running and healthy")
                return True
            else:
                print(f"⚠️  vLLM server responded but may have issues (status: {response.status_code})")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Cannot connect to vLLM server: {e}")
            print(
                f"💡 Make sure to start the server with: uv run vllm serve {self.model_name} --host 127.0.0.1 --port 8001")
            # raise ConnectionError(f"Cannot connect to vLLM server at {self.model_host}")
            return False

    def load_local_model(self):
        """Load model locally using transformers library instead of vLLM server."""
        self.use_server = False
        # Import transformers library - this loads AI models directly onto our computer
        from transformers import AutoModel, AutoTokenizer

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
            self.model_name
            # , cache_dir="./model_cache"  # Optional: specify custom cache directory
        )

        # Set the model to evaluation mode - this makes it ready for making predictions (not training)
        self.model.eval()

        # Check if we have a CUDA-compatible GPU available for faster processing
        if torch.cuda.is_available():
            # Move the model to GPU memory for much faster processing
            # GPU can do thousands of calculations in parallel, CPU does them one by one
            self.model.to('cuda')
            print(f"⚡ Model loaded on GPU")
        else:
            print(f"💻 Model loaded on CPU")
        print(f"✅ Local model is ready to get embeddings")

    # Private method to load either a server client or local model
    # This decides whether to connect to a remote server or load a model on this computer
    def __load_model(self):
        """
        Load model for embeddings using either vLLM server or local transformers.
        
        How To Start vLLM Embedding Server:
        1. Install vLLM: uv add vllm
        
        2. Start the embedding server:
           uv run vllm serve BAAI/bge-base-en-v1.5 \
               --host 127.0.0.1 \
               --port 8001 \
               --gpu-memory-utilization 0.8
        
        3. Test the server:
           curl http://localhost:8001/v1/embeddings \
             -H "Content-Type: application/json" \
             -d '{
               "model": "BAAI/bge-base-en-v1.5",
               "input": ["Hello, world!", "How are you today?"]
             }'
        
        4. If you want to run the model locally instead, set use_server=False
        """

        # Check if we want to use a server (instead of loading model locally)
        if self.use_server:
            print(f"🌐 Using vLLM server at {self.model_host}")
            print(f"📏 Text will be truncated to {self.max_tokens} tokens to avoid context length errors")
            # Test server connection
            if not self.check_if_server_is_running():
                self.load_local_model()
            else:
                print(f"✅ Ready to get embeddings from server")

        # If we're NOT using a server, load the model locally on this computer
        else:
            self.load_local_model()

    # Method to get an embedding (vector) for a single piece of text
    # text: the string we want to convert to numbers, returns: array of numbers representing the text
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text string"""

        # Truncate text to prevent token limit errors
        text = self._truncate_text(text)

        # Check if we're using a server to do the embedding work
        if self.use_server:
            # Prepare the request data for the vLLM server
            data = {
                "model": self.model_name,
                "input": [text]
            }
            # check if server is reachable

            try:
                # Send a POST request to the vLLM server's embeddings endpoint
                response = requests.post(
                    f"{self.model_host}/v1/embeddings",
                    headers={"Content-Type": "application/json"},
                    json=data,
                    timeout=30
                )

                # Check if the request was successful
                if response.status_code == 200:
                    result = response.json()
                    # Extract the embedding from the response
                    embedding = np.array(result['data'][0]['embedding'])
                    return embedding
                else:
                    raise Exception(f"Server returned status {response.status_code}: {response.text}")

            except requests.exceptions.RequestException as e:
                raise Exception(f"Failed to get embedding from server: {e}")

        # If we're using a local model
        else:
            # Import torch for local processing
            import torch

            # Convert our text into numbers using the tokenizer
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=self.max_tokens)

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

    # Method to get embeddings for multiple pieces of text at once (more efficient than one-by-one)
    # texts: list of strings to convert, returns: list of number arrays
    def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for a list of text strings"""

        # Truncate all texts to prevent token limit errors
        texts = [self._truncate_text(text) for text in texts]

        # Check if we're using a server for the embedding work
        if self.use_server:
            # Process in smaller batches to avoid overwhelming the server
            batch_size = 50  # Process 50 texts at a time
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                # Prepare the request data for multiple texts
                data = {
                    "model": self.model_name,
                    "input": batch_texts
                }

                try:
                    # Send batch to the server
                    response = requests.post(
                        f"{self.model_host}/v1/embeddings",
                        headers={"Content-Type": "application/json"},
                        json=data,
                        timeout=120  # Longer timeout for batch processing
                    )

                    # Check if the request was successful
                    if response.status_code == 200:
                        result = response.json()
                        # Extract embeddings for each text from the response
                        batch_embeddings = [np.array(item['embedding']) for item in result['data']]
                        all_embeddings.extend(batch_embeddings)
                    else:
                        raise Exception(f"Server returned status {response.status_code}: {response.text}")

                except requests.exceptions.RequestException as e:
                    raise Exception(f"Failed to get embeddings from server for batch {i // batch_size + 1}: {e}")

            return all_embeddings

        # If we're using a local model
        else:
            # Import torch for local processing
            import torch

            # Convert all texts to numbers at once using the tokenizer
            inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True,
                                    max_length=self.max_tokens)

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
                "max_context": 512,
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
    def detect_embedding_dimension(model_name: str, use_server: bool = True, max_tokens: int = 250) -> int:
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
                use_server=use_server,
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

    @staticmethod
    def choose_optimal_max_tokens( model_name: str, use_server: bool = True,
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

        # Apply additional reductions for server/batch usage
        if use_server:
            # Server calls might have additional overhead
            base_tokens = int(base_tokens * 0.85)  # Reduced from 0.9 to 0.85
            print(f"   Reduced by 15% for server usage: {base_tokens}")

        if batch_processing:
            # Batch processing might need extra buffer
            base_tokens = int(base_tokens * 0.8)  # Reduced from 0.85 to 0.8
            print(f"   Reduced by 20% for batch processing: {base_tokens}")

        print(f"🎯 Selected max_tokens: {base_tokens}")
        return base_tokens

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
                use_server=use_embedding_server,
                batch_processing=batch_processing,
                safety_level = safety_level
            )
            # Step 2: Detect the correct embedding dimension
            actual_embedding_dim = EmbeddingManager.detect_embedding_dimension(
                model_name=embedding_model,
                use_server=use_embedding_server,
                max_tokens=max_tokens
            )
            return max_tokens, actual_embedding_dim
