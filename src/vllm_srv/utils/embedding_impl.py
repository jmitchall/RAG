#!/usr/bin/env python3
"""
Embedding Manager -  Embedding Generation
==============================================

Converts text into numerical vectors (embeddings) for semantic search.
Uses local transformer models via HuggingFace - no server needed.
"""
# It's like a calculator that can work with thousands of numbers at once
from embedding.embedding_manager import EmbeddingManager

# Define a new class called EmbeddingManager - this handles converting text into mathematical vectors
# Think of it like a translator that converts human language into numbers that computers can understand
class EmbeddingManagerServer(EmbeddingManager):
    """
    Manages text-to-vector embeddings using local transformer models.
    Optimized for RTX GPUs with automatic GPU detection.
    """

    # Constructor function that sets up our embedding manager when we create it
    # model_name: which AI model to use for creating embeddings (default is a good general-purpose one)
    # embedding_dim: how many numbers each text gets converted into (like 384 or 1024 numbers per sentence)
    # model_host: the web address where our embedding server is running (like a website URL)
    # use_server: whether to use a remote server (True) or run the model locally on this computer (False)
    # max_tokens: maximum number of tokens per text chunk (default 400 to stay under 512 limit)
    def __init__(
            self,
            model_name: str = "BAAI/bge-base-en-v1.5",
            embedding_dim: int = 1024,
            model_host: str = "http://localhost:8001",
            use_server: bool = True,
            max_tokens: int = 400,
            server_config: Dict = None):
        """
        Initialize the embedding manager with a local model.

        Args:
            model_name: HuggingFace model identifier
            embedding_dim: Expected embedding dimension (for validation)
            model_host: the web address where our embedding server is running (like a website URL)
            use_server: whether to use a remote server (True) or run the model locally on this computer (False)
            max_tokens: Maximum tokens per text chunk
        """
        super().__init__(model_name, embedding_dim, max_tokens)

        # Store the web address where our embedding server is running
        self.model_host = model_host.rstrip('/')  # Remove trailing slash if present

        # Store whether we're using a server (True) or local model (False)
        self.use_server = use_server

        # Store server configuration for vLLM
        # Default configuration optimized for embedding models
        self.server_config = server_config or {
            "runner": "pooling",  # Use pooling runner for embeddings (not generative)
            "gpu_memory_utilization": 0.3,  # Conservative memory usage (leaves room for LLM)
            "dtype": "float16",  # Half precision for efficiency
            "host": "127.0.0.1",  # Localhost only for security
            "port": 8001  # Default embedding server port
        }

        # Load the actual AI model with dtype from server_config
        self.torch_dtype = dtype_map.get(self.server_config.get("dtype", "float16"), torch.float16)

        self.load_model()

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

        # Check if we're using a server to do the embedding work
        if not self.use_server:
            embedding = self.get_embedding_local(text)
        else:
            embedding = self.get_embedding_remote(text)
        # Return the embedding (array of numbers representing our text)
        return embedding

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

        # Truncate all texts to prevent token limit errors
        texts = [self._truncate_text(text) for text in texts]
        # Check if we're using a server for the embedding work
        if not self.use_server:
            # Convert all texts to numbers at once using the tokenizer
            return self.get_embeddings_local(texts)
        else:
            return self.get_embeddings_remote(texts)

    @staticmethod
    def print_setup_guide(model_name: str = "BAAI/bge-base-en-v1.5"):
        """
        Print a complete guide for running embedding + LLM servers together on RTX 5080 16GB.
        """
        print("\n" + "=" * 70)
        print("🚀 RTX 5080 16GB: DUAL SERVER SETUP (EMBEDDING + LLM)")
        print("=" * 70)
        print()
        print("💡 This setup runs both servers on the same GPU efficiently:")
        print()

        print("📊 Memory Allocation:")
        print("   • Embedding Model (BAAI/bge-base-en-v1.5): ~3GB VRAM (30%)")
        print("   • LLM Model (Mistral 7B GPTQ):            ~5GB VRAM (50%)")
        print("   • KV Cache + Overhead:                    ~3GB VRAM (20%)")
        print("   • Total:                                  ~11GB / 16GB ✅")
        print()

        print("🔧 Terminal 1 - Start Embedding Server:")
        print(EmbeddingManager.get_vllm_command(model_name))
        print()

        print("🔧 Terminal 2 - Start LLM Server:")
        print(f"""uv run vllm serve TheBloke/OpenHermes-2.5-Mistral-7B-GPTQ \\
        --host 127.0.0.1 \\
        --port 8000 \\
        --quantization gptq \\
        --dtype half \\
        --gpu-memory-utilization 0.6 \\
        --max-model-len 8192""")
        print()

        print("✅ Verify Both Servers:")
        print(f"   curl http://localhost:8001/health  # Embedding server")
        print(f"   curl http://localhost:8000/health  # LLM server")
        print()
        print("🎯 Usage Example:")
        print("""   from embedding.embedding_manager import EmbeddingManager
    from openai import OpenAI

    # Initialize embedding manager (connects to port 8001)
    embedder = EmbeddingManager(
        model_name="BAAI/bge-base-en-v1.5",
        use_server=True
    )

    # Initialize LLM client (connects to port 8000)
    llm_client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="dummy"
    )

    # Get embeddings
    embedding = embedder.get_embedding("Hello world")

    # Generate text
    response = llm_client.chat.completions.create(
        model="TheBloke/OpenHermes-2.5-Mistral-7B-GPTQ",
        messages=[{"role": "user", "content": "Hello!"}]
    )""")
        print()
        print("=" * 70)

    # Private method to load either a server client or local model
    # This decides whether to connect to a remote server or load a model on this computer
    def load_model(self):
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
            print(f"⚙️  Server config: {self.server_config}")
            # Test server connection
            if not self.check_if_server_is_running():
                print(f"\n💡 To start the vLLM embedding server with optimal settings:")
                print(f"   uv run vllm serve {self.model_name} \\")
                print(f"       --host {self.server_config.get('host', '127.0.0.1')} \\")
                print(f"       --port {self.server_config.get('port', 8001)} \\")
                print(f"       --runner {self.server_config.get('runner', 'pooling')} \\")
                print(f"       --gpu-memory-utilization {self.server_config.get('gpu_memory_utilization', 0.3)} \\")
                print(f"       --dtype {self.server_config.get('dtype', 'float16')}")
                print(f"\n📝 Explanation:")
                print(f"   --runner pooling        → Required for embedding models (not text generation)")
                print(f"   --gpu-memory-utilization 0.3 → Leaves ~70% VRAM for LLM server on same GPU")
                print(f"   --dtype float16         → Half precision = 2x faster, same quality")
                print(f"   --host 127.0.0.1        → Localhost only (security)")
                print(f"   --port 8001             → Avoid conflict with LLM on port 8000")
                self.load_local_model()
            else:
                print(f"✅ Ready to get embeddings from server")

        # If we're NOT using a server, load the model locally on this computer
        else:
            self.load_local_model()

    def get_embedding_remote(self, text: str) -> np.ndarray:
        import requests
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

    def get_embeddings_remote(self, texts: List[str]) -> List[np.ndarray]:
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

    def check_if_server_is_running(self) -> bool:
        """Check if the vLLM server is running and reachable."""
        # Test server connection
        import requests
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

    @staticmethod
    def get_vllm_command(model_name: str, server_config: Dict = None) -> str:
        """
        Generate the complete vLLM command to start an embedding server.

        Args:
            model_name: Embedding model to serve
            server_config: Optional configuration overrides

        Returns:
            str: Complete command to start vLLM server
        """
        default_config = {
            "runner": "pooling",
            "gpu_memory_utilization": 0.3,
            "dtype": "float16",
            "host": "127.0.0.1",
            "port": 8001
        }

        # Merge with provided config
        config = {**default_config, **(server_config or {})}

        command = f"""uv run vllm serve {model_name} \\
        --host {config['host']} \\
        --port {config['port']} \\
        --runner {config['runner']} \\
        --gpu-memory-utilization {config['gpu_memory_utilization']} \\
        --dtype {config['dtype']}"""

        return command


if __name__ == "__main__":
    print("🧪 Testing Remote Embedding Manager")
    # Show setup guide
    EmbeddingManager.print_setup_guide()

    # Custom server config
    embedder = EmbeddingManagerServer(
        model_name="BAAI/bge-base-en-v1.5",
        use_server=True,
        server_config={
            "runner": "pooling",
            "gpu_memory_utilization": 0.25,  # Even more conservative
            "dtype": "bfloat16",  # Use bfloat16 instead
            "port": 8002  # Different port
        }
    )
