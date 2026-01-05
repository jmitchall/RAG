"""
Embedding Manager -  Embedding Generation
==============================================

Converts text into numerical vectors (embeddings) for semantic search.
Uses local transformer models via HuggingFace - no server needed.
"""

# It's like a calculator that can work with thousands of numbers at once
from embeddings.embedding_manager import EmbeddingManager


# Import transformers library - this loads AI models directly onto our computer
# Import List type hint - this tells Python we're working with lists (like [1, 2, 3])


# Define a new class called EmbeddingManager - this handles converting text into mathematical vectors
# Think of it like a translator that converts human language into numbers that computers can understand
class EmbeddingManagerLocal(EmbeddingManager):
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
            dtype: str = "float16"):
        super().__init__(model_name, embedding_dim, max_tokens, dtype)

    @staticmethod
    def print_setup_guide(model_name: str = "BAAI/bge-base-en-v1.5"):
        """
        Print a complete guide for running embedding + LLM servers together on RTX 5080 16GB.
        """
        print("\n" + "=" * 70)
        print("🚀 LOCAL EMBEDDING SETUP GUIDE")
        print("=" * 70)
        print()
        print("✅ No server needed - runs directly in Python!")
        print()
        print("📦 Install dependencies:")
        print("   pip install transformers torch numpy")
        print()
        print("🧪 Basic usage:")
        print("""
    from embedding.embedding_manager import EmbeddingManager

    # Create embedding manager
    embedder = EmbeddingManager(
        model_name="{model_name}",
        max_tokens=350
    )

    # Get single embedding
    embedding = embedder.get_embedding("Hello world")
    print(f"Embedding shape: {embedding.shape}")

    # Get batch embeddings
    texts = ["First text", "Second text", "Third text"]
    embeddings = embedder.get_embeddings(texts)
    print(f"Got {len(embeddings)} embeddings")
        """)
        print()
        print("💡 Recommended models for 16GB GPU:")
        print("   • BAAI/bge-base-en-v1.5    - Best quality (768 dim)")
        print("   • BAAI/bge-small-en-v1.5   - Faster (384 dim)")
        print("   • all-MiniLM-L6-v2         - Fastest (384 dim)")
        print()
        print("🎯 Memory usage:")
        print("   • bge-base:  ~2GB VRAM")
        print("   • bge-small: ~1GB VRAM")
        print("   • MiniLM:    ~500MB VRAM")
        print()
        print("=" * 70)


if __name__ == "__main__":
    print("🧪 Testing Local Embedding Manager")
    
    # Test with default model
    print("\n📝 Testing BAAI/bge-base-en-v1.5...")
    embedder = EmbeddingManagerLocal()
    
    # Show setup guide
    embedder.print_setup_guide()

    # Test single embedding
    test_text = "This is a test sentence for embedding."
    embedding = embedder.get_embedding(test_text)
    print(f"✅ Single embedding shape: {embedding.shape}")

    # Test batch embeddings
    test_texts = [
        "First example sentence",
        "Second example sentence",
        "Third example sentence"
    ]
    embeddings = embedder.get_embeddings(test_texts)
    print(f"✅ Batch embeddings count: {len(embeddings)}")
    print(f"✅ Each embedding shape: {embeddings[0].shape}")

    # Verify dimension
    actual_dim = embedder.get_actual_embedding_dimension()
    print(f"✅ Verified dimension: {actual_dim}")
