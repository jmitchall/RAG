"""
Example: AutoMerging Retriever Pattern for Hierarchical Nodes with LangChain Vector Stores

This demonstrates how to:
1. Create hierarchical nodes with parent-child relationships using LlamaIndex
2. Convert to LangChain format with metadata preserving relationships
3. Store in LangChain vector stores (FAISS, Chroma, Qdrant)
4. Implement auto-merging retrieval: fetch leaf nodes, merge siblings into parents

The pattern works with any LangChain-compatible vector store.
"""
from refection_logger import logger
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import BaseNode, NodeRelationship
from langchain.schema import Document as LangchainDocument

# LangChain Vector Stores
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_qdrant import QdrantVectorStore as Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Import embedding function (you'll need to adjust based on your setup)
from langchain_community.embeddings import HuggingFaceEmbeddings


class HierarchicalNodeStore:
    """
    Store for hierarchical nodes that preserves parent-child relationships.
    Works as a companion to LangChain vector stores for auto-merging retrieval.
    """

    def __init__(self):
        self.nodes_by_id: Dict[str, BaseNode] = {}
        self.parent_to_children: Dict[str, List[str]] = defaultdict(list)

    def add_nodes(self, nodes: List[BaseNode]):
        """Store all hierarchical nodes with their relationships."""
        for node in nodes:
            self.nodes_by_id[node.node_id] = node

            # Track parent-child relationships
            if hasattr(node, 'relationships') and NodeRelationship.PARENT in node.relationships:
                parent_id = node.relationships[NodeRelationship.PARENT].node_id
                self.parent_to_children[parent_id].append(node.node_id)

    def get_node(self, node_id: str) -> Optional[BaseNode]:
        """Get node by ID."""
        return self.nodes_by_id.get(node_id)

    def get_parent(self, node_id: str) -> Optional[BaseNode]:
        """Get parent node for a given node ID."""
        node = self.nodes_by_id.get(node_id)
        if node and hasattr(node, 'relationships') and NodeRelationship.PARENT in node.relationships:
            parent_id = node.relationships[NodeRelationship.PARENT].node_id
            return self.nodes_by_id.get(parent_id)
        return None


def convert_nodes_to_langchain_with_metadata(nodes: List[BaseNode]) -> Tuple[
    List[LangchainDocument], Dict[str, BaseNode]]:
    """
    Convert LlamaIndex nodes to LangChain documents while preserving hierarchy metadata.
    
    Returns:
        Tuple of (langchain_documents, node_store_dict)
    """
    langchain_docs = []
    node_store = {}

    for node in nodes:
        # Preserve original node for later retrieval
        node_store[node.node_id] = node

        # Build metadata with relationship info
        metadata = {
            'node_id': node.node_id,
            'text_length': len(node.text),
        }

        # Add parent relationship
        if hasattr(node, 'relationships') and NodeRelationship.PARENT in node.relationships:
            parent_rel = node.relationships[NodeRelationship.PARENT]
            metadata['parent_id'] = parent_rel.node_id
            metadata['is_leaf'] = True
        else:
            metadata['parent_id'] = None
            metadata['is_leaf'] = False

        # Copy over any existing metadata
        if hasattr(node, 'metadata') and node.metadata:
            for key, value in node.metadata.items():
                if key not in metadata:
                    metadata[key] = value

        langchain_docs.append(LangchainDocument(
            page_content=node.text,
            metadata=metadata
        ))

    return langchain_docs, node_store


def auto_merge_retrieved_nodes(
        retrieved_docs: List[LangchainDocument],
        node_store: HierarchicalNodeStore,
        merge_threshold: int = 2,
        verbose: bool = True
) -> List[LangchainDocument]:
    """
    Implement auto-merging logic: replace sibling leaf nodes with parent when threshold met.
    
    Args:
        retrieved_docs: Documents retrieved from vector store
        node_store: Store containing all hierarchical nodes
        merge_threshold: Minimum siblings to trigger merge (default: 2)
        verbose: logger.info merging decisions
        
    Returns:
        List of documents with merged parents where applicable
    """
    # Group retrieved docs by parent_id
    parent_groups = defaultdict(list)
    no_parent_docs = []

    for doc in retrieved_docs:
        parent_id = doc.metadata.get('parent_id')
        if parent_id:
            parent_groups[parent_id].append(doc)
        else:
            # Root nodes with no parent
            no_parent_docs.append(doc)

    merged_docs = []

    # Process each parent group
    for parent_id, sibling_docs in parent_groups.items():
        if len(sibling_docs) >= merge_threshold:
            # Fetch parent node
            parent_node = node_store.get_node(parent_id)

            if parent_node:
                if verbose:
                    logger.info(f"🔄 Merging {len(sibling_docs)} siblings into parent node")
                    logger.info(f"   Parent ID: {parent_id[:16]}...")
                    logger.info(f"   Sibling IDs: {[doc.metadata['node_id'][:8] for doc in sibling_docs]}")
                    logger.info(
                        f"   Combined text length: {sum(len(d.page_content) for d in sibling_docs)} → {len(parent_node.text)}")

                # Create LangChain doc from parent node
                parent_metadata = {
                    'node_id': parent_node.node_id,
                    'text_length': len(parent_node.text),
                    'is_merged': True,
                    'num_children_merged': len(sibling_docs),
                    'parent_id': None  # Parent nodes don't have parents (or have higher level)
                }

                if hasattr(parent_node, 'metadata') and parent_node.metadata:
                    parent_metadata.update(parent_node.metadata)

                merged_doc = LangchainDocument(
                    page_content=parent_node.text,
                    metadata=parent_metadata
                )
                merged_docs.append(merged_doc)
            else:
                # Parent not found, keep original siblings
                if verbose:
                    logger.info(f"⚠️  Parent {parent_id[:16]} not found, keeping {len(sibling_docs)} siblings")
                merged_docs.extend(sibling_docs)
        else:
            # Not enough siblings to merge
            merged_docs.extend(sibling_docs)

    # Add nodes without parents
    merged_docs.extend(no_parent_docs)

    return merged_docs


def create_hierarchical_nodes(directory_path: str) -> Tuple[List[BaseNode], List[BaseNode]]:
    """
    Create hierarchical nodes from documents.
    
    Returns:
        Tuple of (all_nodes, leaf_nodes_only)
    """
    logger.info("📚 Loading documents...")
    documents = SimpleDirectoryReader(input_dir=directory_path).load_data()
    logger.info(f"   Loaded {len(documents)} documents")

    logger.info("\n🌳 Creating hierarchical nodes...")
    # Create hierarchical parser: 2048 (parent) → 512 (middle) → 128 (leaf)
    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[2048, 512, 128],
        chunk_overlap=20
    )

    all_nodes = node_parser.get_nodes_from_documents(documents, show_progress=False)
    leaf_nodes = get_leaf_nodes(all_nodes)

    logger.info(f"   Created {len(all_nodes)} total nodes")
    logger.info(f"   └─ 🍃 {len(leaf_nodes)} leaf nodes (for retrieval)")
    logger.info(f"   └─ 🌲 {len(all_nodes) - len(leaf_nodes)} parent/middle nodes (for merging)")

    return all_nodes, leaf_nodes


# ============================================================================
# Example 1: FAISS Vector Store
# ============================================================================

def example_1_faiss_auto_merge(directory_path: str, embeddings):
    """
    Demonstrate auto-merging retrieval with FAISS vector store.
    """
    logger.info("\n" + "=" * 80)
    logger.info("📦 EXAMPLE 1: FAISS Vector Store with Auto-Merging Retrieval")
    logger.info("=" * 80)

    # Create hierarchical nodes
    all_nodes, leaf_nodes = create_hierarchical_nodes(directory_path)

    # Convert to LangChain format
    logger.info("\n🔄 Converting to LangChain format...")
    leaf_docs, node_store_dict = convert_nodes_to_langchain_with_metadata(leaf_nodes)
    logger.info(f"   Converted {len(leaf_docs)} leaf documents for indexing")

    # Create node store for all nodes (needed for merging)
    node_store = HierarchicalNodeStore()
    node_store.add_nodes(all_nodes)

    # Create FAISS vector store from leaf nodes only
    logger.info("\n📊 Building FAISS index from leaf nodes...")
    vectorstore = FAISS.from_documents(leaf_docs, embeddings)
    logger.info("   ✅ FAISS index created")

    # Query
    query = "What are the main concepts and techniques in machine learning?"
    logger.info(f"\n🔍 Query: '{query}'")

    # Standard retrieval (no merging)
    logger.info("\n--- Standard Retrieval (no merging) ---")
    standard_results = vectorstore.similarity_search(query, k=6)
    logger.info(f"Retrieved {len(standard_results)} leaf nodes:")
    for i, doc in enumerate(standard_results[:3], 1):
        logger.info(f"\n{i}. Text length: {len(doc.page_content)} chars")
        logger.info(f"   Node ID: {doc.metadata.get('node_id', 'N/A')[:16]}...")
        logger.info(
            f"   Parent ID: {doc.metadata.get('parent_id', 'N/A')[:16] if doc.metadata.get('parent_id') else 'None'}...")
        logger.info(f"   Preview: {doc.page_content[:120]}...")

    # Auto-merging retrieval
    logger.info("\n--- Auto-Merging Retrieval (merges siblings) ---")
    retrieved_docs = vectorstore.similarity_search(query, k=6)
    merged_results = auto_merge_retrieved_nodes(retrieved_docs, node_store, merge_threshold=2, verbose=True)

    logger.info(f"\n✅ After merging: {len(merged_results)} nodes")
    for i, doc in enumerate(merged_results[:3], 1):
        is_merged = doc.metadata.get('is_merged', False)
        logger.info(f"\n{i}. {'[MERGED PARENT]' if is_merged else '[LEAF NODE]'}")
        logger.info(f"   Text length: {len(doc.page_content)} chars")
        if is_merged:
            logger.info(f"   Merged from {doc.metadata.get('num_children_merged', 0)} children")
        logger.info(f"   Preview: {doc.page_content[:120]}...")

    return vectorstore, node_store


# ============================================================================
# Example 2: Chroma Vector Store
# ============================================================================

def example_2_chroma_auto_merge(directory_path: str, embeddings):
    """
    Demonstrate auto-merging retrieval with Chroma vector store.
    """
    logger.info("\n" + "=" * 80)
    logger.info("📦 EXAMPLE 2: Chroma Vector Store with Auto-Merging Retrieval")
    logger.info("=" * 80)

    # Create hierarchical nodes
    all_nodes, leaf_nodes = create_hierarchical_nodes(directory_path)

    # Convert to LangChain format
    logger.info("\n🔄 Converting to LangChain format...")
    leaf_docs, node_store_dict = convert_nodes_to_langchain_with_metadata(leaf_nodes)
    logger.info(f"   Converted {len(leaf_docs)} leaf documents for indexing")

    # Create node store for all nodes
    node_store = HierarchicalNodeStore()
    node_store.add_nodes(all_nodes)

    # Create Chroma vector store
    logger.info("\n📊 Building Chroma index from leaf nodes...")
    persist_directory = "/tmp/chroma_hierarchical_demo"

    # Clean up old data if exists
    import shutil
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    vectorstore = Chroma.from_documents(
        documents=leaf_docs,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="hierarchical_nodes"
    )
    logger.info("   ✅ Chroma index created")

    # Query
    query = "Explain supervised and unsupervised learning differences"
    logger.info(f"\n🔍 Query: '{query}'")

    # Standard retrieval
    logger.info("\n--- Standard Retrieval (no merging) ---")
    standard_results = vectorstore.similarity_search(query, k=6)
    logger.info(f"Retrieved {len(standard_results)} leaf nodes:")
    for i, doc in enumerate(standard_results[:3], 1):
        logger.info(f"\n{i}. Text length: {len(doc.page_content)} chars")
        logger.info(f"   Node ID: {doc.metadata.get('node_id', 'N/A')[:16]}...")
        logger.info(
            f"   Parent ID: {doc.metadata.get('parent_id', 'N/A')[:16] if doc.metadata.get('parent_id') else 'None'}...")
        logger.info(f"   Preview: {doc.page_content[:120]}...")

    # Auto-merging retrieval
    logger.info("\n--- Auto-Merging Retrieval (merges siblings) ---")
    retrieved_docs = vectorstore.similarity_search(query, k=6)
    merged_results = auto_merge_retrieved_nodes(retrieved_docs, node_store, merge_threshold=2, verbose=True)

    logger.info(f"\n✅ After merging: {len(merged_results)} nodes")
    for i, doc in enumerate(merged_results[:3], 1):
        is_merged = doc.metadata.get('is_merged', False)
        logger.info(f"\n{i}. {'[MERGED PARENT]' if is_merged else '[LEAF NODE]'}")
        logger.info(f"   Text length: {len(doc.page_content)} chars")
        if is_merged:
            logger.info(f"   Merged from {doc.metadata.get('num_children_merged', 0)} children")
        logger.info(f"   Preview: {doc.page_content[:120]}...")

    return vectorstore, node_store


# ============================================================================
# Example 3: Qdrant Vector Store
# ============================================================================

def example_3_qdrant_auto_merge(directory_path: str, embeddings):
    """
    Demonstrate auto-merging retrieval with Qdrant vector store.
    """
    logger.info("\n" + "=" * 80)
    logger.info("📦 EXAMPLE 3: Qdrant Vector Store with Auto-Merging Retrieval")
    logger.info("=" * 80)

    # Create hierarchical nodes
    all_nodes, leaf_nodes = create_hierarchical_nodes(directory_path)

    # Convert to LangChain format
    logger.info("\n🔄 Converting to LangChain format...")
    leaf_docs, node_store_dict = convert_nodes_to_langchain_with_metadata(leaf_nodes)
    logger.info(f"   Converted {len(leaf_docs)} leaf documents for indexing")

    # Create node store for all nodes
    node_store = HierarchicalNodeStore()
    node_store.add_nodes(all_nodes)

    # Setup Qdrant client (in-memory for demo)
    logger.info("\n📊 Building Qdrant index from leaf nodes...")
    collection_name = "hierarchical_nodes_demo"

    # Create Qdrant client (in-memory)
    qdrant_client = QdrantClient(":memory:")

    # Get embedding dimension
    sample_embedding = embeddings.embed_query("test")
    embedding_dim = len(sample_embedding)

    # Create collection
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
    )

    # Create Qdrant vector store
    vectorstore = Qdrant(
        client=qdrant_client,
        collection_name=collection_name,
        embeddings=embeddings
    )

    # Add documents
    vectorstore.add_documents(leaf_docs)
    logger.info("   ✅ Qdrant index created")

    # Query
    query = "What is deep learning and neural networks?"
    logger.info(f"\n🔍 Query: '{query}'")

    # Standard retrieval
    logger.info("\n--- Standard Retrieval (no merging) ---")
    standard_results = vectorstore.similarity_search(query, k=6)
    logger.info(f"Retrieved {len(standard_results)} leaf nodes:")
    for i, doc in enumerate(standard_results[:3], 1):
        logger.info(f"\n{i}. Text length: {len(doc.page_content)} chars")
        logger.info(f"   Node ID: {doc.metadata.get('node_id', 'N/A')[:16]}...")
        logger.info(
            f"   Parent ID: {doc.metadata.get('parent_id', 'N/A')[:16] if doc.metadata.get('parent_id') else 'None'}...")
        logger.info(f"   Preview: {doc.page_content[:120]}...")

    # Auto-merging retrieval
    logger.info("\n--- Auto-Merging Retrieval (merges siblings) ---")
    retrieved_docs = vectorstore.similarity_search(query, k=6)
    merged_results = auto_merge_retrieved_nodes(retrieved_docs, node_store, merge_threshold=2, verbose=True)

    logger.info(f"\n✅ After merging: {len(merged_results)} nodes")
    for i, doc in enumerate(merged_results[:3], 1):
        is_merged = doc.metadata.get('is_merged', False)
        logger.info(f"\n{i}. {'[MERGED PARENT]' if is_merged else '[LEAF NODE]'}")
        logger.info(f"   Text length: {len(doc.page_content)} chars")
        if is_merged:
            logger.info(f"   Merged from {doc.metadata.get('num_children_merged', 0)} children")
        logger.info(f"   Preview: {doc.page_content[:120]}...")

    return vectorstore, node_store


# ============================================================================
# Main Demonstration
# ============================================================================

def create_sample_data(directory_path: str):
    """Create sample document if it doesn't exist."""
    if os.path.exists(directory_path) and os.listdir(directory_path):
        return  # Data already exists

    logger.info(f"📁 Creating sample data in {directory_path}...")
    os.makedirs(directory_path, exist_ok=True)

    sample_file = os.path.join(directory_path, "ml_fundamentals.txt")
    with open(sample_file, "w") as f:
        f.write("""
Machine Learning Fundamentals

Machine learning is a subset of artificial intelligence that enables systems to learn from data.
There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.

Supervised Learning:
In supervised learning, the model is trained on labeled data. The algorithm learns to map inputs to outputs based on example pairs.
Common applications include classification tasks like spam detection and regression tasks like price prediction.
Popular algorithms include linear regression, decision trees, and neural networks.

Unsupervised Learning:
Unsupervised learning works with unlabeled data. The algorithm tries to find hidden patterns or structures.
Clustering and dimensionality reduction are common unsupervised techniques.
Examples include customer segmentation and anomaly detection.

Reinforcement Learning:
In reinforcement learning, an agent learns by interacting with an environment and receiving rewards or penalties.
This approach is used in game playing, robotics, and autonomous systems.
The agent learns an optimal policy through trial and error.

Deep Learning:
Deep learning uses neural networks with multiple layers to learn hierarchical representations.
Convolutional neural networks (CNNs) excel at image processing tasks.
Recurrent neural networks (RNNs) are designed for sequential data like text and time series.
Transformers have revolutionized natural language processing with models like BERT and GPT.

Applications:
Machine learning powers recommendation systems, voice assistants, and medical diagnosis tools.
It's used in computer vision for object detection and facial recognition.
Natural language processing enables chatbots, translation, and sentiment analysis.

Model Training:
The training process involves feeding data to the model and adjusting its parameters to minimize error.
Cross-validation helps prevent overfitting by testing the model on unseen data.
Hyperparameter tuning optimizes model performance by finding the best configuration.

Evaluation Metrics:
For classification: accuracy, precision, recall, F1-score, and ROC-AUC.
For regression: mean squared error, mean absolute error, and R-squared.
Choosing the right metric depends on the problem domain and business objectives.
""" * 3)  # Repeat to create longer document

    logger.info(f"   ✅ Created sample document: {sample_file}")


if __name__ == "__main__":
    # Setup
    directory_path = "/home/jmitchall/vllm-srv/data/"

    # Create sample data if needed
    create_sample_data(directory_path)

    # Initialize embeddings (using lightweight model for demo)
    logger.info("\n🔧 Initializing embeddings model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    logger.info("   ✅ Embeddings ready")

    # Run all examples
    try:
        # Example 1: FAISS
        faiss_store, faiss_node_store = example_1_faiss_auto_merge(directory_path, embeddings)

        # Example 2: Chroma
        chroma_store, chroma_node_store = example_2_chroma_auto_merge(directory_path, embeddings)

        # Example 3: Qdrant
        qdrant_store, qdrant_node_store = example_3_qdrant_auto_merge(directory_path, embeddings)

        logger.info("\n" + "=" * 80)
        logger.info("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info("\n💡 Key Takeaways:")
        logger.info("   1. Hierarchical nodes preserve parent-child relationships")
        logger.info("   2. Store only LEAF nodes in vector store for precise retrieval")
        logger.info("   3. Keep ALL nodes in separate store for merging capability")
        logger.info("   4. Auto-merge siblings into parents for broader context")
        logger.info("   5. Pattern works with any LangChain-compatible vector store")
        logger.info("\n🎯 This approach combines:")
        logger.info("   • Precision of small chunk retrieval")
        logger.info("   • Context of larger parent chunks")
        logger.info("   • Flexibility across different vector stores")

    except Exception as e:
        logger.info(f"\n❌ Error: {e}")
        import traceback

        traceback.logger.info_exc()
