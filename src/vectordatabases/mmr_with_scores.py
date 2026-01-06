"""
Utility functions for getting similarity scores with MMR results.

MMR (Maximal Marginal Relevance) balances relevance with diversity but doesn't 
return similarity scores by default. This module provides utilities to combine both.
"""

from langchain.schema import Document
from typing import List, Tuple


def compare_similarity_vs_mmr(vectorstore, query: str, k: int = 5,
                              fetch_k: int = 20, lambda_mult: float = 0.5) -> Tuple[List[Document], List[Document]]:
    """
    Compare pure similarity search results vs MMR results side-by-side.
    
    This helps understand how MMR changes the results to add diversity.
    
    Args:
        vectorstore: Vector database instance (Chroma, FAISS, Qdrant)
        query: Search query
        k: Number of results to return
        fetch_k: For MMR - number of candidates to consider
        lambda_mult: For MMR - balance between relevance (1.0) and diversity (0.0)
        
    Returns:
        Tuple of (similarity_results, mmr_results) - both are lists of Documents
        
    Example:
        >>> sim_results, mmr_results = compare_similarity_vs_mmr(db, "What is a Rogue?")
        >>> print("SIMILARITY TOP-5:")
        >>> for i, doc in enumerate(sim_results):
        ...     score = doc.metadata.get('similarity_score', 'N/A')
        ...     print(f"  {i+1}. Score: {score:.4f} - {doc.page_content[:100]}")
        >>> 
        >>> print("\nMMR TOP-5 (diverse):")
        >>> for i, doc in enumerate(mmr_results):
        ...     print(f"  {i+1}. {doc.page_content[:100]}")
    """
    # Get pure similarity results with scores
    similarity_results = vectorstore.similarity_search_with_score(query, k=k)
    sim_docs = []
    for doc, distance in similarity_results:
        doc.metadata['distance'] = distance
        doc.metadata['similarity_score'] = -distance  # For distance metrics
        doc.metadata['rank'] = len(sim_docs) + 1
        sim_docs.append(doc)

    # Get MMR results (diverse)
    mmr_docs = vectorstore.max_marginal_relevance_search(
        query,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult
    )

    # Add rank to MMR results
    for i, doc in enumerate(mmr_docs):
        doc.metadata['mmr_rank'] = i + 1

    return sim_docs, mmr_docs


def print_comparison(sim_docs: List[Document], mmr_docs: List[Document]):
    """
    Pretty-print comparison between similarity and MMR results.
    
    Args:
        sim_docs: Results from similarity search
        mmr_docs: Results from MMR search
    """
    print("\n" + "=" * 80)
    print("SIMILARITY SEARCH (Pure Relevance)")
    print("=" * 80)
    for i, doc in enumerate(sim_docs, 1):
        score = doc.metadata.get('similarity_score', 'N/A')
        distance = doc.metadata.get('distance', 'N/A')
        source = doc.metadata.get('source', 'unknown')
        page = doc.metadata.get('page', 'N/A')

        print(f"\n{i}. Distance: {distance:.4f} | Score: {score:.4f}")
        print(f"   Source: {source} (page {page})")
        print(f"   Content: {doc.page_content[:150]}...")

    print("\n" + "=" * 80)
    print("MMR SEARCH (Relevance + Diversity)")
    print("=" * 80)
    for i, doc in enumerate(mmr_docs, 1):
        source = doc.metadata.get('source', 'unknown')
        page = doc.metadata.get('page', 'N/A')

        print(f"\n{i}. Rank #{i}")
        print(f"   Source: {source} (page {page})")
        print(f"   Content: {doc.page_content[:150]}...")

    # Show overlap analysis
    sim_contents = {doc.page_content for doc in sim_docs}
    mmr_contents = {doc.page_content for doc in mmr_docs}
    overlap = sim_contents & mmr_contents

    print("\n" + "=" * 80)
    print(f"ANALYSIS: {len(overlap)}/{len(sim_docs)} documents are the same")
    print(f"MMR introduced {len(mmr_docs) - len(overlap)} different documents for diversity")
    print("=" * 80)


def get_mmr_with_scores(vectorstore, query: str, k: int = 5,
                        fetch_k: int = 20, lambda_mult: float = 0.5) -> List[Tuple[Document, float]]:
    """
    Get MMR results WITH similarity scores by computing them post-hoc.
    
    Strategy:
    1. Get MMR results (ordered by relevance + diversity)
    2. Compute similarity score for each result
    3. Return docs with scores, maintaining MMR order
    
    Args:
        vectorstore: Vector database instance
        query: Search query
        k: Number of results
        fetch_k: MMR candidate pool size
        lambda_mult: MMR diversity parameter
        
    Returns:
        List of (Document, similarity_score) tuples in MMR order
        
    Note:
        - Order is still based on MMR (relevance + diversity)
        - Scores show how relevant each document is to the query
        - Lower-ranked docs may have higher scores (due to diversity factor)
    """
    import numpy as np

    # Get MMR results
    mmr_docs = vectorstore.max_marginal_relevance_search(
        query,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult
    )

    # Get query embedding
    if hasattr(vectorstore, '_embedding_function'):
        query_embedding = vectorstore._embedding_function.embed_query(query)
    elif hasattr(vectorstore, 'embeddings'):
        query_embedding = vectorstore.embeddings.embed_query(query)
    else:
        # Fallback: return docs without scores
        return [(doc, None) for doc in mmr_docs]

    results = []
    for doc in mmr_docs:
        # Compute similarity for this document
        # Note: This is a simplified approach - actual implementation depends on vectorstore
        try:
            # Get document embedding and compute similarity
            # This is vectorstore-specific and may need adjustment
            doc_text = doc.page_content

            if hasattr(vectorstore, '_embedding_function'):
                doc_embedding = vectorstore._embedding_function.embed_query(doc_text)
            else:
                doc_embedding = vectorstore.embeddings.embed_query(doc_text)

            # Compute cosine similarity
            query_norm = np.linalg.norm(query_embedding)
            doc_norm = np.linalg.norm(doc_embedding)
            similarity = np.dot(query_embedding, doc_embedding) / (query_norm * doc_norm + 1e-10)

            results.append((doc, float(similarity)))
        except Exception:
            results.append((doc, None))

    return results


# Example usage
if __name__ == "__main__":
    print("""
MMR WITH SCORES - USAGE EXAMPLES
=================================

1. COMPARE SIMILARITY VS MMR:
   ```python
   from vectordatabases.mmr_with_scores import compare_similarity_vs_mmr, print_comparison
   
   sim_docs, mmr_docs = compare_similarity_vs_mmr(
       vectorstore, 
       "What is a Rogue?",
       k=5,
       lambda_mult=0.5
   )
   
   print_comparison(sim_docs, mmr_docs)
   ```

2. GET MMR WITH SCORES:
   ```python
   from vectordatabases.mmr_with_scores import get_mmr_with_scores
   
   results = get_mmr_with_scores(
       vectorstore,
       "What is a Rogue?",
       k=5,
       lambda_mult=0.5
   )
   
   for i, (doc, score) in enumerate(results, 1):
       print(f"{i}. Similarity: {score:.4f} - {doc.page_content[:100]}")
   ```

3. UNDERSTANDING MMR ORDER:
   - MMR rank #1 = Most relevant AND considers diversity
   - Higher MMR ranks may have HIGHER similarity scores but were deprioritized for diversity
   - lambda_mult controls this:
     * 1.0 = pure relevance (same as similarity search)
     * 0.5 = balanced
     * 0.0 = maximum diversity (may sacrifice relevance)

4. WHEN TO USE WHICH:
   - Pure Similarity: When you want the absolute most relevant docs
   - MMR with lambda=0.8: When you want mostly relevance with slight diversity
   - MMR with lambda=0.5: When documents have overlapping content
   - MMR with lambda=0.2: When you want maximum variety (summarization)
    """)
