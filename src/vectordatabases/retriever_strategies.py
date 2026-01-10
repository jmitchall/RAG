"""
Advanced retrieval strategies for better contextual document search.

This module provides configuration and utilities for various retrieval strategies:
- Similarity search: Pure cosine similarity (default)
- MMR (Maximal Marginal Relevance): Balances relevance with diversity
- Score threshold filtering: Only return high-confidence results
- Hybrid approaches: Combine multiple strategies
"""

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class RetrievalConfig:
    """Configuration for retrieval strategies."""

    # Basic parameters
    k: int = 5  # Number of documents to return

    # Search strategy
    search_type: Literal["similarity", "mmr"] = "similarity"
    """
    - "similarity": Pure relevance-based search (fastest)
    - "mmr": Maximal Marginal Relevance - balances relevance with diversity
    """

    # Score filtering
    score_threshold: Optional[float] = None
    """
    Minimum similarity score (0-1 for Qdrant/Chroma, None for FAISS).
    Only documents above this threshold will be returned.
    Recommended: 0.7-0.8 for high confidence results
    """

    # MMR-specific parameters
    fetch_k: int = 20
    """For MMR: Number of candidates to fetch before diversity filtering"""

    lambda_mult: float = 0.5
    """
    For MMR: Balance between relevance and diversity
    - 1.0 = maximum relevance (similar to pure similarity search)
    - 0.5 = balanced (default)
    - 0.0 = maximum diversity (may sacrifice relevance)
    """

    def __str__(self) -> str:
        parts = [f"k={self.k}", f"search={self.search_type}"]
        if self.score_threshold:
            parts.append(f"threshold={self.score_threshold}")
        if self.search_type == "mmr":
            parts.append(f"fetch_k={self.fetch_k}")
            parts.append(f"λ={self.lambda_mult}")
        return f"RetrievalConfig({', '.join(parts)})"


# Pre-configured strategies
DEFAULT_CONFIG = RetrievalConfig()

HIGH_PRECISION_CONFIG = RetrievalConfig(
    k=3,
    search_type="similarity",
    score_threshold=0.8,
)
"""High precision: Fewer results, higher confidence"""

DIVERSE_CONFIG = RetrievalConfig(
    k=5,
    search_type="mmr",
    fetch_k=20,
    lambda_mult=0.5,
)
"""Diverse results: Avoid redundant information"""

COMPREHENSIVE_CONFIG = RetrievalConfig(
    k=10,
    search_type="similarity",
    score_threshold=0.6,
)
"""Comprehensive search: More results with moderate filtering"""


def get_retrieval_strategy(strategy_name: str) -> RetrievalConfig:
    """
    Get a pre-configured retrieval strategy.
    
    Args:
        strategy_name: One of "default", "high_precision", "diverse", "comprehensive"
    
    Returns:
        RetrievalConfig object
    
    Example:
        >>> config = get_retrieval_strategy("high_precision")
        >>> print(config)
        RetrievalConfig(k=3, search=similarity, threshold=0.8)
    """
    strategies = {
        "default": DEFAULT_CONFIG,
        "high_precision": HIGH_PRECISION_CONFIG,
        "diverse": DIVERSE_CONFIG,
        "comprehensive": COMPREHENSIVE_CONFIG,
    }

    if strategy_name not in strategies:
        raise ValueError(
            f"Unknown strategy '{strategy_name}'. "
            f"Available strategies: {list(strategies.keys())}"
        )

    return strategies[strategy_name]


# Documentation for users
STRATEGY_GUIDE = """
RETRIEVAL STRATEGY GUIDE
========================

1. DEFAULT (similarity, k=5)
   - Best for: General-purpose retrieval
   - Speed: Fastest
   - Use when: You want straightforward relevant results

2. HIGH_PRECISION (similarity, k=3, threshold=0.8)
   - Best for: When accuracy is critical
   - Speed: Fast
   - Use when: You only want highly confident matches
   - Trade-off: May return fewer (or zero) results

3. DIVERSE (mmr, k=5, λ=0.5)
   - Best for: Avoiding redundant information
   - Speed: Slower (fetches more candidates)
   - Use when: Documents have overlapping content
   - Example: Multiple documents about the same topic

4. COMPREHENSIVE (similarity, k=10, threshold=0.6)
   - Best for: Exploratory search, broad context
   - Speed: Moderate
   - Use when: You want more context, willing to filter manually
   - Trade-off: May include less relevant results

CHOOSING THE RIGHT STRATEGY:
- Q&A systems → HIGH_PRECISION
- Summarization → DIVERSE
- Research/exploration → COMPREHENSIVE
- Production default → DEFAULT
"""
