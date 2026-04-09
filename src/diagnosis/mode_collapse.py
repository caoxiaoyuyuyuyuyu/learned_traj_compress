"""Phase 1b: Mode collapse detection in SFT-distilled outputs.

Measures diversity loss via n-gram statistics, embedding variance,
and strategy clustering.
"""

import numpy as np
from typing import Optional


def compute_distinct_ngrams(
    outputs: list[str],
    n: int = 4,
) -> dict:
    """Compute distinct-N metrics for a set of model outputs.

    Args:
        outputs: List of generated text outputs.
        n: N-gram order (1, 2, 3, or 4).

    Returns:
        Dict with distinct_n ratio, total_ngrams, unique_ngrams,
        and per-output stats.
    """
    raise NotImplementedError


def compute_embedding_variance(
    outputs: list[str],
    model,
    tokenizer=None,
    batch_size: int = 16,
) -> dict:
    """Compute variance of output embeddings to measure diversity.

    Args:
        outputs: List of generated text outputs.
        model: Model to compute embeddings (uses last hidden state mean pool).
        tokenizer: Tokenizer for the model. If None, uses model's tokenizer.
        batch_size: Batch size for encoding.

    Returns:
        Dict with mean_variance, per_dim_variance, effective_rank of
        covariance matrix.
    """
    raise NotImplementedError


def cluster_strategies(
    outputs: list[str],
    n_clusters: int = 5,
    embedding_model=None,
) -> dict:
    """Cluster outputs to identify distinct strategies.

    Args:
        outputs: List of generated text outputs.
        n_clusters: Number of clusters for k-means.
        embedding_model: Model for computing embeddings. If None, uses TF-IDF.

    Returns:
        Dict with cluster_labels, cluster_sizes, silhouette_score,
        cluster_exemplars.
    """
    raise NotImplementedError
