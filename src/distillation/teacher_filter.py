"""Phase 2b: Teacher-likelihood filtering.

Filters RL-generated candidates by teacher model likelihood to remove
outputs that the student model's capacity cannot support.
"""

import torch
from typing import Optional


def compute_teacher_loglikelihood(
    teacher,
    candidates: list[str],
    prompts: list[str],
    batch_size: int = 8,
) -> list[float]:
    """Compute teacher model log-likelihood for each candidate.

    Args:
        teacher: Teacher model for scoring.
        candidates: List of candidate outputs.
        prompts: Corresponding input prompts.
        batch_size: Batch size for scoring.

    Returns:
        List of log-likelihood scores (one per candidate).
    """
    raise NotImplementedError


def filter_by_likelihood(
    candidates: list[dict],
    threshold: Optional[float] = None,
    percentile: float = 50.0,
) -> list[dict]:
    """Filter candidates by teacher log-likelihood.

    Args:
        candidates: List of candidate dicts with 'text' and 'log_likelihood' keys.
        threshold: Absolute log-likelihood threshold. If None, uses percentile.
        percentile: Keep candidates above this percentile of log-likelihood.

    Returns:
        Filtered list of candidates.
    """
    raise NotImplementedError
