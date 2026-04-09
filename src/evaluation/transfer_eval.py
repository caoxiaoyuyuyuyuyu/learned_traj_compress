"""Phase 3b: Transfer evaluation on SWE-Smith and other benchmarks.

Tests whether MemDistill's improvements generalize beyond MEM1
to other multi-step agent tasks.
"""

from typing import Optional


def load_swe_smith_data(
    split: str = "test",
    max_samples: Optional[int] = None,
) -> list[dict]:
    """Load SWE-Smith benchmark data.

    Args:
        split: Dataset split ("train", "test").
        max_samples: Cap on number of samples.

    Returns:
        List of evaluation examples.
    """
    raise NotImplementedError


def evaluate_transfer(
    model,
    dataset: list[dict],
    n_samples: Optional[int] = None,
    max_new_tokens: int = 4096,
) -> dict:
    """Evaluate model on transfer benchmark.

    Args:
        model: Model to evaluate.
        dataset: Transfer evaluation examples.
        n_samples: Number of examples. None = all.
        max_new_tokens: Max generation length.

    Returns:
        Dict with pass_rate, per_task_results, comparison_with_baseline.
    """
    raise NotImplementedError
