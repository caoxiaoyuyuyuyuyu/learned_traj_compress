"""Phase 3a: EM evaluation and collapse curve analysis.

Evaluates models across objective counts to measure performance
degradation and identify collapse thresholds. Uses utils/metrics.py.
"""

from typing import Optional


def evaluate_model(
    model,
    dataset,
    n_objectives: int,
    n_samples: int = 200,
    temperature: float = 0.0,
    max_new_tokens: int = 2048,
) -> dict:
    """Evaluate model on examples with specified objective count.

    Args:
        model: Model to evaluate.
        dataset: Evaluation dataset.
        n_objectives: Number of objectives per example.
        n_samples: Number of examples to evaluate.
        temperature: Generation temperature (0 = greedy).
        max_new_tokens: Max generation length.

    Returns:
        Dict with em_mean, em_total, f1_mean, per_question_breakdown,
        extraction_failure_rate.
    """
    raise NotImplementedError


def compute_collapse_curve(
    results: dict[int, dict],
) -> dict:
    """Compute collapse curve from per-objective evaluation results.

    Args:
        results: Dict mapping n_objectives -> evaluation results dict
            from evaluate_model.

    Returns:
        Dict with collapse_curve (n_obj -> em_mean), collapse_threshold
        (n_obj where performance drops below 50% of n=1), area_under_curve,
        and degradation_rate.
    """
    raise NotImplementedError
