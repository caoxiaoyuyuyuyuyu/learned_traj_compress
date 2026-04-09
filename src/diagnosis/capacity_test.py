"""Phase 1d: Model capacity experiments.

Tests how model size interacts with objective count to produce collapse,
identifying the capacity threshold for successful multi-objective distillation.
"""

from typing import Callable, Optional


def run_capacity_experiment(
    model_sizes: list[str],
    n_objectives_list: list[int],
    train_fn: Callable,
    eval_fn: Callable,
    config: Optional[dict] = None,
) -> dict:
    """Run capacity scaling experiment across model sizes and objective counts.

    For each (model_size, n_objectives) pair, trains via SFT and evaluates,
    producing a 2D matrix of performance to identify the collapse boundary.

    Args:
        model_sizes: List of model identifiers (e.g., ["Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen2.5-3B"]).
        n_objectives_list: List of objective counts to test (e.g., [1, 2, 4, 8, 16]).
        train_fn: Training function with signature (model_name, dataset, config) -> model.
        eval_fn: Evaluation function with signature (model, dataset) -> metrics dict.
        config: Additional training/eval configuration.

    Returns:
        Dict with results_matrix (model_size x n_obj), collapse_thresholds
        per model, and scaling analysis.
    """
    raise NotImplementedError
