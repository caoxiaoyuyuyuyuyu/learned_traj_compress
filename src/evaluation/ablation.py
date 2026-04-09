"""Phase 3c: Ablation studies.

Systematically evaluates the contribution of each MemDistill component
by removing or modifying them individually.
"""

from typing import Optional


def run_ablation_suite(
    base_config: dict,
    ablation_configs: list[dict],
    train_fn=None,
    eval_fn=None,
) -> dict:
    """Run a suite of ablation experiments.

    Each ablation config modifies one aspect of the base config.
    Trains and evaluates each variant, comparing against the full system.

    Args:
        base_config: Full MemDistill training configuration.
        ablation_configs: List of dicts, each with keys:
            - name: Ablation name (e.g., "no_curriculum", "no_teacher_filter").
            - overrides: Dict of config overrides to apply.
        train_fn: Training function. If None, uses default pipeline.
        eval_fn: Evaluation function. If None, uses default eval.

    Returns:
        Dict with per_ablation results, delta_from_base for each metric,
        and statistical significance tests.
    """
    raise NotImplementedError
