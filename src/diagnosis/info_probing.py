"""Phase 1c: Information probing — what knowledge survives SFT collapse.

Trains linear probes on frozen representations to measure how much
task-relevant information is retained vs lost during SFT distillation.
"""

import torch
import numpy as np
from typing import Optional


def train_linear_probe(
    representations: torch.Tensor,
    labels: torch.Tensor,
    n_epochs: int = 100,
    lr: float = 1e-3,
    val_split: float = 0.2,
) -> dict:
    """Train a linear probe on frozen representations.

    Args:
        representations: Input features, shape (n_samples, hidden_dim).
        labels: Target labels, shape (n_samples,).
        n_epochs: Training epochs.
        lr: Learning rate.
        val_split: Fraction of data for validation.

    Returns:
        Dict with trained probe, train_acc, val_acc, training_history.
    """
    raise NotImplementedError


def evaluate_probe(
    probe,
    representations: torch.Tensor,
    labels: torch.Tensor,
) -> dict:
    """Evaluate a trained probe on test data.

    Args:
        probe: Trained linear probe.
        representations: Test features, shape (n_samples, hidden_dim).
        labels: Test labels, shape (n_samples,).

    Returns:
        Dict with accuracy, per_class_accuracy, confusion_matrix.
    """
    raise NotImplementedError


def probe_per_objective(
    model,
    dataset,
    n_objectives: list[int],
    layer_ids: Optional[list[int]] = None,
) -> dict:
    """Run probing analysis across different objective counts.

    For each n_obj, trains probes to predict which question is being
    answered, measuring how well the model's representations separate
    different sub-tasks.

    Args:
        model: Model to probe.
        dataset: Dataset with multi-objective examples.
        n_objectives: List of objective counts to test.
        layer_ids: Layers to probe. None = last layer only.

    Returns:
        Dict mapping n_obj -> layer_id -> probe results.
    """
    raise NotImplementedError
