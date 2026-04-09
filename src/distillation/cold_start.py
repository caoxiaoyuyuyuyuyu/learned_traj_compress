"""Phase 2c: Cold-start training on low-objective examples.

Initializes the student model on easy (few-objective) examples before
progressing to harder multi-objective tasks via curriculum.
"""

from typing import Optional


def create_cold_start_dataset(
    dataset,
    max_objectives: int = 2,
    max_samples: Optional[int] = None,
) -> list[dict]:
    """Create cold-start dataset with low objective count examples.

    Args:
        dataset: Full training dataset.
        max_objectives: Maximum number of objectives to include.
        max_samples: Cap on dataset size. None = all matching.

    Returns:
        List of training examples filtered to <= max_objectives.
    """
    raise NotImplementedError


def train_cold_start(
    model,
    dataset: list[dict],
    output_dir: str = "checkpoints/cold_start",
    n_epochs: int = 3,
    lr: float = 2e-5,
    batch_size: int = 4,
    warmup_ratio: float = 0.1,
) -> dict:
    """Train model on cold-start dataset.

    Args:
        model: Student model to train.
        dataset: Cold-start training examples.
        output_dir: Where to save checkpoints.
        n_epochs: Number of training epochs.
        lr: Learning rate.
        batch_size: Training batch size.
        warmup_ratio: Fraction of steps for LR warmup.

    Returns:
        Dict with final_loss, training_history, checkpoint_path.
    """
    raise NotImplementedError
