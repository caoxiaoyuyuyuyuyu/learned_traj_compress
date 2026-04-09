"""Phase 2d: Curriculum learning — progressive objective scaling.

Gradually increases the number of objectives during training,
allowing the student model to build capacity incrementally.
"""

from typing import Optional


def create_curriculum_schedule(
    min_obj: int = 1,
    max_obj: int = 16,
    total_steps: int = 10000,
    strategy: str = "linear",
) -> list[dict]:
    """Create a curriculum schedule for objective count progression.

    Args:
        min_obj: Starting number of objectives.
        max_obj: Final number of objectives.
        total_steps: Total training steps.
        strategy: Progression strategy.
            "linear": Linear increase over training.
            "exponential": Exponential increase (more time on easy).
            "step": Discrete jumps at fixed intervals.

    Returns:
        List of dicts with keys: step, n_objectives, describing the
        curriculum at each transition point.
    """
    raise NotImplementedError


def get_curriculum_data(
    dataset,
    schedule: list[dict],
    step: int,
    batch_size: int = 8,
) -> list[dict]:
    """Get training batch matching current curriculum stage.

    Args:
        dataset: Full training dataset with variable objective counts.
        schedule: Curriculum schedule from create_curriculum_schedule.
        step: Current training step.
        batch_size: Number of examples to return.

    Returns:
        List of training examples with n_objectives matching current
        curriculum stage.
    """
    raise NotImplementedError
