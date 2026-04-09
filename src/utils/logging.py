"""Logging and experiment tracking utilities for MemDistill."""

from typing import Optional


def setup_wandb(
    project: str = "memdistill",
    entity: Optional[str] = None,
    config: Optional[dict] = None,
    run_name: Optional[str] = None,
    tags: Optional[list[str]] = None,
):
    """Initialize Weights & Biases run.

    Args:
        project: W&B project name.
        entity: W&B entity/team.
        config: Experiment configuration to log.
        run_name: Custom run name. None = auto-generated.
        tags: Tags for the run.

    Returns:
        W&B run object.
    """
    raise NotImplementedError


def log_metrics(
    metrics: dict,
    step: Optional[int] = None,
    prefix: Optional[str] = None,
):
    """Log metrics to W&B.

    Args:
        metrics: Dict of metric name -> value.
        step: Training step. None = auto-increment.
        prefix: Prefix for metric names (e.g., "eval/", "train/").
    """
    raise NotImplementedError
