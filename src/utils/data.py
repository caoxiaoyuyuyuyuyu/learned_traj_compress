"""Data loading and preprocessing utilities for MemDistill."""

from typing import Optional, Callable


def load_mem1_dataset(
    split: str = "train",
    n_objectives: Optional[int] = None,
    max_samples: Optional[int] = None,
    data_dir: Optional[str] = None,
) -> list[dict]:
    """Load MEM1 dataset with optional objective count filtering.

    Args:
        split: Dataset split ("train", "val", "test").
        n_objectives: Filter to examples with this many objectives. None = all.
        max_samples: Cap on number of samples.
        data_dir: Override data directory path.

    Returns:
        List of dicts with keys: prompt, targets (list of list of answers),
        n_objectives, episode_id.
    """
    raise NotImplementedError


def create_bon_pairs(
    episodes: list[dict],
    n: int = 8,
    reward_fn: Optional[Callable] = None,
) -> list[dict]:
    """Create Best-of-N preference pairs from episode data.

    Args:
        episodes: List of episode dicts with 'candidates' and 'targets' keys.
        n: Number of candidates per episode.
        reward_fn: Custom reward function. Default uses compute_em_reward
            from utils/metrics.py.

    Returns:
        List of preference pair dicts with keys: prompt, chosen, rejected,
        chosen_reward, rejected_reward.
    """
    raise NotImplementedError
