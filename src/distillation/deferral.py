"""Phase 2e: Deferral mechanism — route hard examples to teacher.

Enables the student model to recognize when it cannot handle a query
and defer to the larger teacher model, trading latency for accuracy.
"""

import torch
from typing import Optional


def compute_confidence(
    model,
    inputs: list[str],
    method: str = "entropy",
    batch_size: int = 8,
) -> list[float]:
    """Compute confidence scores for model predictions.

    Args:
        model: Student model.
        inputs: List of input prompts.
        method: Confidence estimation method.
            "entropy": Negative mean token entropy.
            "perplexity": Negative log perplexity.
            "max_prob": Mean max token probability.
        batch_size: Batch size for inference.

    Returns:
        List of confidence scores (higher = more confident).
    """
    raise NotImplementedError


def should_defer(
    confidence: float,
    threshold: float,
) -> bool:
    """Decide whether to defer to teacher based on confidence.

    Args:
        confidence: Student's confidence score.
        threshold: Deferral threshold (defer if confidence < threshold).

    Returns:
        True if should defer to teacher.
    """
    raise NotImplementedError


class DeferralWrapper:
    """Wraps student + teacher with automatic deferral routing.

    Usage:
        wrapper = DeferralWrapper(student, teacher, threshold=0.5)
        output = wrapper.generate(prompt)
        # output.source indicates "student" or "teacher"
    """

    def __init__(
        self,
        student_model,
        teacher_model,
        threshold: float = 0.5,
        confidence_method: str = "entropy",
    ):
        """Initialize deferral wrapper.

        Args:
            student_model: Small sidecar model.
            teacher_model: Large teacher model.
            threshold: Confidence threshold for deferral.
            confidence_method: Method for confidence estimation.
        """
        raise NotImplementedError

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
    ) -> dict:
        """Generate with automatic deferral.

        Args:
            prompt: Input prompt.
            max_new_tokens: Max generation length.

        Returns:
            Dict with keys: output, source ("student"/"teacher"),
            confidence, deferred.
        """
        raise NotImplementedError
