"""Model loading utilities for MemDistill."""

from typing import Optional


def load_teacher_model(
    model_name: str = "Qwen/Qwen2.5-3B",
    device: str = "auto",
    dtype: str = "bfloat16",
):
    """Load teacher model for distillation.

    Args:
        model_name: HuggingFace model identifier.
        device: Device placement ("auto", "cuda", "cpu").
        dtype: Model dtype ("bfloat16", "float16", "float32").

    Returns:
        Loaded model ready for inference.
    """
    raise NotImplementedError


def load_student_model(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    device: str = "auto",
    dtype: str = "bfloat16",
):
    """Load student model for training.

    Args:
        model_name: HuggingFace model identifier.
        device: Device placement.
        dtype: Model dtype.

    Returns:
        Loaded model ready for training.
    """
    raise NotImplementedError


def load_tokenizer(
    model_name: str,
    padding_side: str = "left",
):
    """Load tokenizer with appropriate settings.

    Args:
        model_name: HuggingFace model identifier.
        padding_side: Padding side for batch generation.

    Returns:
        Configured tokenizer.
    """
    raise NotImplementedError


def merge_lora_adapter(
    base_path: str,
    adapter_path: str,
    output_path: Optional[str] = None,
):
    """Merge LoRA adapter into base model.

    Args:
        base_path: Path to base model.
        adapter_path: Path to LoRA adapter checkpoint.
        output_path: Where to save merged model. None = return without saving.

    Returns:
        Merged model.
    """
    raise NotImplementedError
