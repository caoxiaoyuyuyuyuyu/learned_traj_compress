"""Model loading utilities for MemDistill.

Teacher model is MEM1 7B RL policy (full model, NOT LoRA adapter).
Stored as float32 on disk (~30GB), loaded as bfloat16 to save VRAM.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def load_teacher_model(
    model_path: str = "/root/autodl-tmp/models/Qwen2.5-7B-RL-RAG-Q2-EM-Release",
    device: str = "auto",
    dtype: str = "bfloat16",
):
    """Load MEM1 7B RL teacher model (complete model, D004).

    Args:
        model_path: Path to complete MEM1 7B model.
        device: Device placement ("auto", "cuda", "cpu").
        dtype: Model dtype (bf16 recommended; on-disk is float32).

    Returns:
        Model ready for inference.
    """
    torch_dtype = DTYPE_MAP[dtype]
    return AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch_dtype, device_map=device
    )


def load_student_model(
    model_name: str = "Qwen/Qwen2.5-1.5B",
    device: str = "auto",
    dtype: str = "bfloat16",
):
    """Load student (<3B) model for distillation training.

    Args:
        model_name: HuggingFace model identifier (Qwen2.5-0.5B/1.5B/3B).
        device: Device placement.
        dtype: Model dtype.

    Returns:
        Loaded model ready for training.
    """
    torch_dtype = DTYPE_MAP[dtype]
    return AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map=device
    )


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
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = padding_side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


