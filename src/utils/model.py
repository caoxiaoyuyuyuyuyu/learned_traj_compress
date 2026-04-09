"""Model loading utilities for MemDistill.

Teacher model is MEM1 7B RL policy = Qwen2.5-7B base + LoRA adapter.
Loading options:
  1. Load base + adapter, merge at runtime (flexible, slower startup)
  2. Load pre-merged checkpoint (faster startup, needs disk space)
"""

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def load_teacher_model(
    base_path: str = "Qwen/Qwen2.5-7B",
    adapter_path: str = "Mem-Lab/Qwen2.5-7B-RL-RAG-Q2-EM-Release",
    merged_path: Optional[str] = None,
    device: str = "auto",
    dtype: str = "bfloat16",
):
    """Load MEM1 7B RL teacher model (base + LoRA adapter).

    If merged_path is provided, loads pre-merged model directly.
    Otherwise loads base model and merges adapter at runtime.

    Args:
        base_path: Path to Qwen2.5-7B base model.
        adapter_path: Path to MEM1 LoRA adapter.
        merged_path: Path to pre-merged model (skips runtime merge).
        device: Device placement ("auto", "cuda", "cpu").
        dtype: Model dtype.

    Returns:
        Merged model ready for inference.
    """
    torch_dtype = DTYPE_MAP[dtype]

    if merged_path is not None:
        return AutoModelForCausalLM.from_pretrained(
            merged_path, torch_dtype=torch_dtype, device_map=device
        )

    base = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=torch_dtype, device_map=device
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model = model.merge_and_unload()
    return model


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


def merge_and_save(
    base_path: str,
    adapter_path: str,
    output_path: str,
    dtype: str = "bfloat16",
):
    """Merge LoRA adapter into base model and save to disk.

    Use this to create a pre-merged checkpoint for faster loading
    during Phase 2 Best-of-N sampling.

    Args:
        base_path: Path to base model.
        adapter_path: Path to LoRA adapter.
        output_path: Where to save merged model.
        dtype: Model dtype for saving.
    """
    torch_dtype = DTYPE_MAP[dtype]
    base = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=torch_dtype, device_map="cpu"
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model = model.merge_and_unload()
    model.save_pretrained(output_path)
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    tokenizer.save_pretrained(output_path)
