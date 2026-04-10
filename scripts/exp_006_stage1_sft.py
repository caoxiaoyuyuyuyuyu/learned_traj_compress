#!/usr/bin/env python3
"""
Exp 006 Stage 1 — Shared SFT Training (D019).

Merges SFT data from all N values, applies prompt-level train/val/test split,
trains a shared LoRA adapter on Qwen2.5-3B.

Split is deterministic (hash-based) so DPO script reproduces the same split.
"""

import argparse
import glob
import hashlib
import json
import os
import random
import re
import time

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from trl import SFTConfig, SFTTrainer


# ── Prompt-level split (shared with DPO script) ──────────────────────

def get_prompt_text(item):
    """Extract prompt text for split hashing.

    Pinned to `user_content` (raw trajectory source of truth). Fallback chain
    removed to prevent silent split drift if schema fields are added/renamed.
    """
    assert "user_content" in item, (
        f"missing 'user_content' key for SFT split hashing; "
        f"got keys: {list(item.keys())[:5]}"
    )
    return item["user_content"]


def assign_split(prompt_text, seed=42, test_frac=0.15, val_frac=0.10):
    """Deterministic hash-based split. Same prompt always maps to same split."""
    h = hashlib.sha256(f"{seed}|{prompt_text[:500]}".encode()).hexdigest()
    v = int(h[:8], 16) / 0xFFFFFFFF
    if v < test_frac:
        return "test"
    elif v < test_frac + val_frac:
        return "val"
    return "train"


def split_data(data, seed=42):
    """Split data list into train/val/test by prompt-level hashing."""
    splits = {"train": [], "val": [], "test": []}
    for item in data:
        prompt_text = get_prompt_text(item)
        s = assign_split(prompt_text, seed=seed)
        splits[s].append(item)
    return splits


# ── Data loading ─────────────────────────────────────────────────────

def truncate_after_last_answer(text):
    """Truncate text after the last </answer> tag to remove garbage."""
    match = list(re.finditer(r"</answer>", text))
    if match:
        return text[:match[-1].end()]
    return text


def load_and_merge_sft_data(data_dir, balanced=False, n2_target=50, seed=42):
    """Load and merge sft_data_N*.json files from data_dir.

    Each item is tagged with `_n_value` (int) so we can log per-N train loss.
    If balanced=True, down-sample N=2 items to ~n2_target (for the balanced
    SFT ablation — addresses reviewer Warning 2: N=2 dominates shared SFT).
    """
    pattern = os.path.join(data_dir, "sft_data_N*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No SFT data files found matching {pattern}")

    per_n_data = {}
    truncated_count = 0
    for f in files:
        m = re.search(r"sft_data_N(\d+)\.json$", os.path.basename(f))
        if not m:
            print(f"  [skip] {os.path.basename(f)}: cannot parse N value")
            continue
        n_val = int(m.group(1))
        with open(f) as fh:
            items = json.load(fh)
            for item in items:
                key = "assistant_content" if "assistant_content" in item else "response"
                if key in item:
                    original = item[key]
                    item[key] = truncate_after_last_answer(original)
                    if len(item[key]) < len(original):
                        truncated_count += 1
                item["_n_value"] = n_val
            print(f"  {os.path.basename(f)}: {len(items)} samples (N={n_val})")
            per_n_data[n_val] = items

    if balanced and 2 in per_n_data:
        orig = len(per_n_data[2])
        target = min(n2_target, orig)
        rng = random.Random(seed)
        per_n_data[2] = rng.sample(per_n_data[2], target)
        print(f"  [balanced_sft] N=2 down-sampled: {orig} -> {target} (seed={seed})")

    all_data = []
    for n_val in sorted(per_n_data.keys()):
        all_data.extend(per_n_data[n_val])

    per_n_counts = {n: len(items) for n, items in per_n_data.items()}
    print(f"Total merged SFT samples: {len(all_data)} ({truncated_count} truncated)")
    print(f"Per-N counts: {per_n_counts}")
    return all_data, per_n_counts


def format_for_sft(data, tokenizer, max_seq_length):
    """Pre-tokenize SFT data with per-sample n_value tag.

    Returns a Dataset with columns: input_ids, attention_mask, labels, n_value.
    We pre-tokenize here (instead of letting SFTTrainer handle it) so that the
    n_value column survives into compute_loss for per-N train loss logging.
    Paired with SFTConfig(dataset_kwargs={'skip_prepare_dataset': True}).
    """
    input_ids, attention_mask, labels, n_values = [], [], [], []
    skipped = 0
    for item in data:
        # Try multiple possible field names for user/assistant content
        user = item.get("user_content") or item.get("prompt") or item.get("input", "")
        assistant = item.get("assistant_content") or item.get("response") or item.get("output", "")

        if not user or not assistant:
            skipped += 1
            continue

        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        toks = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_attention_mask=True,
        )
        input_ids.append(toks["input_ids"])
        attention_mask.append(toks["attention_mask"])
        labels.append(list(toks["input_ids"]))
        n_values.append(int(item.get("_n_value", 0)))

    if skipped:
        print(f"WARNING: format_for_sft skipped {skipped} items with empty fields")

    return Dataset.from_dict({
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "n_value": n_values,
    })


# ── Custom collator & trainer for per-N train loss logging ───────────

class PerNDataCollator:
    """Pops `n_value` before standard LM collation, re-attaches as tensor."""

    def __init__(self, tokenizer):
        self.base = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    def __call__(self, features):
        n_values = [f.pop("n_value") for f in features]
        batch = self.base(features)
        batch["n_value"] = torch.tensor(n_values, dtype=torch.long)
        return batch


class PerNLossSFTTrainer(SFTTrainer):
    """SFTTrainer that logs per-N train loss in addition to total train loss.

    Pops `n_value` from the batch (so model forward doesn't see it), computes
    the standard LM loss, and bookkeeps per-N running means to emit at each
    `logging_step` via the `log` override.

    Note: with per_device_train_batch_size=1 each batch is a single sample,
    so attributing the batch-mean loss to its N is exact. For batch_size>1
    this is an approximation (uniform attribution of batch mean to each N
    present in the batch).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._per_n_loss_sum = {2: 0.0, 4: 0.0, 8: 0.0}
        self._per_n_loss_cnt = {2: 0, 4: 0, 8: 0}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        n_values = inputs.pop("n_value", None)
        try:
            result = super().compute_loss(
                model, inputs, return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
        except TypeError:
            # Older Trainer signature without num_items_in_batch
            result = super().compute_loss(model, inputs, return_outputs=return_outputs)
        loss = result[0] if return_outputs else result

        if n_values is not None and model.training:
            loss_f = float(loss.detach())
            if isinstance(n_values, torch.Tensor):
                n_values = n_values.tolist()
            for n in n_values:
                n = int(n)
                if n in self._per_n_loss_sum:
                    self._per_n_loss_sum[n] += loss_f
                    self._per_n_loss_cnt[n] += 1
        return result

    def log(self, logs, *args, **kwargs):
        # Only inject per-N losses on train log events (not eval logs)
        if any(k.startswith("loss") and not k.startswith("eval_") for k in logs):
            for n in (2, 4, 8):
                if self._per_n_loss_cnt[n] > 0:
                    logs[f"train/loss_N{n}"] = (
                        self._per_n_loss_sum[n] / self._per_n_loss_cnt[n]
                    )
                    logs[f"train/count_N{n}"] = self._per_n_loss_cnt[n]
                    self._per_n_loss_sum[n] = 0.0
                    self._per_n_loss_cnt[n] = 0
        return super().log(logs, *args, **kwargs)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Exp 006 Stage 1: Shared SFT Training")
    parser.add_argument("--model_path", default="/root/autodl-tmp/models/Qwen2.5-3B")
    parser.add_argument("--data_dir", default="/root/autodl-tmp/learned_traj_compress/artifacts/phase1d_v2_data")
    parser.add_argument("--output_dir", default="/root/autodl-tmp/learned_traj_compress/checkpoints/exp_006_sft_shared")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--wandb_project", default="learned_traj_compress")
    parser.add_argument("--run_name", default="exp_006_sft_shared")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=42,
                        help="Seed for prompt-level train/val/test split")
    parser.add_argument("--balanced_sft", action="store_true",
                        help="(Reviewer Warning 2 ablation) Down-sample N=2 SFT samples "
                             "to --balanced_n2_target so shared SFT is not dominated by "
                             "N=2 (~80%% natural). Default OFF → natural pooled SFT "
                             "(main paper result). When ON, output_dir and run_name get "
                             "'_balanced' suffix.")
    parser.add_argument("--balanced_n2_target", type=int, default=50,
                        help="Target N=2 sample count when --balanced_sft is set. "
                             "Default 50, approximating N=4 perfect count (~76).")
    args = parser.parse_args()

    # Auto-suffix output_dir / run_name for balanced ablation
    if args.balanced_sft:
        if not args.output_dir.rstrip("/").endswith("_balanced"):
            args.output_dir = args.output_dir.rstrip("/") + "_balanced"
        if not args.run_name.endswith("_balanced"):
            args.run_name = args.run_name + "_balanced"
        print(f"[balanced_sft] output_dir -> {args.output_dir}")
        print(f"[balanced_sft] run_name   -> {args.run_name}")

    t0 = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load & split data ──
    print("=== Loading SFT data ===")
    raw_data, per_n_counts = load_and_merge_sft_data(
        args.data_dir,
        balanced=args.balanced_sft,
        n2_target=args.balanced_n2_target,
        seed=args.seed,
    )

    splits = split_data(raw_data, seed=args.split_seed)
    print(f"\nPrompt-level split (seed={args.split_seed}):")
    print(f"  train: {len(splits['train'])} samples")
    print(f"  val:   {len(splits['val'])} samples")
    print(f"  test:  {len(splits['test'])} samples")

    # Save split info for DPO script and eval script
    split_info = {
        "seed": args.split_seed,
        "counts": {k: len(v) for k, v in splits.items()},
        "test_prompts": [get_prompt_text(item)[:500] for item in splits["test"]],
    }
    split_path = os.path.join(args.output_dir, "prompt_split_info.json")
    with open(split_path, "w") as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)
    print(f"Split info saved to: {split_path}")

    # Save test data for eval script
    test_data_path = os.path.join(args.output_dir, "test_data.json")
    with open(test_data_path, "w") as f:
        json.dump(splits["test"], f, indent=2, ensure_ascii=False)
    print(f"Test data ({len(splits['test'])} samples) saved to: {test_data_path}")

    # ── Load tokenizer & format datasets ──
    print(f"\n=== Loading model: {args.model_path} ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = format_for_sft(splits["train"], tokenizer, args.max_seq_length)
    val_dataset = format_for_sft(splits["val"], tokenizer, args.max_seq_length)
    print(f"Formatted train: {len(train_dataset)}, val: {len(val_dataset)}")

    # Per-N distribution sanity check (helps diagnose Warning 2 imbalance)
    def _per_n_breakdown(ds):
        out = {}
        for n in ds["n_value"]:
            out[int(n)] = out.get(int(n), 0) + 1
        return out
    print(f"  train per-N: {_per_n_breakdown(train_dataset)}")
    print(f"  val   per-N: {_per_n_breakdown(val_dataset)}")

    # ── Load model ──
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # ── LoRA config ──
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    # ── Training config ──
    # NOTE: dataset_kwargs={'skip_prepare_dataset': True} + remove_unused_columns=False
    # keeps the `n_value` column alive through the Trainer's data pipeline so the
    # custom PerNLossSFTTrainer can log per-N train losses (Warning 2 diagnostic).
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # NOTE: TRL >= 1.0 renamed `max_seq_length` -> `max_length`. We pre-tokenize
        # in format_for_sft() with truncation already applied, so this is mostly
        # belt-and-braces, but passing the wrong kwarg crashes SFTConfig.__init__.
        max_length=args.max_seq_length,
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        report_to="wandb",
        run_name=args.run_name,
        seed=args.seed,
        dataloader_pin_memory=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # ── Train ──
    print("\n=== Starting SFT Training ===")
    collator = PerNDataCollator(tokenizer)
    trainer = PerNLossSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
        data_collator=collator,
    )
    trainer.train()

    # Save LoRA adapter
    print(f"\nSaving LoRA adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save training metadata
    meta = {
        "experiment": "exp_006_stage1_sft",
        "base_model": args.model_path,
        "data_dir": args.data_dir,
        "split_seed": args.split_seed,
        "balanced_sft": args.balanced_sft,
        "balanced_n2_target": args.balanced_n2_target if args.balanced_sft else None,
        "per_n_counts_after_balance": per_n_counts,
        "train_per_n_counts": _per_n_breakdown(train_dataset),
        "val_per_n_counts": _per_n_breakdown(val_dataset),
        "num_train": len(train_dataset),
        "num_val": len(val_dataset),
        "num_test": len(splits["test"]),
        "num_epochs": args.num_epochs,
        "lr": args.lr,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "elapsed_seconds": time.time() - t0,
    }
    with open(os.path.join(args.output_dir, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nSFT training complete in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Adapter saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
