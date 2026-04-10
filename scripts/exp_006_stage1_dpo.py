#!/usr/bin/env python3
"""
Exp 006 Stage 1 — Per-N DPO Training (D019).

Loads the shared SFT LoRA adapter, merges into base, then trains a new LoRA
with DPO on a specific N's preference data. Uses the same prompt-level split
as SFT (deterministic hash-based).
"""

import argparse
import json
import os
import random
import re
import sys
import time

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

# Canonical split from shared module (D024: full text hash + test_frac=0.25)
sys.path.insert(0, os.path.dirname(__file__))
from utils.split import assign_split, split_data, verify_split_counts


def get_prompt_text(item):
    """Extract prompt text for split hashing.

    DPO data items have a `prompt` key whose value is the raw `user_content`
    text (see scripts/phase1d_generate_data.py:288 — `"prompt": item["user_content"]`).
    """
    assert "prompt" in item, (
        f"missing 'prompt' key for DPO split hashing; "
        f"got keys: {list(item.keys())[:5]}"
    )
    return item["prompt"]


# ── Data loading ─────────────────────────────────────────────────────

def truncate_after_last_answer(text):
    """Truncate text after the last </answer> tag to remove garbage."""
    match = list(re.finditer(r"</answer>", text))
    if match:
        return text[:match[-1].end()]
    return text


def load_dpo_data(data_dir, n_objectives):
    """Load DPO data for a specific N, truncating chosen/rejected after last </answer>."""
    path = os.path.join(data_dir, f"dpo_data_N{n_objectives}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"DPO data not found: {path}")

    with open(path) as f:
        data = json.load(f)

    truncated_count = 0
    for item in data:
        for key in ("chosen", "rejected"):
            if key in item:
                original = item[key]
                item[key] = truncate_after_last_answer(original)
                if len(item[key]) < len(original):
                    truncated_count += 1

    print(f"Loaded {len(data)} DPO pairs from {os.path.basename(path)} ({truncated_count} fields truncated)")
    return data


def count_dpo_pairs_all_n(data_dir, n_list=(2, 4, 8)):
    """Count DPO pairs per N across data_dir (for matched-sample mode)."""
    counts = {}
    for n in n_list:
        p = os.path.join(data_dir, f"dpo_data_N{n}.json")
        if os.path.exists(p):
            with open(p) as fh:
                counts[n] = len(json.load(fh))
    return counts


def format_for_dpo(data, tokenizer):
    """Convert raw DPO data to TRL DPOTrainer format."""
    prompts, chosens, rejecteds = [], [], []
    for item in data:
        prompt_msgs = [{"role": "user", "content": item["prompt"]}]
        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )

        chosen_msgs = [
            {"role": "user", "content": item["prompt"]},
            {"role": "assistant", "content": item["chosen"]},
        ]
        chosen_text = tokenizer.apply_chat_template(
            chosen_msgs, tokenize=False, add_generation_prompt=False
        )

        rejected_msgs = [
            {"role": "user", "content": item["prompt"]},
            {"role": "assistant", "content": item["rejected"]},
        ]
        rejected_text = tokenizer.apply_chat_template(
            rejected_msgs, tokenize=False, add_generation_prompt=False
        )

        prompts.append(prompt_text)
        chosens.append(chosen_text)
        rejecteds.append(rejected_text)

    return Dataset.from_dict({
        "prompt": prompts,
        "chosen": chosens,
        "rejected": rejecteds,
    })


def load_sft_merged_model(model_path, sft_adapter):
    """Load base model + SFT LoRA adapter, merge, return merged model."""
    print(f"Loading base model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading SFT adapter: {sft_adapter}")
    model = PeftModel.from_pretrained(model, sft_adapter)

    print("Merging SFT LoRA into base...")
    model = model.merge_and_unload()
    return model


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Exp 006 Stage 1: Per-N DPO Training")
    parser.add_argument("--model_path", default="/root/autodl-tmp/models/Qwen2.5-3B")
    parser.add_argument("--sft_adapter", required=True,
                        help="Path to shared SFT LoRA adapter directory")
    parser.add_argument("--data_dir", default="/root/autodl-tmp/learned_traj_compress/artifacts/phase1d_v2_data")
    parser.add_argument("--n_objectives", type=int, required=True, choices=[2, 4, 8],
                        help="Which N's DPO data to use")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=6144)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--wandb_project", default="learned_traj_compress")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=42,
                        help="Must match SFT script's split_seed")
    parser.add_argument("--full_data", action="store_true",
                        help="(Reviewer Warning 1 ablation) Use ALL available DPO pairs "
                             "for this N instead of matched-sample. Default OFF → "
                             "matched-sample across N=2/4/8 (main paper result, fair "
                             "comparison). When ON, output_dir and run_name get "
                             "'_full' suffix.")
    parser.add_argument("--matched_size", type=int, default=None,
                        help="Manual override of matched-sample size. If None, "
                             "auto-compute min(pairs) across N=2/4/8.")
    parser.add_argument("--subsample_seed", type=int, default=42,
                        help="Seed for random matched-sample subsampling "
                             "(decoupled from --seed so training rng can vary).")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Disable eval during DPO training (saves GPU memory "
                             "for long sequences). Eval is done separately in Step 3.")
    args = parser.parse_args()

    # Auto-suffix output_dir / run_name for full-data ablation
    full_suffix = "_full" if args.full_data else ""
    if args.output_dir is None:
        base = os.path.dirname(args.sft_adapter)
        args.output_dir = os.path.join(base, f"exp_006_dpo_N{args.n_objectives}{full_suffix}")
    elif args.full_data and not args.output_dir.rstrip("/").endswith("_full"):
        args.output_dir = args.output_dir.rstrip("/") + "_full"
    if args.run_name is None:
        args.run_name = f"exp_006_dpo_N{args.n_objectives}{full_suffix}"
    elif args.full_data and not args.run_name.endswith("_full"):
        args.run_name = args.run_name + "_full"

    mode_label = "full" if args.full_data else "matched"
    print(f"[dpo] mode = {mode_label}")
    print(f"[dpo] output_dir = {args.output_dir}")
    print(f"[dpo] run_name   = {args.run_name}")

    t0 = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load data ──
    print(f"=== Loading DPO data (N={args.n_objectives}) ===")
    raw_data = load_dpo_data(args.data_dir, args.n_objectives)
    raw_count_original = len(raw_data)

    # ── Matched-sample subsampling (Reviewer Warning 1) ──
    matched_size_used = None
    if not args.full_data:
        all_counts = count_dpo_pairs_all_n(args.data_dir)
        if args.matched_size is not None:
            min_pairs = args.matched_size
            print(f"[matched] user-specified matched_size={min_pairs}")
        else:
            if not all_counts:
                raise RuntimeError(f"No dpo_data_N*.json found in {args.data_dir}")
            min_pairs = min(all_counts.values())
            print(f"[matched] pair counts across all N: {all_counts} -> min={min_pairs}")
        matched_size_used = min_pairs
        if len(raw_data) > min_pairs:
            rng = random.Random(args.subsample_seed)
            raw_data = rng.sample(raw_data, min_pairs)
            print(f"[matched] N={args.n_objectives} subsampled: "
                  f"{raw_count_original} -> {len(raw_data)} "
                  f"(subsample_seed={args.subsample_seed})")
        else:
            print(f"[matched] N={args.n_objectives} already has {len(raw_data)} <= {min_pairs}, no subsample")
    else:
        print(f"[full] using all {len(raw_data)} pairs for N={args.n_objectives} (ablation, not main claim)")

    splits = split_data(raw_data, prompt_key="prompt", seed=args.split_seed)
    print(f"\nPrompt-level split (seed={args.split_seed}):")
    print(f"  train: {len(splits['train'])} pairs")
    print(f"  val:   {len(splits['val'])} pairs")
    print(f"  test:  {len(splits['test'])} pairs (held out)")

    # D024 split integrity assertion: abort if counts don't match stats_d024.json
    # Only verify when using full data (not matched-subsampled, which changes counts)
    if args.full_data or matched_size_used is None or matched_size_used >= raw_count_original:
        verify_split_counts(splits, args.data_dir, data_type="dpo", n_value=args.n_objectives)

    if len(splits["train"]) == 0:
        raise ValueError("No training data after split! Check data or split seed.")

    # Save test DPO pairs for eval
    test_dpo_path = os.path.join(args.output_dir, "test_dpo_data.json")
    with open(test_dpo_path, "w") as f:
        json.dump(splits["test"], f, indent=2, ensure_ascii=False)

    # ── Load tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(args.sft_adapter, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Format datasets
    train_dataset = format_for_dpo(splits["train"], tokenizer)
    print(f"Formatted DPO train: {len(train_dataset)} pairs")

    # Val dataset (may be small but useful for monitoring)
    val_dataset = None
    if len(splits["val"]) >= 2:
        val_dataset = format_for_dpo(splits["val"], tokenizer)
        print(f"Formatted DPO val: {len(val_dataset)} pairs")
    else:
        print(f"WARNING: Only {len(splits['val'])} val pairs, skipping eval during training")

    # ── Load base + SFT merged model ──
    print(f"\n=== Loading SFT-merged model ===")
    model = load_sft_merged_model(args.model_path, args.sft_adapter)

    # New LoRA for DPO
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    # ── DPO training config ──
    # Compute warmup_steps from ratio (TRL 1.0 deprecated warmup_ratio)
    world_size = max(int(os.environ.get("WORLD_SIZE", 1)), 1)
    total_train_steps = int(
        args.num_epochs * len(train_dataset)
        / (args.batch_size * args.grad_accum * world_size)
    )
    warmup_steps = max(int(total_train_steps * 0.1), 1)
    print(f"[dpo] total_train_steps={total_train_steps}, warmup_steps={warmup_steps}")

    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        beta=args.beta,
        warmup_steps=warmup_steps,
        logging_steps=5,
        eval_strategy="steps" if (val_dataset is not None and not args.skip_eval) else "no",
        eval_steps=20 if (val_dataset is not None and not args.skip_eval) else None,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_length,
        report_to="wandb",
        run_name=args.run_name,
        seed=args.seed,
        dataloader_pin_memory=False,
    )

    # ── Train ──
    print(f"\n=== Starting DPO Training (N={args.n_objectives}) ===")
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )
    trainer.train()

    # Save DPO adapter
    print(f"\nSaving DPO adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save training metadata
    meta = {
        "experiment": f"exp_006_stage1_dpo_N{args.n_objectives}{full_suffix}",
        "base_model": args.model_path,
        "sft_adapter": args.sft_adapter,
        "n_objectives": args.n_objectives,
        "mode": mode_label,
        "full_data": args.full_data,
        "matched_size_used": matched_size_used,
        "subsample_seed": args.subsample_seed,
        "raw_pairs_before_subsample": raw_count_original,
        "pairs_after_subsample": len(raw_data),
        "split_seed": args.split_seed,
        "num_train": len(train_dataset),
        "num_val": len(val_dataset) if val_dataset else 0,
        "num_test": len(splits["test"]),
        "num_epochs": args.num_epochs,
        "lr": args.lr,
        "beta": args.beta,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "elapsed_seconds": time.time() - t0,
    }
    with open(os.path.join(args.output_dir, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDPO training (N={args.n_objectives}) complete in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Adapter saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
