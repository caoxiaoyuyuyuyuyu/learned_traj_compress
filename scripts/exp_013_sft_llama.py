#!/usr/bin/env python3
"""
exp_013: W2 cross-backbone pooled SFT — Llama-3.2-1B student.

Mirrors exp_006_stage1_sft.py (rank-32 LoRA on attn+MLP, pooled N=2+4+8 data,
hash-based split) but targets Llama-3.2-1B (unsloth mirror) as the student.
Purpose: replicate the silent-failure mode across model families for W2.

Base Llama-3.2-1B has no chat_template — we apply the official Llama-3.1
chat template string if missing.
"""

import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType
from trl import SFTConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.split import split_data, verify_split_counts
from exp_006_stage1_sft import (
    load_and_merge_sft_data,
    format_for_sft,
    PerNDataCollator,
    PerNLossSFTTrainer,
    get_prompt_text,
)

# Official Llama-3.1/3.2 chat template (base models ship without one).
LLAMA3_CHAT_TEMPLATE = (
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + "
    "'<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}"
    "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
    "{{ content }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    "{% endif %}"
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="/root/autodl-tmp/models/Llama-3.2-1B")
    p.add_argument("--data_dir", default="/root/autodl-tmp/learned_traj_compress/artifacts/phase1d_v2_data")
    p.add_argument("--output_dir", default="/root/autodl-tmp/learned_traj_compress/artifacts/exp_013/sft_llama1b")
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_seq_length", type=int, default=6144)
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--wandb_project", default="learned_traj_compress")
    p.add_argument("--run_name", default="exp_013_sft_llama1b")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    t0 = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=== Loading pooled SFT data ===")
    raw_data, per_n_counts = load_and_merge_sft_data(args.data_dir, balanced=False)
    splits = split_data(raw_data, prompt_key="user_content", seed=args.split_seed)
    print(f"Split: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
    verify_split_counts(splits, args.data_dir, data_type="sft", n_value=None)

    # Save split info / test set for eval parity
    with open(os.path.join(args.output_dir, "prompt_split_info.json"), "w") as f:
        json.dump({
            "seed": args.split_seed,
            "counts": {k: len(v) for k, v in splits.items()},
            "test_prompts": [get_prompt_text(it)[:500] for it in splits["test"]],
        }, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.output_dir, "test_data.json"), "w") as f:
        json.dump(splits["test"], f, indent=2, ensure_ascii=False)

    print(f"\n=== Loading tokenizer: {args.model_path} ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.chat_template is None:
        print("[chat_template] tokenizer has no chat_template; applying Llama-3 template")
        tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = format_for_sft(splits["train"], tokenizer, args.max_seq_length)
    val_ds = format_for_sft(splits["val"], tokenizer, args.max_seq_length)
    print(f"Train={len(train_ds)} Val={len(val_ds)}")

    if args.dry_run:
        print("[dry_run] exiting before model load")
        return

    print(f"\n=== Loading model: {args.model_path} ===")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    world_size = max(int(os.environ.get("WORLD_SIZE", 1)), 1)
    total_steps = int(args.num_epochs * len(train_ds) / (args.batch_size * args.grad_accum * world_size))
    warmup_steps = max(int(total_steps * 0.1), 1)

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        logging_steps=5,
        eval_strategy="steps", eval_steps=20,
        save_strategy="steps", save_steps=20, save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
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

    collator = PerNDataCollator(tokenizer)
    trainer = PerNLossSFTTrainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        peft_config=lora_config, processing_class=tokenizer,
        data_collator=collator,
    )
    trainer.train()

    print(f"\nSaving LoRA adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    meta = {
        "experiment": "exp_013_sft_llama1b",
        "base_model": args.model_path,
        "data_dir": args.data_dir,
        "split_seed": args.split_seed,
        "per_n_counts": per_n_counts,
        "num_train": len(train_ds), "num_val": len(val_ds), "num_test": len(splits["test"]),
        "num_epochs": args.num_epochs, "lr": args.lr,
        "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
        "elapsed_seconds": time.time() - t0,
    }
    with open(os.path.join(args.output_dir, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Done in {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
