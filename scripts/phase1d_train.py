#!/usr/bin/env python3
"""
Phase 1d Training: SFT or DPO LoRA training for student models.

Trains a student model (1.5B or 3B) using either SFT or DPO on teacher-generated data.
Uses LoRA for parameter-efficient training.

Usage:
    # SFT training
    python scripts/phase1d_train.py \
        --base_dir /root/autodl-tmp/models/Qwen2.5-1.5B \
        --data_dir /root/autodl-tmp/learned_traj_compress/artifacts/phase1d_data \
        --output_dir /root/autodl-tmp/models/phase1d/Qwen2.5-1.5B-SFT \
        --method sft

    # DPO training
    python scripts/phase1d_train.py \
        --base_dir /root/autodl-tmp/models/Qwen2.5-3B \
        --data_dir /root/autodl-tmp/learned_traj_compress/artifacts/phase1d_data \
        --output_dir /root/autodl-tmp/models/phase1d/Qwen2.5-3B-DPO \
        --method dpo
"""

import argparse
import gc
import json
import os
import time

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sft_data(data_dir, tokenizer, max_length=4096):
    """Load SFT data and tokenize."""
    with open(os.path.join(data_dir, "sft_data.json")) as f:
        sft_data = json.load(f)

    texts = []
    for item in sft_data:
        messages = [
            {"role": "user", "content": item["user_content"]},
            {"role": "assistant", "content": item["assistant_content"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length, padding=False)

    dataset = Dataset.from_dict({"text": texts})
    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    print(f"SFT dataset: {len(dataset)} samples")
    return dataset


def load_dpo_data(data_dir, tokenizer, max_length=4096):
    """Load DPO data as prompt/chosen/rejected triplets."""
    with open(os.path.join(data_dir, "dpo_data.json")) as f:
        dpo_data = json.load(f)

    prompts, chosens, rejecteds = [], [], []
    for item in dpo_data:
        # Format as chat messages
        prompt_msgs = [{"role": "user", "content": item["prompt"]}]
        prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)

        chosen_msgs = [
            {"role": "user", "content": item["prompt"]},
            {"role": "assistant", "content": item["chosen"]},
        ]
        chosen_text = tokenizer.apply_chat_template(chosen_msgs, tokenize=False, add_generation_prompt=False)

        rejected_msgs = [
            {"role": "user", "content": item["prompt"]},
            {"role": "assistant", "content": item["rejected"]},
        ]
        rejected_text = tokenizer.apply_chat_template(rejected_msgs, tokenize=False, add_generation_prompt=False)

        prompts.append(prompt_text)
        chosens.append(chosen_text)
        rejecteds.append(rejected_text)

    dataset = Dataset.from_dict({
        "prompt": prompts,
        "chosen": chosens,
        "rejected": rejecteds,
    })
    print(f"DPO dataset: {len(dataset)} pairs")
    return dataset


# ---------------------------------------------------------------------------
# SFT Training
# ---------------------------------------------------------------------------

def train_sft(base_dir, data_dir, output_dir, num_epochs=3, lr=2e-5, batch_size=1,
              grad_accum=4, lora_rank=16, lora_alpha=32, device="cuda:0"):
    """SFT LoRA training."""
    print(f"\n=== SFT LoRA Training ===")
    print(f"Base model: {base_dir}")

    tokenizer = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_sft_data(data_dir, tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        base_dir, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True)

    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    from transformers import Trainer, DataCollatorForLanguageModeling
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        dataloader_pin_memory=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()

    # Save merged model (LoRA + base)
    print("Merging LoRA weights...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Merged model saved to {output_dir}")

    del model, merged_model, trainer
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# DPO Training
# ---------------------------------------------------------------------------

def train_dpo(base_dir, data_dir, output_dir, num_epochs=3, lr=5e-6, batch_size=1,
              grad_accum=4, lora_rank=16, lora_alpha=32, beta=0.1, device="cuda:0",
              sft_checkpoint=None):
    """DPO LoRA training using TRL. If sft_checkpoint is provided, init from SFT model."""
    init_dir = sft_checkpoint or base_dir
    print(f"\n=== DPO LoRA Training ===")
    print(f"Init model: {init_dir}")
    if sft_checkpoint:
        print(f"(SFT checkpoint, base was: {base_dir})")
    print(f"Beta: {beta}")

    from trl import DPOTrainer, DPOConfig

    tokenizer = AutoTokenizer.from_pretrained(init_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dpo_data(data_dir, tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        init_dir, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True)

    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        beta=beta,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        peft_config=lora_config,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save merged model
    print("Merging LoRA weights...")
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Merged model saved to {output_dir}")

    del model, merged_model, trainer
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 1d: SFT/DPO LoRA Training")
    parser.add_argument("--base_dir", required=True, help="Path to base model")
    parser.add_argument("--data_dir", default="/root/autodl-tmp/learned_traj_compress/artifacts/phase1d_data")
    parser.add_argument("--output_dir", required=True, help="Output directory for trained model")
    parser.add_argument("--method", choices=["sft", "dpo"], required=True)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: 2e-5 for SFT, 5e-6 for DPO)")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    parser.add_argument("--sft_checkpoint", default=None, help="SFT checkpoint for DPO init (required for DPO)")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.lr is None:
        args.lr = 2e-5 if args.method == "sft" else 5e-6

    t0 = time.time()

    if args.method == "sft":
        train_sft(args.base_dir, args.data_dir, args.output_dir,
                  num_epochs=args.num_epochs, lr=args.lr, batch_size=args.batch_size,
                  grad_accum=args.grad_accum, lora_rank=args.lora_rank,
                  lora_alpha=args.lora_alpha, device=args.device)
    else:
        if args.sft_checkpoint is None:
            print("WARNING: --sft_checkpoint not provided for DPO. Training from base model.")
            print("This is likely wrong — DPO should init from SFT checkpoint.")
        train_dpo(args.base_dir, args.data_dir, args.output_dir,
                  num_epochs=args.num_epochs, lr=args.lr, batch_size=args.batch_size,
                  grad_accum=args.grad_accum, lora_rank=args.lora_rank,
                  lora_alpha=args.lora_alpha, beta=args.beta, device=args.device,
                  sft_checkpoint=args.sft_checkpoint)

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
