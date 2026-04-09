#!/usr/bin/env python3
"""
Quick SFT Baseline: Generate RL trajectories → SFT train Qwen2.5-7B

Purpose: Create a minimal SFT model for three-way SVD comparison.
Not optimized for performance — just needs to exist for weight-space analysis.

Steps:
1. Generate N=2 trajectories from MEM1 RL (~200 samples)
2. Filter for perfect EM=2 (~100 trajectories expected)
3. SFT train Qwen2.5-7B on filtered trajectories (2 epochs)
4. Save checkpoint

Usage:
    python scripts/train_sft_baseline.py \
        --rl_dir /root/autodl-tmp/models/MEM1 \
        --base_dir /root/autodl-tmp/models/Qwen2.5-7B \
        --output_dir /root/autodl-tmp/models/Qwen2.5-7B-SFT-N2 \
        [--dry_run]
"""

import argparse
import gc
import json
import os
import random
import re
import time
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Prompt and eval (reused from Phase 1a)
# ---------------------------------------------------------------------------

MEM1_MULTI_PROMPT = """You will answer multiple complex questions using iterative reasoning, summarization, and web search.

At each step, you will see the questions, a cumulative summary of relevant information, the current search query, and search results (except in the first step, where only the questions are provided). Your task is to:

1. Perform reasoning and update a cumulative, concise summary within <think> ... </think>. This acts as persistent memory and must include all essential information from previous <think> and <information> tags.

2. Then choose one of the following actions:
   - If any question remains unanswered, issue a single query for one question inside <search> ... </search>. The query should consist of keywords or a short phrase. Only search one question at a time.
   - If all questions are answered, provide the final answers—separated by semicolons—within <answer> answer1; answer2; ... </answer>. The answers must be concise, contain only essential words, and avoid any explanations.

Important:
- Always follow this structure after <information> or the initial questions: <think> ... </think><search> ... </search> or <think> ... </think><answer> ... </answer>.
- Do not search multiple queries or questions simultaneously.

Answer the following questions: {questions}"""


def normalize_answer(s):
    import string
    s = s.lower().strip()
    for art in ["a ", "an ", "the "]:
        while s.startswith(art):
            s = s[len(art):]
    s = "".join(c for c in s if c not in string.punctuation)
    s = " ".join(s.split())
    return s


def em_check(prediction, gold_answers):
    pred_norm = normalize_answer(prediction)
    return any(normalize_answer(g) == pred_norm for g in gold_answers)


def extract_answer(text):
    matches = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def load_qa_pool(cache_dir, seed):
    rng = random.Random(seed)
    pool = []
    hotpot = load_dataset("hotpot_qa", "distractor", split="validation", cache_dir=cache_dir)
    for item in hotpot:
        sup_titles = set(item["supporting_facts"]["title"])
        passages = []
        for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
            if title in sup_titles:
                passages.append({"title": title, "text": " ".join(sentences)})
        if passages:
            pool.append({
                "question": item["question"],
                "answers": [item["answer"]],
                "passages": passages,
            })
    rng.shuffle(pool)
    return pool


def build_prompt_and_info(sample, tokenizer, max_tokens=4096):
    """Build prompt and return (prompt, info_block) for SFT data construction."""
    questions = [item["question"] for item in sample]
    questions_str = "; ".join(questions)
    user_content = MEM1_MULTI_PROMPT.format(questions=questions_str)

    all_passages = []
    for item in sample:
        for p in item["passages"]:
            all_passages.append(p)

    info_lines = ["[HINT]You have 1 turn left. You must answer the question now.[/HINT]"]
    for i, p in enumerate(all_passages, 1):
        info_lines.append(f"Doc {i}(Title: {p['title']}) {p['text']}")
    info_block = "\n".join(info_lines)

    search_query = questions[0].split()[:5]
    assistant_prefix = (
        f"<think>Let me search for information about these questions.</think>"
        f"<search>{' '.join(search_query)}</search>\n\n"
        f"<information>\n{info_block}\n</information>\n"
    )

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_prefix},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    if prompt.endswith("<|im_end|>\n"):
        prompt = prompt[:-len("<|im_end|>\n")]
    elif prompt.endswith("<|im_end|>"):
        prompt = prompt[:-len("<|im_end|>")]

    return prompt, user_content, assistant_prefix


# ---------------------------------------------------------------------------
# Step 1: Generate trajectories
# ---------------------------------------------------------------------------

def generate_trajectories(rl_dir, tokenizer, pool, n_objectives, n_generate, seed, device, max_new_tokens=512):
    """Generate trajectories from RL model and evaluate EM."""
    print(f"\n=== Generating {n_generate} trajectories (N={n_objectives}) ===")

    rng = random.Random(seed)
    needed = n_objectives * n_generate
    if needed > len(pool):
        items = [rng.choice(pool) for _ in range(needed)]
    else:
        items = rng.sample(pool, needed)
    samples = [items[i * n_objectives:(i + 1) * n_objectives] for i in range(n_generate)]

    # Load RL model
    print("Loading RL model...")
    rl_model = AutoModelForCausalLM.from_pretrained(
        rl_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True)
    rl_model.eval()

    trajectories = []
    perfect_count = 0

    for idx, sample in enumerate(tqdm(samples, desc="Generating")):
        prompt, user_content, assistant_prefix = build_prompt_and_info(sample, tokenizer)

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = rl_model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=1.0)
        response = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True)

        # Evaluate EM
        raw_answer = extract_answer(response)
        per_obj_em = []
        if raw_answer is not None:
            answer_parts = [a.strip() for a in raw_answer.split(";")]
            for q_idx, item in enumerate(sample):
                if q_idx < len(answer_parts):
                    em = 1 if em_check(answer_parts[q_idx], item["answers"]) else 0
                else:
                    em = 0
                per_obj_em.append(em)
        else:
            per_obj_em = [0] * n_objectives

        total_em = sum(per_obj_em)
        is_perfect = (total_em == n_objectives)
        if is_perfect:
            perfect_count += 1

        trajectories.append({
            "user_content": user_content,
            "assistant_prefix": assistant_prefix,
            "response": response,
            "full_assistant": assistant_prefix + response,
            "per_obj_em": per_obj_em,
            "total_em": total_em,
            "is_perfect": is_perfect,
        })

    del rl_model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Generated {len(trajectories)} trajectories, {perfect_count} perfect ({perfect_count/len(trajectories):.1%})")
    return trajectories


# ---------------------------------------------------------------------------
# Step 2: SFT training
# ---------------------------------------------------------------------------

def prepare_sft_dataset(trajectories, tokenizer, max_length=4096):
    """Convert perfect trajectories to SFT training format."""
    perfect = [t for t in trajectories if t["is_perfect"]]
    print(f"Using {len(perfect)} perfect trajectories for SFT")

    texts = []
    for t in perfect:
        # Format: user message + full assistant response (prefix + generation)
        messages = [
            {"role": "user", "content": t["user_content"]},
            {"role": "assistant", "content": t["full_assistant"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)

    # Tokenize
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    dataset = Dataset.from_dict({"text": texts})
    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    return dataset


def train_sft(base_dir, dataset, output_dir, tokenizer, num_epochs=2, batch_size=1,
              gradient_accumulation=4, lr=2e-5, device="cuda"):
    """Train SFT baseline."""
    print(f"\n=== SFT Training ===")
    print(f"Dataset size: {len(dataset)}")
    print(f"Epochs: {num_epochs}, effective batch: {batch_size * gradient_accumulation}")

    model = AutoModelForCausalLM.from_pretrained(
        base_dir, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
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
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"SFT model saved to {output_dir}")

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Quick SFT Baseline Training")
    parser.add_argument("--rl_dir", default="/root/autodl-tmp/models/MEM1")
    parser.add_argument("--base_dir", default="/root/autodl-tmp/models/Qwen2.5-7B")
    parser.add_argument("--output_dir", default="/root/autodl-tmp/models/Qwen2.5-7B-SFT-N2")
    parser.add_argument("--cache_dir", default="/root/autodl-tmp/.hf_cache")
    parser.add_argument("--n_generate", type=int, default=200,
                        help="Number of trajectories to generate from RL model")
    parser.add_argument("--n_objectives", type=int, default=2,
                        help="Number of objectives per sample")
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry_run", action="store_true",
                        help="Generate 20 trajectories, train 1 epoch")
    args = parser.parse_args()

    if args.dry_run:
        args.n_generate = 20
        args.num_epochs = 1
        print("*** DRY RUN: 20 trajectories, 1 epoch ***")

    t0 = time.time()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.rl_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load QA pool
    print("Loading QA pool...")
    pool = load_qa_pool(args.cache_dir, args.seed)
    print(f"QA pool: {len(pool)} items")

    # Step 1: Generate trajectories
    trajectories = generate_trajectories(
        args.rl_dir, tokenizer, pool, args.n_objectives,
        args.n_generate, args.seed, args.device)

    # Save trajectories
    traj_path = os.path.join(os.path.dirname(args.output_dir), "sft_trajectories_n2.json")
    os.makedirs(os.path.dirname(traj_path), exist_ok=True)
    with open(traj_path, "w") as f:
        json.dump({
            "config": {"n_generate": args.n_generate, "n_objectives": args.n_objectives},
            "stats": {
                "total": len(trajectories),
                "perfect": sum(1 for t in trajectories if t["is_perfect"]),
                "perfect_rate": sum(1 for t in trajectories if t["is_perfect"]) / len(trajectories),
            },
            "trajectories": trajectories,
        }, f, indent=2, ensure_ascii=False)
    print(f"Trajectories saved to {traj_path}")

    # Step 2: Prepare SFT dataset
    perfect_trajs = [t for t in trajectories if t["is_perfect"]]
    if len(perfect_trajs) < 10:
        print(f"WARNING: Only {len(perfect_trajs)} perfect trajectories. Using all trajectories with EM>0 instead.")
        perfect_trajs = [t for t in trajectories if t["total_em"] > 0]
        if len(perfect_trajs) < 5:
            print("ERROR: Not enough usable trajectories. Aborting SFT training.")
            return

    dataset = prepare_sft_dataset(
        [t for t in trajectories if t["is_perfect"]] if len(perfect_trajs) >= 10
        else perfect_trajs,
        tokenizer)

    # Step 3: Train
    train_sft(args.base_dir, dataset, args.output_dir, tokenizer,
              num_epochs=args.num_epochs, lr=args.lr, device=args.device)

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"SFT model at: {args.output_dir}")


if __name__ == "__main__":
    main()
