#!/usr/bin/env python3
"""
Phase 1d Data Generation: Generate trajectories from MEM1 RL teacher for SFT + DPO training.

Generates 2 responses per prompt (T=0.7) at N=2,4,8.
- SFT data: perfect EM trajectories
- DPO data: (chosen, rejected) pairs where chosen_em > rejected_em

Usage:
    python scripts/phase1d_generate_data.py \
        --rl_dir /root/autodl-tmp/models/MEM1 \
        --output_dir /root/autodl-tmp/learned_traj_compress/artifacts/phase1d_data \
        [--dry_run]
"""

import argparse
import gc
import json
import os
import random
import re
import string
import time
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Prompt and eval (shared with Phase 1a)
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


def build_prompt_and_info(sample, tokenizer):
    """Build prompt for a sample of N QA items. Returns (prompt, user_content, assistant_prefix)."""
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


def evaluate_response(response, sample):
    """Evaluate a response against the sample's ground truths. Returns (total_em, per_obj_em)."""
    raw_answer = extract_answer(response)
    n = len(sample)
    if raw_answer is None:
        return 0, [0] * n

    parts = [a.strip() for a in raw_answer.split(";")]
    per_obj_em = []
    for idx, item in enumerate(sample):
        if idx < len(parts):
            em = 1 if em_check(parts[idx], item["answers"]) else 0
        else:
            em = 0
        per_obj_em.append(em)
    return sum(per_obj_em), per_obj_em


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_paired_trajectories(model, tokenizer, pool, n_objectives, n_prompts, seed, device,
                                  max_new_tokens=512):
    """Generate 2 responses per prompt for DPO pair construction."""
    rng = random.Random(seed)
    needed = n_objectives * n_prompts
    if needed > len(pool):
        items = [rng.choice(pool) for _ in range(needed)]
    else:
        items = rng.sample(pool, needed)
    samples = [items[i * n_objectives:(i + 1) * n_objectives] for i in range(n_prompts)]

    results = []
    for idx, sample in enumerate(tqdm(samples, desc=f"N={n_objectives}")):
        prompt, user_content, assistant_prefix = build_prompt_and_info(sample, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        responses = []
        for resp_idx in range(2):
            with torch.no_grad():
                if resp_idx == 0:
                    # Greedy for response A
                    output_ids = model.generate(
                        **inputs, max_new_tokens=max_new_tokens,
                        do_sample=False, temperature=None, top_p=None)
                else:
                    # Sampled for response B
                    output_ids = model.generate(
                        **inputs, max_new_tokens=max_new_tokens,
                        do_sample=True, temperature=0.7, top_p=0.9)

            response = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:],
                                        skip_special_tokens=True)
            total_em, per_obj_em = evaluate_response(response, sample)
            responses.append({
                "response": response,
                "full_assistant": assistant_prefix + response,
                "total_em": total_em,
                "per_obj_em": per_obj_em,
                "is_perfect": total_em == n_objectives,
            })

        results.append({
            "prompt_idx": idx,
            "n_objectives": n_objectives,
            "user_content": user_content,
            "assistant_prefix": assistant_prefix,
            "questions": [item["question"] for item in sample],
            "gold_answers": [item["answers"] for item in sample],
            "responses": responses,
        })

    return results


def build_sft_and_dpo_data(all_results):
    """Split generated trajectories into SFT and DPO datasets."""
    sft_data = []
    dpo_data = []

    for item in all_results:
        r0, r1 = item["responses"]

        # SFT: any perfect response
        for r in [r0, r1]:
            if r["is_perfect"]:
                sft_data.append({
                    "user_content": item["user_content"],
                    "assistant_content": r["full_assistant"],
                    "n_objectives": item["n_objectives"],
                    "total_em": r["total_em"],
                })

        # DPO: pair with EM difference (filtered by margin >= 1 objective)
        if r0["total_em"] != r1["total_em"]:
            if r0["total_em"] > r1["total_em"]:
                chosen, rejected = r0, r1
            else:
                chosen, rejected = r1, r0

            # Require meaningful EM gap (at least 1 objective difference)
            n = item["n_objectives"]
            em_gap = (chosen["total_em"] - rejected["total_em"]) / n
            if em_gap >= 0.1:
                dpo_data.append({
                    "prompt": item["user_content"],
                    "chosen": chosen["full_assistant"],
                    "rejected": rejected["full_assistant"],
                    "n_objectives": item["n_objectives"],
                    "chosen_em": chosen["total_em"],
                    "rejected_em": rejected["total_em"],
                    "em_gap": float(em_gap),
                })

    return sft_data, dpo_data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 1d: Data Generation")
    parser.add_argument("--rl_dir", default="/root/autodl-tmp/models/MEM1")
    parser.add_argument("--output_dir", default="/root/autodl-tmp/learned_traj_compress/artifacts/phase1d_data")
    parser.add_argument("--cache_dir", default="/root/autodl-tmp/.hf_cache")
    parser.add_argument("--n_values", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--prompts_per_n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        args.prompts_per_n = 10
        print("*** DRY RUN: 10 prompts per N ***")

    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()

    # Load tokenizer and QA pool
    tokenizer = AutoTokenizer.from_pretrained(args.rl_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading QA pool...")
    pool = load_qa_pool(args.cache_dir, args.seed)
    print(f"QA pool: {len(pool)} items")

    # Load RL model
    print("Loading MEM1 RL model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.rl_dir, torch_dtype=torch.float16, device_map=args.device, trust_remote_code=True)
    model.eval()

    # Generate for each N
    all_results = []
    for n in args.n_values:
        # Use different seed offset per N to get different prompts
        results = generate_paired_trajectories(
            model, tokenizer, pool, n, args.prompts_per_n,
            seed=args.seed + n * 1000, device=args.device)
        all_results.extend(results)
        print(f"N={n}: {len(results)} prompt pairs generated")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Build SFT and DPO datasets
    sft_data, dpo_data = build_sft_and_dpo_data(all_results)

    # Stats
    stats = {
        "total_prompts": len(all_results),
        "sft_samples": len(sft_data),
        "dpo_pairs": len(dpo_data),
        "per_n": {},
    }
    for n in args.n_values:
        n_items = [r for r in all_results if r["n_objectives"] == n]
        n_perfect = sum(1 for r in n_items for resp in r["responses"] if resp["is_perfect"])
        n_dpo = sum(1 for r in n_items
                    if r["responses"][0]["total_em"] != r["responses"][1]["total_em"])
        stats["per_n"][str(n)] = {
            "prompts": len(n_items),
            "perfect_responses": n_perfect,
            "dpo_pairs": n_dpo,
        }

    print(f"\n=== Data Generation Summary ===")
    print(f"Total prompts: {stats['total_prompts']}")
    print(f"SFT samples (perfect EM): {stats['sft_samples']}")
    print(f"DPO pairs: {stats['dpo_pairs']}")
    for n_str, ns in stats["per_n"].items():
        print(f"  N={n_str}: {ns['prompts']} prompts, {ns['perfect_responses']} perfect, {ns['dpo_pairs']} DPO pairs")

    # Save
    with open(os.path.join(args.output_dir, "sft_data.json"), "w") as f:
        json.dump(sft_data, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.output_dir, "dpo_data.json"), "w") as f:
        json.dump(dpo_data, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.output_dir, "raw_trajectories.json"), "w") as f:
        json.dump({"stats": stats, "results": all_results}, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.output_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Data saved to {args.output_dir}")


if __name__ == "__main__":
    main()
