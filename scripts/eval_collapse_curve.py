#!/usr/bin/env python3
"""Phase 1a: Collapse Curve evaluation for MEM1 RL 7B.

Measures EM as a function of objective count N to identify the collapse point
where multi-objective memory consolidation degrades.

Usage:
    cd /root/autodl-tmp/learned_traj_compress
    python scripts/eval_collapse_curve.py
    python scripts/eval_collapse_curve.py --n_values 1 2 4 --samples_per_n 10  # quick test
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Ensure project root is on path for src imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.metrics import extract_solution, evaluate_multi_objective


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

DEFAULT_N_VALUES = [1, 2, 4, 8, 12, 16]
DEFAULT_SAMPLES_PER_N = 50
DEFAULT_SEED = 42
DEFAULT_MEM1_DIR = "/root/autodl-tmp/models/MEM1"
DEFAULT_BASE_DIR = "/root/autodl-tmp/models/Qwen2.5-7B"
DEFAULT_TOKENIZER_DIR = "/root/autodl-tmp/models/MEM1"
DEFAULT_OUTPUT_PATH = "/root/autodl-tmp/learned_traj_compress/artifacts/exp_002_collapse_curve_results.json"
DEFAULT_HF_CACHE = "/root/autodl-tmp/.hf_cache"
MAX_INPUT_TOKENS = 7168  # conservative limit for 8K context models


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_qa_pool(cache_dir: str, seed: int) -> list[dict]:
    """Load QA pool from HotpotQA (validation, distractor).

    Only uses datasets with real gold passages to avoid data leakage.
    HotpotQA validation has ~7405 samples, sufficient for 16*50=800 needed.

    Each item: {"question": str, "answers": list[str], "passage": str}
    """
    rng = random.Random(seed)
    pool = []

    # --- HotpotQA (validation, distractor) ---
    print("Loading HotpotQA validation split...")
    hotpot = load_dataset("hotpot_qa", "distractor", split="validation", cache_dir=cache_dir)
    for item in hotpot:
        # Concatenate the supporting paragraphs as the gold passage
        passage_parts = []
        sup_titles = set(item["supporting_facts"]["title"])
        for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
            if title in sup_titles:
                passage_parts.append(title + ": " + " ".join(sentences))
        passage = " ".join(passage_parts)
        if passage.strip():
            pool.append({
                "question": item["question"],
                "answers": [item["answer"]],
                "passage": passage,
                "source": "hotpotqa",
            })

    rng.shuffle(pool)
    print(f"QA pool size: {len(pool)} (HotpotQA only, gold passages)")
    return pool


def build_samples(pool: list[dict], n: int, num_samples: int, seed: int) -> list[list[dict]]:
    """Build evaluation samples: each sample = N independent QA items.

    Returns list of samples, each sample is a list of N QA dicts.
    """
    rng = random.Random(seed + n)  # different seed per N for independence
    # Need n * num_samples unique items
    needed = n * num_samples
    if needed > len(pool):
        # Sample with replacement if pool is too small
        items = [rng.choice(pool) for _ in range(needed)]
    else:
        items = rng.sample(pool, needed)

    samples = []
    for i in range(num_samples):
        samples.append(items[i * n : (i + 1) * n])
    return samples


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful assistant. You have access to a search engine. "
    "You need to answer all of the following questions. For each question, "
    "search for information and then provide the answer."
)


def build_prompt(sample: list[dict], tokenizer, max_tokens: int = MAX_INPUT_TOKENS) -> str:
    """Build MEM1-style multi-objective prompt with gold passages.

    Returns the full prompt string after applying chat template.
    """
    # Build question list
    q_lines = []
    for i, item in enumerate(sample, 1):
        q_lines.append(f"Q{i}: {item['question']}")

    # Build passage list
    p_lines = []
    for i, item in enumerate(sample, 1):
        p_lines.append(f"[{i}] {item['passage']}")

    user_content = (
        "Answer all of the following questions:\n"
        + "\n".join(q_lines)
        + "\n\nHere are relevant passages:\n"
        + "\n".join(p_lines)
        + "\n\nBased on the passages above, provide your answers separated by semicolons in <answer> tags."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Truncate passages if too long
    token_count = len(tokenizer.encode(prompt))
    if token_count > max_tokens:
        # Truncate each passage proportionally
        overflow = token_count - max_tokens
        chars_to_cut = overflow * 4  # rough estimate: 1 token ~ 4 chars
        cut_per_passage = chars_to_cut // len(sample) + 1
        for item in sample:
            if len(item["passage"]) > cut_per_passage + 50:
                item["passage"] = item["passage"][:-(cut_per_passage)] + "..."
        # Rebuild
        p_lines = [f"[{i}] {item['passage']}" for i, item in enumerate(sample, 1)]
        user_content = (
            "Answer all of the following questions:\n"
            + "\n".join(q_lines)
            + "\n\nHere are relevant passages:\n"
            + "\n".join(p_lines)
            + "\n\nBased on the passages above, provide your answers separated by semicolons in <answer> tags."
        )
        messages[1]["content"] = user_content
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return prompt


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 2048,
) -> tuple[str, float, int]:
    """Run greedy inference, return (output_text, elapsed_seconds, num_generated_tokens)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    elapsed = time.time() - t0

    generated_ids = output_ids[0, input_len:]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return output_text, elapsed, len(generated_ids)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate_model_full(
    model_name: str,
    model,
    tokenizer,
    all_samples: dict[int, list[list[dict]]],
    n_values: list[int],
) -> tuple[dict, dict]:
    """Run evaluation across all N values for one model. Returns (agg_results, per_sample)."""
    agg_results = {}
    per_sample = {}

    for n in n_values:
        samples = all_samples[n]
        sample_results = []
        ems = []
        f1s = []
        extraction_fails = 0
        total_objectives = 0
        collapsed_objectives = 0
        total_tokens = 0
        total_time = 0.0

        desc = f"{model_name} N={n}"
        for sample in tqdm(samples, desc=desc):
            ground_truths = [item["answers"] for item in sample]
            prompt = build_prompt(sample, tokenizer)

            output_text, elapsed, n_tokens = run_inference(model, tokenizer, prompt)
            total_tokens += n_tokens
            total_time += elapsed

            eval_result = evaluate_multi_objective(
                output_text, ground_truths, min_answer_tags=1
            )

            sample_results.append({
                "n_objectives": n,
                "em_mean": eval_result["em_mean"],
                "f1_mean": eval_result["f1_mean"],
                "em_total": eval_result["em_total"],
                "extraction_failed": eval_result["extraction_failed"],
                "per_question": eval_result["per_question"],
                "output_text": output_text[:500],
                "elapsed_s": round(elapsed, 2),
                "n_tokens": n_tokens,
            })

            ems.append(eval_result["em_mean"])
            f1s.append(eval_result["f1_mean"])

            if eval_result["extraction_failed"]:
                extraction_fails += 1

            for q in eval_result["per_question"]:
                total_objectives += 1
                if q["em"] == 0:
                    collapsed_objectives += 1

        mean_em = float(np.mean(ems))
        std_em = float(np.std(ems))
        mean_f1 = float(np.mean(f1s))
        collapse_rate = collapsed_objectives / total_objectives if total_objectives > 0 else 0.0
        extraction_fail_rate = extraction_fails / len(samples)
        mean_tps = total_tokens / total_time if total_time > 0 else 0.0

        agg_results[str(n)] = {
            "mean_em": round(mean_em, 4),
            "std_em": round(std_em, 4),
            "mean_f1": round(mean_f1, 4),
            "collapse_rate": round(collapse_rate, 4),
            "extraction_fail_rate": round(extraction_fail_rate, 4),
            "mean_tps": round(mean_tps, 1),
            "n_samples": len(samples),
        }
        per_sample[str(n)] = sample_results

        print(f"  {model_name} N={n}: EM={mean_em:.3f}+-{std_em:.3f}, "
              f"F1={mean_f1:.3f}, collapse={collapse_rate:.2%}, "
              f"extract_fail={extraction_fail_rate:.2%}, tps={mean_tps:.0f}")

    return agg_results, per_sample


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def run_statistical_tests(
    per_sample: dict[str, list[dict]],
    model_name: str,
) -> dict:
    """Mann-Whitney U tests: N>=8 vs N=1,2."""
    tests = {}
    baselines = ["1", "2"]
    high_ns = [str(n) for n in [8, 12, 16]]

    for high_n in high_ns:
        if high_n not in per_sample:
            continue
        high_ems = [s["em_mean"] for s in per_sample[high_n]]
        for base_n in baselines:
            if base_n not in per_sample:
                continue
            base_ems = [s["em_mean"] for s in per_sample[base_n]]
            try:
                u_stat, p_value = stats.mannwhitneyu(
                    base_ems, high_ems, alternative="greater"
                )
                tests[f"{model_name}_n{high_n}_vs_n{base_n}"] = {
                    "u_stat": round(float(u_stat), 2),
                    "p_value": round(float(p_value), 6),
                    "significant": p_value < 0.05,
                }
            except ValueError:
                tests[f"{model_name}_n{high_n}_vs_n{base_n}"] = {
                    "u_stat": None,
                    "p_value": None,
                    "significant": False,
                    "error": "identical distributions",
                }
    return tests


# ---------------------------------------------------------------------------
# Go/No-Go
# ---------------------------------------------------------------------------

def compute_go_no_go(
    mem1_results: dict,
    base_results: dict,
    stat_tests: dict,
) -> dict:
    """Automatic go/no-go assessment."""

    # 1. Collapse exists: MEM1 N>=8 significantly worse than N=1/2 AND collapse_rate > 20%
    collapse_exists = False
    for key, test in stat_tests.items():
        if key.startswith("mem1") and test.get("significant"):
            # Check collapse rate at the high N
            high_n = key.split("_n")[1].split("_")[0]
            if high_n in mem1_results and mem1_results[high_n]["collapse_rate"] > 0.2:
                collapse_exists = True
                break

    # 2. RL-base gap: MEM1 > base by >=0.1 EM at N>=8
    rl_base_gap = False
    for n_str in ["8", "12", "16"]:
        if n_str in mem1_results and n_str in base_results:
            gap = mem1_results[n_str]["mean_em"] - base_results[n_str]["mean_em"]
            if gap >= 0.1:
                rl_base_gap = True
                break

    # 3. Curve nonlinearity: EM drop from N=4->8 much steeper than N=1->4
    curve_nonlinear = False
    if all(n in mem1_results for n in ["1", "4", "8"]):
        drop_1_4 = mem1_results["1"]["mean_em"] - mem1_results["4"]["mean_em"]
        drop_4_8 = mem1_results["4"]["mean_em"] - mem1_results["8"]["mean_em"]
        # Nonlinear if the 4->8 drop is at least 2x the 1->4 drop (and meaningful)
        if drop_4_8 > 0.05 and (drop_1_4 <= 0 or drop_4_8 / max(drop_1_4, 1e-6) > 2.0):
            curve_nonlinear = True

    # Overall
    checks = [collapse_exists, rl_base_gap, curve_nonlinear]
    if all(checks):
        overall = "go"
    elif any(checks):
        overall = "partial"
    else:
        overall = "no-go"

    return {
        "collapse_exists": collapse_exists,
        "rl_base_gap": rl_base_gap,
        "curve_nonlinear": curve_nonlinear,
        "overall": overall,
    }


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(mem1_results: dict, base_results: dict, go_no_go: dict):
    """Print readable summary table."""
    print("\n" + "=" * 80)
    print("COLLAPSE CURVE RESULTS")
    print("=" * 80)

    header = f"{'N':>4} | {'MEM1 EM':>10} | {'Base EM':>10} | {'Gap':>8} | {'MEM1 Collapse%':>15} | {'MEM1 F1':>8}"
    print(header)
    print("-" * len(header))

    all_ns = sorted(set(list(mem1_results.keys()) + list(base_results.keys())), key=int)
    for n in all_ns:
        m = mem1_results.get(n, {})
        b = base_results.get(n, {})
        m_em = m.get("mean_em", float("nan"))
        b_em = b.get("mean_em", float("nan"))
        gap = m_em - b_em if not (np.isnan(m_em) or np.isnan(b_em)) else float("nan")
        collapse = m.get("collapse_rate", float("nan"))
        f1 = m.get("mean_f1", float("nan"))
        print(f"{n:>4} | {m_em:>10.4f} | {b_em:>10.4f} | {gap:>+8.4f} | {collapse:>14.2%} | {f1:>8.4f}")

    print("\n--- Go/No-Go ---")
    for k, v in go_no_go.items():
        print(f"  {k}: {v}")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 1a: Collapse Curve Evaluation")
    parser.add_argument("--mem1_dir", default=DEFAULT_MEM1_DIR)
    parser.add_argument("--base_dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--tokenizer_dir", default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--n_values", type=int, nargs="+", default=DEFAULT_N_VALUES)
    parser.add_argument("--samples_per_n", type=int, default=DEFAULT_SAMPLES_PER_N)
    parser.add_argument("--output_path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--hf_cache", default=DEFAULT_HF_CACHE)
    parser.add_argument("--skip_base", action="store_true", help="Skip base model eval")
    args = parser.parse_args()

    os.environ["HF_HOME"] = args.hf_cache
    random.seed(args.seed)
    np.random.seed(args.seed)

    # --- Data ---
    print("=== Loading QA data ===")
    pool = load_qa_pool(args.hf_cache, args.seed)

    all_samples = {}
    for n in args.n_values:
        all_samples[n] = build_samples(pool, n, args.samples_per_n, args.seed)
        print(f"  N={n}: {len(all_samples[n])} samples, {n * len(all_samples[n])} questions")

    # --- Tokenizer ---
    print("\n=== Loading tokenizer ===")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- MEM1 ---
    print("\n=== Evaluating MEM1 ===")
    model = AutoModelForCausalLM.from_pretrained(
        args.mem1_dir,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()
    mem1_results, mem1_per_sample = evaluate_model_full(
        "mem1", model, tokenizer, all_samples, args.n_values
    )
    del model
    torch.cuda.empty_cache()

    # --- Base ---
    base_results = {}
    base_per_sample = {}
    if not args.skip_base:
        print("\n=== Evaluating Qwen2.5-7B base ===")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_dir,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
        )
        model.eval()
        base_results, base_per_sample = evaluate_model_full(
            "base", model, tokenizer, all_samples, args.n_values
        )
        del model
        torch.cuda.empty_cache()

    # --- Statistical tests ---
    print("\n=== Statistical tests ===")
    stat_tests = {}
    stat_tests.update(run_statistical_tests(mem1_per_sample, "mem1"))
    if base_per_sample:
        stat_tests.update(run_statistical_tests(base_per_sample, "base"))

    for name, result in stat_tests.items():
        sig = "*" if result.get("significant") else ""
        p = result.get("p_value", "N/A")
        print(f"  {name}: U={result.get('u_stat')}, p={p} {sig}")

    # --- Go/No-Go ---
    go_no_go = compute_go_no_go(mem1_results, base_results, stat_tests)

    # --- Output ---
    output = {
        "config": {
            "n_values": args.n_values,
            "samples_per_n": args.samples_per_n,
            "seed": args.seed,
            "mem1_dir": args.mem1_dir,
            "base_dir": args.base_dir,
        },
        "results": {
            "mem1": mem1_results,
            "base": base_results,
        },
        "statistical_tests": stat_tests,
        "go_no_go": go_no_go,
        "per_sample_details": {
            "mem1": mem1_per_sample,
            "base": base_per_sample,
        },
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")

    # --- Summary ---
    print_summary(mem1_results, base_results, go_no_go)


if __name__ == "__main__":
    main()
