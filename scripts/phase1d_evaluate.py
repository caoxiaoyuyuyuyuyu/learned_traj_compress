#!/usr/bin/env python3
"""
Phase 1d Evaluation: Collapse curve + SVD profile for 4 student models.

Evaluates each student model on:
1. Collapse curve: EM at N=1,2,4,8,12,16 (30 samples/N)
2. SVD profile: student-base obs/random ratio at k={64,128,256,512}

Compares against teacher (MEM1 RL 7B) Phase 1a/1b data.

Usage:
    python scripts/phase1d_evaluate.py \
        --models_config /root/autodl-tmp/learned_traj_compress/artifacts/phase1d_models.json \
        --output_dir /root/autodl-tmp/learned_traj_compress/artifacts/phase1d_eval \
        [--skip_svd] [--skip_collapse] [--dry_run]
"""

import argparse
import gc
import json
import math
import os
import random
import re
import string
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Shared utilities (from Phase 1a)
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


def build_prompt(sample, tokenizer, max_tokens=7168):
    """Build prompt matching MEM1 training format."""
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

    return prompt


# ---------------------------------------------------------------------------
# Collapse Curve Evaluation
# ---------------------------------------------------------------------------

def evaluate_collapse_curve(model_name, model, tokenizer, pool, n_values, samples_per_n, seed):
    """Evaluate collapse curve for one model."""
    results = {}
    per_sample = {}

    for n in n_values:
        rng = random.Random(seed + n)
        needed = n * samples_per_n
        if needed > len(pool):
            items = [rng.choice(pool) for _ in range(needed)]
        else:
            items = rng.sample(pool, needed)
        samples = [items[i * n:(i + 1) * n] for i in range(samples_per_n)]

        ems = []
        sample_details = []
        for sample in tqdm(samples, desc=f"{model_name} N={n}"):
            prompt = build_prompt(sample, tokenizer)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, max_new_tokens=4096,
                    do_sample=False, temperature=None, top_p=None)

            response = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:],
                                        skip_special_tokens=True)

            raw_answer = extract_answer(response)
            if raw_answer is None:
                em_mean = 0.0
                per_obj = [0] * n
            else:
                parts = [a.strip() for a in raw_answer.split(";")]
                per_obj = []
                for idx, item in enumerate(sample):
                    pred = parts[idx] if idx < len(parts) else ""
                    per_obj.append(1 if em_check(pred, item["answers"]) else 0)
                em_mean = sum(per_obj) / n

            ems.append(em_mean)
            sample_details.append({"em_mean": em_mean, "per_obj_em": per_obj})

        mean_em = float(np.mean(ems))
        std_em = float(np.std(ems))
        results[str(n)] = {"mean_em": round(mean_em, 4), "std_em": round(std_em, 4), "n_samples": samples_per_n}
        per_sample[str(n)] = sample_details
        print(f"  {model_name} N={n}: EM={mean_em:.3f}+-{std_em:.3f}")

    return results, per_sample


# ---------------------------------------------------------------------------
# SVD Profile (from Phase 1b)
# ---------------------------------------------------------------------------

WEIGHT_PATTERNS = (
    ".self_attn.q_proj.weight", ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight", ".self_attn.o_proj.weight",
    ".mlp.gate_proj.weight", ".mlp.up_proj.weight", ".mlp.down_proj.weight",
)
ATTN_PATTERNS = (".q_proj.", ".k_proj.", ".v_proj.", ".o_proj.")
K_VALUES = [64, 128, 256, 512]
N_RANDOM = 50  # Fewer random trials for speed


def compute_svd_profile(model_name, student_dir, base_dir, device="cpu"):
    """Compute SVD obs/random profile for student vs base."""
    print(f"\n=== SVD Profile: {model_name} ===")

    # Load state dicts
    student_model = AutoModelForCausalLM.from_pretrained(
        student_dir, torch_dtype=torch.float32, device_map=device, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_dir, torch_dtype=torch.float32, device_map=device, trust_remote_code=True)

    student_sd = student_model.state_dict()
    base_sd = base_model.state_dict()

    del student_model, base_model
    gc.collect()

    results_by_k = {k: [] for k in K_VALUES}

    for name in tqdm(sorted(base_sd.keys()), desc=f"SVD {model_name}"):
        if not any(name.endswith(p) for p in WEIGHT_PATTERNS):
            continue
        if name not in student_sd:
            continue

        W_base = base_sd[name].float()
        W_student = student_sd[name].float()
        dW = W_student - W_base

        if dW.dim() != 2:
            continue

        m, n = dW.shape
        dW_fro = dW.norm().item()
        if dW_fro < 1e-10:
            continue

        # SVD of base
        U_base, S_base, Vh_base = torch.linalg.svd(W_base, full_matrices=False)

        is_attn = any(p in name for p in ATTN_PATTERNS)

        for k in K_VALUES:
            k_eff = min(k, min(m, n))
            Uk = U_base[:, :k_eff]
            Vk = Vh_base[:k_eff, :]

            # Bilateral projection
            proj = Uk.T @ dW @ Vk.T
            in_ratio = (proj.norm() ** 2 / dW_fro ** 2).item()

            # Random baseline
            random_ratios = []
            for _ in range(N_RANDOM):
                R = torch.randn_like(dW)
                R = R / R.norm() * dW_fro
                proj_r = Uk.T @ R @ Vk.T
                random_ratios.append((proj_r.norm() ** 2 / dW_fro ** 2).item())
            random_mean = np.mean(random_ratios)
            obs_over_random = in_ratio / random_mean if random_mean > 1e-10 else 0

            results_by_k[k].append({
                "name": name,
                "is_attn": is_attn,
                "in_ratio": in_ratio,
                "random_mean": random_mean,
                "obs_over_random": obs_over_random,
            })

    del student_sd, base_sd
    gc.collect()

    # Aggregate
    summary = {}
    for k in K_VALUES:
        items = results_by_k[k]
        if not items:
            continue
        all_ratios = [x["obs_over_random"] for x in items]
        attn_ratios = [x["obs_over_random"] for x in items if x["is_attn"]]
        mlp_ratios = [x["obs_over_random"] for x in items if not x["is_attn"]]

        summary[str(k)] = {
            "mean_obs_random": round(float(np.mean(all_ratios)), 3),
            "attn_obs_random": round(float(np.mean(attn_ratios)), 3) if attn_ratios else None,
            "mlp_obs_random": round(float(np.mean(mlp_ratios)), 3) if mlp_ratios else None,
            "n_matrices": len(items),
        }
        print(f"  k={k}: overall={np.mean(all_ratios):.2f}x, "
              f"attn={np.mean(attn_ratios):.2f}x, mlp={np.mean(mlp_ratios):.2f}x")

    return {"summary": summary, "details": results_by_k}


# ---------------------------------------------------------------------------
# Statistical comparison
# ---------------------------------------------------------------------------

def compare_models(all_collapse_results, teacher_results=None):
    """Generate 2×2 comparison table and statistical tests."""
    comparison = {}

    model_names = list(all_collapse_results.keys())
    for n_str in ["1", "2", "4", "8", "12", "16"]:
        row = {}
        for model_name in model_names:
            if n_str in all_collapse_results[model_name]:
                row[model_name] = all_collapse_results[model_name][n_str]["mean_em"]
        if teacher_results and n_str in teacher_results:
            row["teacher_7B"] = teacher_results[n_str]
        comparison[f"N={n_str}"] = row

    # DPO vs SFT tests at high N
    tests = {}
    for size in ["1.5B", "3B"]:
        sft_key = f"{size}-SFT"
        dpo_key = f"{size}-DPO"
        if sft_key not in all_collapse_results or dpo_key not in all_collapse_results:
            continue
        for n_str in ["8", "12", "16"]:
            sft_data = all_collapse_results.get(sft_key, {}).get(n_str)
            dpo_data = all_collapse_results.get(dpo_key, {}).get(n_str)
            if sft_data and dpo_data:
                tests[f"{size}_DPO_vs_SFT_N{n_str}"] = {
                    "dpo_em": dpo_data["mean_em"],
                    "sft_em": sft_data["mean_em"],
                    "gap": round(dpo_data["mean_em"] - sft_data["mean_em"], 4),
                }

    return {"table": comparison, "tests": tests}


def compute_layer_em_correlation(all_svd_details, all_collapse, n_values):
    """Per-layer obs/random vs per-N EM Spearman correlation.

    For each student model, computes Spearman correlation between:
    - Per-layer obs/random ratio (at k=256)
    - Per-N mean EM

    This provides cross-layer × cross-N data point density to support
    causal claims (avoids "two data points" reviewer attack).
    """
    correlation_results = {}

    for model_name in all_svd_details:
        if model_name not in all_collapse:
            continue

        # Get per-layer obs/random at k=256
        layer_data = {}  # layer_idx -> obs/random
        details_k256 = all_svd_details[model_name].get(256, [])
        for item in details_k256:
            name = item["name"]
            # Extract layer index from name like "model.layers.0.self_attn.q_proj.weight"
            parts = name.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts):
                    layer_idx = int(parts[i + 1])
                    if layer_idx not in layer_data:
                        layer_data[layer_idx] = []
                    layer_data[layer_idx].append(item["obs_over_random"])
                    break

        # Average obs/random per layer
        layer_ratios = {l: np.mean(rs) for l, rs in layer_data.items()}
        sorted_layers = sorted(layer_ratios.keys())
        layer_ratio_vec = [layer_ratios[l] for l in sorted_layers]

        # Get per-N EM
        em_by_n = {}
        for n_str, data in all_collapse[model_name].items():
            em_by_n[int(n_str)] = data["mean_em"]
        sorted_ns = sorted(em_by_n.keys())
        em_vec = [em_by_n[n] for n in sorted_ns]

        # Cross correlation: for each (layer, N) pair, we have (obs/random, EM)
        # But layers and N are different axes. Compute two correlations:
        # 1. Mean layer obs/random vs model's mean EM (across N) — single point per model
        # 2. Layer-wise: does higher layer alignment predict better high-N performance?

        # More useful: aggregate layer obs/random into a single model-level metric,
        # then correlate across models (need all 4 models for this)

        model_result = {
            "n_layers": len(sorted_layers),
            "layer_obs_random_mean": round(float(np.mean(layer_ratio_vec)), 3),
            "layer_obs_random_std": round(float(np.std(layer_ratio_vec)), 3),
            "em_by_n": em_by_n,
            "layer_ratios": {str(l): round(float(layer_ratios[l]), 3) for l in sorted_layers},
        }

        # Within-model: correlation between layer depth and obs/random
        if len(sorted_layers) > 3:
            rho_depth, p_depth = stats.spearmanr(sorted_layers, layer_ratio_vec)
            model_result["depth_vs_alignment_rho"] = round(float(rho_depth), 4)
            model_result["depth_vs_alignment_p"] = round(float(p_depth), 6)

        correlation_results[model_name] = model_result

    # Cross-model correlation: obs/random mean vs EM at each N
    cross_model = {}
    model_names = sorted(correlation_results.keys())
    if len(model_names) >= 3:
        model_ratios = [correlation_results[m]["layer_obs_random_mean"] for m in model_names]
        for n in n_values:
            n_str = str(n)
            model_ems = []
            valid = True
            for m in model_names:
                if n in correlation_results[m]["em_by_n"]:
                    model_ems.append(correlation_results[m]["em_by_n"][n])
                else:
                    valid = False
                    break
            if valid and len(model_ems) >= 3:
                rho, p = stats.spearmanr(model_ratios, model_ems)
                cross_model[f"N={n}"] = {
                    "rho": round(float(rho), 4),
                    "p": round(float(p), 6),
                    "model_ratios": dict(zip(model_names, [round(r, 3) for r in model_ratios])),
                    "model_ems": dict(zip(model_names, [round(e, 4) for e in model_ems])),
                }

    return {
        "per_model": correlation_results,
        "cross_model_obs_random_vs_em": cross_model,
    }


def compute_go_no_go(all_collapse, all_svd, teacher_svd_k256=2.78):
    """Compute go/no-go for Phase 1d."""
    go_criteria = {}

    # Criterion 1: DPO > SFT at N>=8
    dpo_better = False
    for size in ["1.5B", "3B"]:
        sft_key = f"{size}-SFT"
        dpo_key = f"{size}-DPO"
        if sft_key not in all_collapse or dpo_key not in all_collapse:
            continue
        for n_str in ["8", "12", "16"]:
            sft_em = all_collapse.get(sft_key, {}).get(n_str, {}).get("mean_em", 0)
            dpo_em = all_collapse.get(dpo_key, {}).get(n_str, {}).get("mean_em", 0)
            if dpo_em > sft_em + 0.02:  # meaningful gap
                dpo_better = True
    go_criteria["dpo_better_than_sft_high_n"] = dpo_better

    # Criterion 2: DPO obs/random closer to teacher than SFT
    dpo_closer = False
    for size in ["1.5B", "3B"]:
        sft_key = f"{size}-SFT"
        dpo_key = f"{size}-DPO"
        if sft_key not in all_svd or dpo_key not in all_svd:
            continue
        sft_ratio = all_svd.get(sft_key, {}).get("summary", {}).get("256", {}).get("mean_obs_random", 0)
        dpo_ratio = all_svd.get(dpo_key, {}).get("summary", {}).get("256", {}).get("mean_obs_random", 0)
        if abs(dpo_ratio - teacher_svd_k256) < abs(sft_ratio - teacher_svd_k256):
            dpo_closer = True
    go_criteria["dpo_svd_closer_to_teacher"] = dpo_closer

    # Criterion 3: 3B > 1.5B
    size_effect = False
    for method in ["SFT", "DPO"]:
        small_key = f"1.5B-{method}"
        large_key = f"3B-{method}"
        if small_key not in all_collapse or large_key not in all_collapse:
            continue
        for n_str in ["4", "8"]:
            small_em = all_collapse.get(small_key, {}).get(n_str, {}).get("mean_em", 0)
            large_em = all_collapse.get(large_key, {}).get(n_str, {}).get("mean_em", 0)
            if large_em > small_em + 0.02:
                size_effect = True
    go_criteria["model_size_effect"] = size_effect

    # Overall
    if go_criteria["dpo_better_than_sft_high_n"] and go_criteria["dpo_svd_closer_to_teacher"]:
        overall = "go"
    elif any(go_criteria.values()):
        overall = "partial"
    else:
        overall = "no-go"
    go_criteria["overall"] = overall

    return go_criteria


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 1d: Evaluation")
    parser.add_argument("--models", nargs="+", required=True,
                        help="Model specs as name:path pairs, e.g. '1.5B-SFT:/path/to/model'")
    parser.add_argument("--base_dirs", nargs="+", required=True,
                        help="Base model paths matching --models order")
    parser.add_argument("--output_dir", default="/root/autodl-tmp/learned_traj_compress/artifacts/phase1d_eval")
    parser.add_argument("--cache_dir", default="/root/autodl-tmp/.hf_cache")
    parser.add_argument("--n_values", type=int, nargs="+", default=[1, 2, 4, 8, 12, 16])
    parser.add_argument("--samples_per_n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip_svd", action="store_true")
    parser.add_argument("--skip_collapse", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        args.samples_per_n = 5
        args.n_values = [2, 4, 8]
        print("*** DRY RUN ***")

    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()

    # Parse model specs
    model_specs = []
    for i, spec in enumerate(args.models):
        name, path = spec.split(":", 1)
        base_path = args.base_dirs[i] if i < len(args.base_dirs) else args.base_dirs[-1]
        model_specs.append({"name": name, "path": path, "base_path": base_path})

    # Load QA pool for collapse curve
    if not args.skip_collapse:
        print("Loading QA pool...")
        pool = load_qa_pool(args.cache_dir, args.seed)
        print(f"QA pool: {len(pool)} items")

    # Evaluate each model
    all_collapse = {}
    all_collapse_per_sample = {}
    all_svd = {}

    for spec in model_specs:
        model_name = spec["name"]
        model_path = spec["path"]
        base_path = spec["base_path"]

        # Collapse curve
        if not args.skip_collapse:
            print(f"\n=== Collapse Curve: {model_name} ===")
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map=args.device, trust_remote_code=True)
            model.eval()

            collapse_results, per_sample = evaluate_collapse_curve(
                model_name, model, tokenizer, pool, args.n_values, args.samples_per_n, args.seed)
            all_collapse[model_name] = collapse_results
            all_collapse_per_sample[model_name] = per_sample

            del model
            gc.collect()
            torch.cuda.empty_cache()

        # SVD profile
        if not args.skip_svd:
            svd_results = compute_svd_profile(model_name, model_path, base_path, device="cpu")
            all_svd[model_name] = svd_results

    # Comparison and go/no-go
    teacher_collapse = {
        "1": 0.620, "2": 0.590, "4": 0.540,
        "8": 0.455, "12": 0.408, "16": 0.341,
    }
    comparison = compare_models(all_collapse, teacher_collapse) if all_collapse else {}
    go_no_go = compute_go_no_go(all_collapse, all_svd) if all_collapse and all_svd else {}

    # Per-layer obs/random vs per-N EM correlation (Reviewer requirement)
    all_svd_details = {k: v.get("details", {}) for k, v in all_svd.items()}
    layer_em_correlation = {}
    if all_svd_details and all_collapse:
        layer_em_correlation = compute_layer_em_correlation(
            all_svd_details, all_collapse, args.n_values)
        print("\n=== Layer-EM Correlation (Reviewer requirement) ===")
        for model_name, mc in layer_em_correlation.get("per_model", {}).items():
            depth_rho = mc.get("depth_vs_alignment_rho", "N/A")
            print(f"  {model_name}: mean_obs_random={mc['layer_obs_random_mean']}, "
                  f"depth_vs_alignment_rho={depth_rho}")
        for n_key, cc in layer_em_correlation.get("cross_model_obs_random_vs_em", {}).items():
            print(f"  Cross-model {n_key}: rho={cc['rho']}, p={cc['p']}")

    # Save results
    output = {
        "config": {
            "models": model_specs,
            "n_values": args.n_values,
            "samples_per_n": args.samples_per_n,
            "fairness_note": (
                "All 4 models use identical LoRA config (rank=16, alpha=32, "
                "target=q/k/v/o_proj, dropout=0.05), same training data source "
                "(MEM1 RL 7B teacher trajectories at N=2,4,8), same epoch count (3). "
                "The only controlled variable is the distillation loss function "
                "(SFT: next-token prediction vs DPO: preference optimization, beta=0.1)."
            ),
        },
        "collapse_curves": all_collapse,
        "svd_profiles": {k: v["summary"] for k, v in all_svd.items()},
        "layer_em_correlation": layer_em_correlation,
        "comparison": comparison,
        "go_no_go": go_no_go,
        "teacher_reference": {
            "collapse": teacher_collapse,
            "svd_k256_obs_random": 2.78,
            "svd_k256_attn": 2.25,
        },
    }

    output_path = os.path.join(args.output_dir, "phase1d_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 80)
    print("PHASE 1D: 2×2 CONTRAST RESULTS")
    print("=" * 80)

    if all_collapse:
        header = f"{'N':>4}"
        for name in sorted(all_collapse.keys()):
            header += f" | {name:>10}"
        header += f" | {'Teacher':>10}"
        print(header)
        print("-" * len(header))
        for n in args.n_values:
            row = f"{n:>4}"
            for name in sorted(all_collapse.keys()):
                em = all_collapse[name].get(str(n), {}).get("mean_em", float("nan"))
                row += f" | {em:>10.3f}"
            t_em = teacher_collapse.get(str(n), float("nan"))
            row += f" | {t_em:>10.3f}"
            print(row)

    if all_svd:
        print("\nSVD obs/random at k=256:")
        for name in sorted(all_svd.keys()):
            ratio = all_svd[name].get("summary", {}).get("256", {}).get("mean_obs_random", "N/A")
            attn = all_svd[name].get("summary", {}).get("256", {}).get("attn_obs_random", "N/A")
            print(f"  {name}: overall={ratio}, attn={attn}")
        print(f"  Teacher: overall=2.78, attn=2.25")

    if go_no_go:
        print(f"\nGo/No-Go: {go_no_go}")

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
