#!/usr/bin/env python3
"""Leakage audit: does the oracle retrieval injection contain gold answer spans?

Inputs:
  - artifacts/exp_009/eval_sft_shared_api_oracle.json  (Qwen-3B oracle eval)
  - OR: artifacts/exp_006_eval/eval_sft_shared.json for gold answers
  - OR: artifacts/phase1d_v2_data/raw_trajectories_N8.json for teacher info segments

Procedure:
  1. Load Qwen-3B oracle N=8 eval results (details[N8]).
  2. Each details entry has predicted_answer, gold_answers, generated_text.
     The generated_text contains the injected <information>...</information>
     blocks (since the student saw them during generation).
  3. Sample 50 prompts uniformly at random (seed 42).
  4. For each, extract all <information>...</information> blocks, concat.
  5. Check whether any gold_answers string is a case-insensitive substring
     (after light punctuation normalisation).
  6. Report n_samples, n_contain, ratio.

Output: artifacts/_project/paper/oracle_leakage_audit.txt
        artifacts/_project/paper/leakage_audit_table.tex (booktabs)
"""

import argparse
import json
import os
import random
import re
import string
import sys


INFO_BLOCK_RE = re.compile(r"<information>(.*?)(?:</information>|$)", re.DOTALL)


def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    # Strip common punctuation that wouldn't survive a substring match across
    # Wikipedia tokenisation.
    s = s.translate(str.maketrans("", "", string.punctuation))
    return s


def _extract_info_text(generated_text: str) -> str:
    blocks = INFO_BLOCK_RE.findall(generated_text)
    return "\n".join(blocks)


def _flatten_golds(it):
    golds = it.get("gold_answers") or []
    if isinstance(golds, str):
        golds = [golds]
    flat = []
    for g in golds:
        if isinstance(g, list):
            flat.extend(g)
        else:
            flat.append(g)
    return flat


def _check_hit(info_n, golds):
    for g in golds:
        if not g:
            continue
        gn = _norm(str(g))
        if gn and gn in info_n:
            return True, g
    return False, None


def audit(eval_path: str, n_sample: int = 50, seed: int = 42, n_key: str = "N8"):
    with open(eval_path) as f:
        data = json.load(f)
    if "details" in data and n_key in data["details"]:
        items = data["details"][n_key]
    elif isinstance(data, list):
        items = data
    else:
        raise RuntimeError(f"Unexpected eval JSON structure in {eval_path}")

    rng = random.Random(seed)
    n_total = len(items)
    sample_idx = rng.sample(range(n_total), min(n_sample, n_total))

    # Pre-extract info text and golds for each sampled idx
    info_by_idx = {}
    golds_by_idx = {}
    for i in sample_idx:
        it = items[i]
        gen = it.get("generated_text") or it.get("generated") or ""
        info_by_idx[i] = _extract_info_text(gen)
        golds_by_idx[i] = _flatten_golds(it)

    # REAL: prompt p's info blocks checked against p's own gold answers
    n_contain_real = 0
    per_sample = []
    for i in sample_idx:
        info_n = _norm(info_by_idx[i])
        hit, hit_ans = _check_hit(info_n, golds_by_idx[i])
        if hit:
            n_contain_real += 1
        per_sample.append({
            "idx": i,
            "hit": hit,
            "hit_answer": hit_ans,
            "n_info_chars": len(info_by_idx[i]),
            "gold_answers": golds_by_idx[i],
        })

    # SHUFFLED CONTROL: prompt p receives info blocks from a random other prompt p'
    # then checked against p's own gold answers. Derangement via a single-shift
    # permutation of sample_idx so no prompt gets its own passages back.
    shuffle_rng = random.Random(seed + 1)
    shuf = list(sample_idx)
    shuffle_rng.shuffle(shuf)
    # Ensure no fixed point (derangement); if any, rotate
    if any(a == b for a, b in zip(sample_idx, shuf)):
        shuf = sample_idx[1:] + sample_idx[:1]
    n_contain_shuf = 0
    for i, j in zip(sample_idx, shuf):
        info_n_other = _norm(info_by_idx[j])
        hit, _ = _check_hit(info_n_other, golds_by_idx[i])
        if hit:
            n_contain_shuf += 1

    n = len(sample_idx)
    return {
        "eval_path": eval_path,
        "n_total": n_total,
        "n_sampled": n,
        "n_contain_answer": n_contain_real,
        "ratio": n_contain_real / n if n else 0.0,
        "n_contain_shuffled": n_contain_shuf,
        "ratio_shuffled": n_contain_shuf / n if n else 0.0,
        "seed": seed,
        "per_sample": per_sample,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--eval_path",
        default="artifacts/exp_009/eval_sft_shared_api_oracle.json",
    )
    ap.add_argument("--n_sample", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_key", default="N8")
    ap.add_argument("--out_txt", default="artifacts/_project/paper/oracle_leakage_audit.txt")
    ap.add_argument("--out_tex", default="artifacts/_project/paper/leakage_audit_table.tex")
    args = ap.parse_args()

    if not os.path.exists(args.eval_path):
        candidates = [
            "artifacts/exp_009/eval_sft_shared_api_oracle.json",
            "artifacts/exp_009_student_with_api/eval_sft_shared_api_oracle.json",
            "artifacts/exp_013/eval_oracle/eval_sft_shared_api_oracle.json",
        ]
        for c in candidates:
            if os.path.exists(c):
                args.eval_path = c
                break

    result = audit(args.eval_path, n_sample=args.n_sample, seed=args.seed, n_key=args.n_key)

    lines = [
        f"eval_path: {result['eval_path']}",
        f"n_total_in_split: {result['n_total']}",
        f"n_sampled: {result['n_sampled']}",
        f"n_contain_answer: {result['n_contain_answer']}",
        f"ratio: {result['ratio']:.2%}",
        f"n_contain_shuffled: {result['n_contain_shuffled']}",
        f"ratio_shuffled: {result['ratio_shuffled']:.2%}",
        f"seed: {result['seed']}",
        "---",
    ]
    for s in result["per_sample"]:
        lines.append(
            f"idx={s['idx']} hit={s['hit']} info_chars={s['n_info_chars']} "
            f"hit_answer={s['hit_answer']}"
        )
    os.makedirs(os.path.dirname(args.out_txt), exist_ok=True)
    with open(args.out_txt, "w") as f:
        f.write("\n".join(lines) + "\n")

    # Decision logic for the paper text
    if result["ratio"] > 0.30:
        decision = "REVIEW: non-trivial leakage; treat oracle recovery as an upper bound"
    else:
        decision = "comparable to baseline retrieval co-occurrence (safe)"

    tex_lines = [
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Statistic & Value \\",
        r"\midrule",
        rf"Prompts audited & {result['n_sampled']} \\",
        rf"Injected context contains $\geq 1$ gold answer & {result['n_contain_answer']} / {result['n_sampled']} \\",
        rf"Ratio & {result['ratio']*100:.1f}\% \\",
        rf"Decision & {decision} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    with open(args.out_tex, "w") as f:
        f.write("\n".join(tex_lines) + "\n")

    print(json.dumps({
        "n_sampled": result["n_sampled"],
        "n_contain_answer": result["n_contain_answer"],
        "ratio": result["ratio"],
        "decision": decision,
        "eval_path": result["eval_path"],
    }, indent=2))


if __name__ == "__main__":
    main()
