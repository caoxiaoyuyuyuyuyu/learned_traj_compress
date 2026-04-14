#!/usr/bin/env python3
"""Analyze raw_trajectories.json from phase1d_generate_data.py.

Reports per-N:
- prompts, perfect responses, DPO pairs (matches built-in stats)
- mean total_em for Response A (greedy, idx 0) and Response B (sampled, idx 1)
- among DPO pairs: fraction where chosen is Response A (greedy wins rate)
- truncation rate: fraction of responses missing </answer> tag (proxy for hitting
  max_new_tokens=512 limit)

Usage:
    python scripts/analyze_phase1d_v2_raw.py \
        --raw /root/autodl-tmp/learned_traj_compress/artifacts/phase1d_v2_data/raw_trajectories.json
"""

import argparse
import json
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    args = ap.parse_args()

    with open(args.raw) as f:
        blob = json.load(f)

    results = blob["results"]
    print(f"Loaded {len(results)} prompts from {args.raw}")
    print(f"Built-in stats: {json.dumps(blob.get('stats', {}), indent=2)}")

    by_n = defaultdict(list)
    for r in results:
        by_n[r["n_objectives"]].append(r)

    for n in sorted(by_n):
        items = by_n[n]
        n_prompts = len(items)

        emA = [r["responses"][0]["total_em"] for r in items]
        emB = [r["responses"][1]["total_em"] for r in items]
        meanA = sum(emA) / n_prompts / n  # normalized to [0,1]
        meanB = sum(emB) / n_prompts / n

        perfectA = sum(1 for r in items if r["responses"][0]["is_perfect"])
        perfectB = sum(1 for r in items if r["responses"][1]["is_perfect"])

        # DPO pair analysis
        dpo_items = [r for r in items
                     if r["responses"][0]["total_em"] != r["responses"][1]["total_em"]]
        n_dpo = len(dpo_items)
        chosen_is_A = sum(1 for r in dpo_items
                          if r["responses"][0]["total_em"] > r["responses"][1]["total_em"])
        ties = n_prompts - n_dpo

        # Truncation: response missing </answer> tag
        truncA = sum(1 for r in items if "</answer>" not in r["responses"][0]["response"])
        truncB = sum(1 for r in items if "</answer>" not in r["responses"][1]["response"])

        print(f"\n=== N={n} ({n_prompts} prompts) ===")
        print(f"Response A (greedy):  mean_em={meanA:.3f}  perfect={perfectA}/{n_prompts}  "
              f"trunc(no </answer>)={truncA}/{n_prompts} ({100*truncA/n_prompts:.1f}%)")
        print(f"Response B (T=0.7):   mean_em={meanB:.3f}  perfect={perfectB}/{n_prompts}  "
              f"trunc(no </answer>)={truncB}/{n_prompts} ({100*truncB/n_prompts:.1f}%)")
        print(f"DPO pairs: {n_dpo}/{n_prompts}  ties: {ties}")
        if n_dpo > 0:
            print(f"  chosen=A (greedy wins): {chosen_is_A}/{n_dpo} "
                  f"({100*chosen_is_A/n_dpo:.1f}%)")
            print(f"  chosen=B (sampled wins): {n_dpo - chosen_is_A}/{n_dpo} "
                  f"({100*(n_dpo - chosen_is_A)/n_dpo:.1f}%)")


if __name__ == "__main__":
    main()
