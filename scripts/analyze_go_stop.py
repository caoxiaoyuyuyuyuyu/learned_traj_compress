#!/usr/bin/env python3
"""D032 Go/Stop decision analysis.

Reads eval JSONs from artifacts/exp_006_eval/, computes:
  - per-N em_partial (DPO vs SFT)
  - matched N=8 DPO - SFT difference
  - paired bootstrap p-value
  - Go/Stop matrix verdict

Usage:
    python scripts/analyze_go_stop.py --eval_dir artifacts/exp_006_eval/

Go/Stop matrix (D021+D022+D024):
  GO (strong):  matched N=8 DPO-SFT >= 5pp + p<0.05 + high-N trend stronger
  GO (soft):    matched N=8 DPO-SFT >= 5pp + p<0.05 + trend flat
  Finding:      matched ~ SFT but full-data > SFT
  Noise:        matched diff < 2pp
  Stop:         matched < SFT and N trend contradicts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np


def load_eval(eval_dir: Path, model_name: str) -> Optional[dict]:
    path = eval_dir / f"eval_{model_name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def get_per_prompt_em_partial(eval_data: dict, n: int) -> np.ndarray:
    """Extract per-prompt em_partial for a given N."""
    key = f"N{n}"
    details = eval_data.get("details", {}).get(key, [])
    return np.array([d["em_partial"] for d in details])


def paired_bootstrap_pvalue(a: np.ndarray, b: np.ndarray, n_boot: int = 10000,
                            seed: int = 42) -> float:
    """One-sided paired bootstrap test: H1: mean(a) > mean(b).

    Returns p-value = P(observed diff >= 0 under null).
    """
    assert len(a) == len(b), f"Length mismatch: {len(a)} vs {len(b)}"
    rng = np.random.RandomState(seed)
    observed_diff = np.mean(a) - np.mean(b)
    diffs = a - b
    count = 0
    for _ in range(n_boot):
        idx = rng.randint(0, len(diffs), size=len(diffs))
        boot_diff = np.mean(diffs[idx])
        # Under null, flip signs randomly
        signs = rng.choice([-1, 1], size=len(diffs))
        null_diff = np.mean(diffs[idx] * signs)
        if null_diff >= observed_diff:
            count += 1
    return count / n_boot


def analyze(eval_dir: Path, sft_name: str, dpo_names: Dict[int, str]):
    """Run full Go/Stop analysis."""
    print("=" * 60)
    print("D032 Go/Stop Analysis")
    print("=" * 60)

    # Load SFT baseline
    sft_data = load_eval(eval_dir, sft_name)
    if sft_data is None:
        print(f"[ERROR] SFT eval not found: eval_{sft_name}.json")
        sys.exit(1)

    # Load DPO models
    dpo_data = {}
    for n, name in dpo_names.items():
        data = load_eval(eval_dir, name)
        if data is None:
            print(f"[WARN] DPO N={n} eval not found: eval_{name}.json")
        else:
            dpo_data[n] = data

    if not dpo_data:
        print("[ERROR] No DPO eval results found. Cannot proceed.")
        sys.exit(1)

    # --- Per-N comparison table ---
    print("\n## Per-N em_partial (DPO vs SFT)")
    print(f"{'N':>3} | {'SFT':>8} | {'DPO':>8} | {'Diff':>8} | {'p-value':>8} | {'n_prompts':>9}")
    print("-" * 60)

    results = {}
    for n in [2, 4, 8]:
        sft_summary = sft_data.get("summary", {}).get(f"N{n}", {})
        sft_em = sft_summary.get("em_partial", None)
        sft_trunc = sft_summary.get("truncated_rate", None)
        n_prompts = sft_summary.get("n_prompts", "?")

        if n not in dpo_data:
            print(f"{n:>3} | {sft_em:>8.4f} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8} | {n_prompts:>9}")
            continue

        dpo_summary = dpo_data[n].get("summary", {}).get(f"N{n}", {})
        dpo_em = dpo_summary.get("em_partial", None)
        dpo_trunc = dpo_summary.get("truncated_rate", None)

        # Paired bootstrap on per-prompt em_partial
        sft_arr = get_per_prompt_em_partial(sft_data, n)
        dpo_arr = get_per_prompt_em_partial(dpo_data[n], n)

        if len(sft_arr) != len(dpo_arr):
            print(f"[WARN] N={n}: prompt count mismatch SFT={len(sft_arr)} vs DPO={len(dpo_arr)}")
            min_len = min(len(sft_arr), len(dpo_arr))
            sft_arr = sft_arr[:min_len]
            dpo_arr = dpo_arr[:min_len]

        diff = dpo_em - sft_em
        p_val = paired_bootstrap_pvalue(dpo_arr, sft_arr)

        results[n] = {
            "sft_em": sft_em, "dpo_em": dpo_em, "diff": diff,
            "p_value": p_val, "n_prompts": len(sft_arr),
            "sft_trunc": sft_trunc, "dpo_trunc": dpo_trunc,
        }

        sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
        print(f"{n:>3} | {sft_em:>8.4f} | {dpo_em:>8.4f} | {diff:>+8.4f} | {p_val:>8.4f} | {n_prompts:>9} {sig}")

    # --- Truncation rates ---
    print("\n## Truncated rates")
    print(f"{'N':>3} | {'SFT':>8} | {'DPO':>8}")
    print("-" * 30)
    for n in [2, 4, 8]:
        if n in results:
            print(f"{n:>3} | {results[n]['sft_trunc']:>7.1%} | {results[n]['dpo_trunc']:>7.1%}")

    # --- Go/Stop verdict ---
    print("\n" + "=" * 60)
    print("## Go/Stop Verdict")
    print("=" * 60)

    if 8 not in results:
        print("[BLOCKED] N=8 DPO results not available. Cannot make Go/Stop decision.")
        return

    r8 = results[8]
    diff_pp = r8["diff"] * 100  # percentage points
    p_val = r8["p_value"]

    print(f"\nCritical metric: matched N=8 DPO - SFT = {diff_pp:+.2f}pp")
    print(f"Bootstrap p-value: {p_val:.4f}")
    print(f"Threshold: >= +5pp AND p < 0.05")

    # Check N trend: does DPO advantage increase with N?
    diffs_by_n = {n: results[n]["diff"] for n in sorted(results.keys())}
    trend_increasing = all(
        diffs_by_n.get(n2, 0) >= diffs_by_n.get(n1, 0)
        for n1, n2 in zip([2, 4], [4, 8])
        if n1 in diffs_by_n and n2 in diffs_by_n
    )

    print(f"\nDPO-SFT diff by N: {', '.join(f'N={n}: {d*100:+.2f}pp' for n, d in sorted(diffs_by_n.items()))}")
    print(f"Trend increasing with N: {'YES' if trend_increasing else 'NO'}")

    # Decision matrix
    if diff_pp >= 5 and p_val < 0.05:
        if trend_increasing:
            verdict = "GO (strong)"
            action = "Proceed to Phase 1.5 + Phase 2"
        else:
            verdict = "GO (soft)"
            action = "Phase 1.5, but add student size ablation first"
    elif abs(diff_pp) < 2:
        verdict = "NOISE (D024)"
        action = "Stop fallback — difference within noise band"
    elif diff_pp < 0:
        if not trend_increasing:
            verdict = "STOP"
            action = "Stop fallback — DPO worse than SFT with contradicting trend"
        else:
            verdict = "FINDING: DPO needs more data"
            action = "Downgrade claim, consider epochs ablation"
    else:  # 2 <= diff < 5 or p >= 0.05
        verdict = "INCONCLUSIVE"
        action = "Consider epochs ablation (3ep max), then re-evaluate"

    print(f"\n>>> VERDICT: {verdict}")
    print(f">>> ACTION:  {action}")

    # --- Save machine-readable output ---
    output = {
        "per_n": {str(n): v for n, v in results.items()},
        "critical_n8_diff_pp": diff_pp,
        "critical_n8_p_value": p_val,
        "trend_increasing": trend_increasing,
        "verdict": verdict,
        "action": action,
    }
    out_path = eval_dir / "go_stop_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="D032 Go/Stop analysis")
    parser.add_argument("--eval_dir", type=str,
                        default="artifacts/exp_006_eval",
                        help="Directory containing eval_*.json files")
    parser.add_argument("--sft_name", type=str, default="sft_shared",
                        help="SFT model name in eval JSON filename")
    parser.add_argument("--dpo_prefix", type=str, default="merged",
                        help="DPO model name prefix (merged or dpo)")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        print(f"[ERROR] eval_dir not found: {eval_dir}")
        sys.exit(1)

    # DPO names: eval_{prefix}_N{n}.json
    dpo_names = {n: f"{args.dpo_prefix}_N{n}" for n in [2, 4, 8]}

    analyze(eval_dir, args.sft_name, dpo_names)


if __name__ == "__main__":
    main()
