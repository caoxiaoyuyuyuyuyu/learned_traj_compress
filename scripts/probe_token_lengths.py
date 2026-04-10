#!/usr/bin/env python3
"""Probe token lengths of SFT/DPO data to validate max_length setting."""

import argparse
import json
import os
import numpy as np
from transformers import AutoTokenizer


def stats(lengths):
    a = np.array(lengths)
    return {
        "count": len(a), "p50": int(np.percentile(a, 50)),
        "p90": int(np.percentile(a, 90)), "p95": int(np.percentile(a, 95)),
        "p99": int(np.percentile(a, 99)), "max": int(a.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/root/autodl-tmp/learned_traj_compress/artifacts/phase1d_v2_data")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--n_subsample", type=int, default=0)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    report = {}

    for n in (2, 4, 8):
        lengths = {"dpo_chosen": [], "dpo_rejected": [], "sft": []}

        dpo_path = os.path.join(args.data_dir, f"dpo_data_N{n}.json")
        if os.path.exists(dpo_path):
            with open(dpo_path) as f:
                dpo = json.load(f)
            if args.n_subsample > 0:
                dpo = dpo[:args.n_subsample]
            for item in dpo:
                lengths["dpo_chosen"].append(len(tok.encode(item["prompt"] + item["chosen"])))
                lengths["dpo_rejected"].append(len(tok.encode(item["prompt"] + item["rejected"])))

        sft_path = os.path.join(args.data_dir, f"sft_data_N{n}.json")
        if os.path.exists(sft_path):
            with open(sft_path) as f:
                sft = json.load(f)
            if args.n_subsample > 0:
                sft = sft[:args.n_subsample]
            for item in sft:
                user = item.get("user_content") or item.get("prompt", "")
                asst = item.get("assistant_content") or item.get("response", "")
                lengths["sft"].append(len(tok.encode(user + asst)))

        n_report = {}
        print(f"\n{'='*60}")
        print(f"  N={n}")
        print(f"{'='*60}")
        print(f"  {'dataset':<16} {'count':>6} {'p50':>6} {'p90':>6} {'p95':>6} {'p99':>6} {'max':>6}  p99<=6144  max<=6144  max<=8192")
        for key, lens in lengths.items():
            if not lens:
                continue
            s = stats(lens)
            n_report[key] = s
            p99_ok = "YES" if s["p99"] <= 6144 else "NO"
            max_ok = "YES" if s["max"] <= 6144 else "NO"
            max8k = "YES" if s["max"] <= 8192 else "NO"
            print(f"  {key:<16} {s['count']:>6} {s['p50']:>6} {s['p90']:>6} {s['p95']:>6} {s['p99']:>6} {s['max']:>6}  {p99_ok:^9}  {max_ok:^9}  {max8k:^9}")
        report[f"N{n}"] = n_report

    # Global verdict
    all_p99 = [s["p99"] for nr in report.values() for s in nr.values()]
    all_max = [s["max"] for nr in report.values() for s in nr.values()]
    global_p99 = max(all_p99) if all_p99 else 0
    global_max = max(all_max) if all_max else 0

    print(f"\n{'='*60}")
    print(f"  GLOBAL: p99={global_p99}, max={global_max}")
    if global_p99 <= 6144 and global_max <= 6144:
        verdict = "6144 SAFE"
    elif global_max > 6144 and global_p99 <= 6144:
        verdict = f"6144 sufficient for p99, max={global_max} tail truncation acceptable"
    elif global_p99 > 6144 and global_p99 <= 8192:
        verdict = "RECOMMEND BUMP TO 8192"
    else:
        verdict = "REQUIRES PRE-TRUNCATION"
    print(f"  VERDICT: **{verdict}**")
    print(f"{'='*60}")

    report["_global"] = {"p99": global_p99, "max": global_max, "verdict": verdict}
    with open("/tmp/token_length_probe.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nJSON report written to /tmp/token_length_probe.json")


if __name__ == "__main__":
    main()
