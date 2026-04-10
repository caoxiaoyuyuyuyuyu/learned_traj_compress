#!/usr/bin/env python3
"""
D023: Re-extract SFT data from raw_trajectories_N{2,4,8}.json using
partial-credit criterion (total_em >= 1) instead of perfect-only (is_perfect).

Does NOT re-run RL teacher. Pure filter + dump.
"""
import argparse
import json
import shutil
from pathlib import Path


def extract_sft(results, n_value, criterion="partial"):
    """
    Extract SFT samples from raw trajectory results.

    Args:
        results: list of dicts from raw_trajectories_N{n}.json["results"]
        n_value: number of objectives (2, 4, or 8)
        criterion: 'partial' (total_em >= 1) or 'perfect' (total_em == n_value)
    Returns:
        list of SFT samples
    """
    sft = []
    for item in results:
        # Collect qualifying responses
        cands = []
        for r in item["responses"]:
            em = r["total_em"]
            if criterion == "partial" and em >= 1:
                cands.append(r)
            elif criterion == "perfect" and r["is_perfect"]:
                cands.append(r)

        if not cands:
            continue

        # Pick best response (highest total_em)
        cands.sort(key=lambda x: -x["total_em"])
        chosen = cands[0]

        sft.append({
            "user_content": item["user_content"],
            "assistant_content": chosen["full_assistant"],
            "n_objectives": n_value,
            "total_em": chosen["total_em"],
        })
    return sft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="phase1d_v2_data dir containing raw_trajectories_N{n}.json")
    ap.add_argument("--n_values", nargs="+", type=int, default=[2, 4, 8])
    ap.add_argument("--backup_suffix", default="perfect_only")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    summary = {"criterion": "partial-credit (D023)", "sft": {}, "dpo": {}}

    for n in args.n_values:
        raw_path = data_dir / f"raw_trajectories_N{n}.json"
        sft_path = data_dir / f"sft_data_N{n}.json"
        backup_path = data_dir / f"sft_data_N{n}.{args.backup_suffix}.json"

        assert raw_path.exists(), f"missing {raw_path}"

        # Backup old SFT file
        if sft_path.exists() and not backup_path.exists():
            shutil.copy(sft_path, backup_path)
            print(f"backup: {sft_path.name} -> {backup_path.name}")

        # Load raw
        raw = json.load(open(raw_path))
        results = raw["results"]
        old_perfect = raw["stats"].get("sft_samples", "?")
        print(f"\nN={n}: {len(results)} raw prompts, old perfect SFT={old_perfect}")

        # Extract with partial-credit
        sft = extract_sft(results, n_value=n, criterion="partial")
        with open(sft_path, "w") as f:
            json.dump(sft, f, indent=2, ensure_ascii=False)
        print(f"N={n}: new partial-credit SFT = {len(sft)}")
        summary["sft"][f"N{n}"] = len(sft)

        # Count existing DPO pairs
        dpo_path = data_dir / f"dpo_data_N{n}.json"
        if dpo_path.exists():
            dpo = json.load(open(dpo_path))
            summary["dpo"][f"N{n}"] = len(dpo)

    summary["matched_size_sft"] = min(summary["sft"].values()) if summary["sft"] else None
    summary["matched_size_dpo"] = min(summary["dpo"].values()) if summary["dpo"] else None

    stats_path = data_dir / "stats_d023.json"
    with open(stats_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{json.dumps(summary, indent=2)}")
    print(f"\nwrote {stats_path}")


if __name__ == "__main__":
    main()
