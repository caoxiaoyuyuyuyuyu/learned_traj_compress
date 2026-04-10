#!/usr/bin/env python3
"""
D024: Re-extract SFT + DPO data from raw_trajectories_N{2,4,8}.json with
hash-based split test_frac=0.25 (was 0.15).

Changes from D023:
  - test_frac 0.15 -> 0.25 (N=8 test prompts ~75 -> ~125, power 0.45 -> ~0.71)
  - Re-extract BOTH SFT (partial-credit, total_em>=1) and DPO pairs
  - Backup old files before overwriting

SFT criterion: partial-credit (total_em >= 1), pick best response per prompt.
DPO criterion: chosen_em > rejected_em, em_gap >= 0.1 (same as phase1d_generate_data.py).

Usage:
    python scripts/re_extract_d024.py \
        --data_dir /root/autodl-tmp/learned_traj_compress/artifacts/phase1d_v2_data
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


# ---------------------------------------------------------------------------
# Hash-based split (canonical implementation, test_frac=0.25)
# ---------------------------------------------------------------------------

def assign_split(prompt_text, seed=42, test_frac=0.25, val_frac=0.10):
    """Deterministic hash-based split. Same prompt always maps to same split.

    Note: uses full prompt text (not [:500]) because the MEM1 prompt template
    is >500 chars of shared prefix before the per-prompt questions appear.
    Truncating at 500 would hash all items identically.
    """
    h = hashlib.sha256(f"{seed}|{prompt_text}".encode()).hexdigest()
    v = int(h[:8], 16) / 0xFFFFFFFF
    if v < test_frac:
        return "test"
    elif v < test_frac + val_frac:
        return "val"
    return "train"


def get_prompt_text(item):
    """Extract prompt text for hashing (works for both SFT and DPO items)."""
    if "user_content" in item:
        return item["user_content"]
    if "prompt" in item:
        return item["prompt"]
    raise KeyError("item has neither 'user_content' nor 'prompt'")


# ---------------------------------------------------------------------------
# SFT extraction (D023 partial-credit criterion)
# ---------------------------------------------------------------------------

def extract_sft(results, n_value):
    """Extract SFT samples: total_em >= 1, pick best response per prompt."""
    sft = []
    for item in results:
        cands = [r for r in item["responses"] if r["total_em"] >= 1]
        if not cands:
            continue
        cands.sort(key=lambda x: -x["total_em"])
        chosen = cands[0]
        sft.append({
            "user_content": item["user_content"],
            "assistant_content": chosen["full_assistant"],
            "n_objectives": n_value,
            "total_em": chosen["total_em"],
        })
    return sft


# ---------------------------------------------------------------------------
# DPO extraction (same logic as phase1d_generate_data.py:build_sft_and_dpo_data)
# ---------------------------------------------------------------------------

def extract_dpo(results, n_value):
    """Extract DPO pairs: chosen_em > rejected_em, em_gap >= 0.1."""
    dpo = []
    for item in results:
        r0, r1 = item["responses"]
        if r0["total_em"] == r1["total_em"]:
            continue
        if r0["total_em"] > r1["total_em"]:
            chosen, rejected = r0, r1
        else:
            chosen, rejected = r1, r0

        n = item["n_objectives"]
        em_gap = (chosen["total_em"] - rejected["total_em"]) / n
        if em_gap >= 0.1:
            dpo.append({
                "prompt": item["user_content"],
                "chosen": chosen["full_assistant"],
                "rejected": rejected["full_assistant"],
                "n_objectives": n_value,
                "chosen_em": chosen["total_em"],
                "rejected_em": rejected["total_em"],
                "em_gap": float(em_gap),
            })
    return dpo


# ---------------------------------------------------------------------------
# Split + stats
# ---------------------------------------------------------------------------

def split_items(items, prompt_key="user_content"):
    """Split items into train/val/test using assign_split(test_frac=0.25)."""
    splits = {"train": [], "val": [], "test": []}
    for item in items:
        text = item.get(prompt_key) or item.get("prompt")
        s = assign_split(text)
        splits[s].append(item)
    return splits


def count_unique_prompts(items, prompt_key="user_content"):
    """Count unique prompts in a list of items."""
    texts = set()
    for item in items:
        t = item.get(prompt_key) or item.get("prompt")
        texts.add(t[:500])
    return len(texts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="D024: re-extract SFT+DPO with test_frac=0.25")
    ap.add_argument("--data_dir", required=True,
                    help="phase1d_v2_data dir with raw_trajectories_N{n}.json")
    ap.add_argument("--n_values", nargs="+", type=int, default=[2, 4, 8])
    ap.add_argument("--dry_run", action="store_true",
                    help="Print stats without writing files")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    stats = {
        "decision": "D024",
        "test_frac": 0.25,
        "split_method": "hash_based",
        "sft_inclusion": "partial_credit_total_em>=1",
        "per_n": {},
    }

    all_sft_train_counts = []
    all_dpo_train_counts = []

    for n in args.n_values:
        raw_path = data_dir / f"raw_trajectories_N{n}.json"
        assert raw_path.exists(), f"missing {raw_path}"

        raw = json.loads(raw_path.read_text())
        results = raw["results"]
        print(f"\nN={n}: {len(results)} raw prompts")

        # --- Extract ---
        sft = extract_sft(results, n)
        dpo = extract_dpo(results, n)
        print(f"  SFT (partial-credit): {len(sft)}")
        print(f"  DPO pairs: {len(dpo)}")

        # --- Split ---
        sft_splits = split_items(sft, prompt_key="user_content")
        dpo_splits = split_items(dpo, prompt_key="prompt")

        n_sft_train = len(sft_splits["train"])
        n_sft_test = len(sft_splits["test"])
        n_dpo_train = len(dpo_splits["train"])
        n_dpo_test = len(dpo_splits["test"])

        # Count unique test prompts for N=8
        n_test_prompts = count_unique_prompts(
            sft_splits["test"] + dpo_splits["test"], "user_content"
        ) if n == 8 else None

        print(f"  SFT split: train={n_sft_train} val={len(sft_splits['val'])} test={n_sft_test}")
        print(f"  DPO split: train={n_dpo_train} val={len(dpo_splits['val'])} test={n_dpo_test}")
        if n_test_prompts is not None:
            print(f"  N=8 unique test prompts: {n_test_prompts}")

        all_sft_train_counts.append(n_sft_train)
        all_dpo_train_counts.append(n_dpo_train)

        stats["per_n"][str(n)] = {
            "n_prompts": len(results),
            "n_sft_total": len(sft),
            "n_sft_train": n_sft_train,
            "n_sft_test": n_sft_test,
            "n_sft_val": len(sft_splits["val"]),
            "n_dpo_total": len(dpo),
            "n_dpo_train": n_dpo_train,
            "n_dpo_test": n_dpo_test,
            "n_dpo_val": len(dpo_splits["val"]),
        }
        if n_test_prompts is not None:
            stats["per_n"][str(n)]["n_test_prompts_unique"] = n_test_prompts

        if args.dry_run:
            continue

        # --- Backup old files ---
        sft_path = data_dir / f"sft_data_N{n}.json"
        dpo_path = data_dir / f"dpo_data_N{n}.json"

        sft_backup = data_dir / f"sft_data_N{n}.d023_split015.json"
        dpo_backup = data_dir / f"dpo_data_N{n}.split015.json"

        if sft_path.exists() and not sft_backup.exists():
            shutil.copy(sft_path, sft_backup)
            print(f"  backup: {sft_path.name} -> {sft_backup.name}")

        if dpo_path.exists() and not dpo_backup.exists():
            shutil.copy(dpo_path, dpo_backup)
            print(f"  backup: {dpo_path.name} -> {dpo_backup.name}")

        # --- Write new files ---
        sft_path.write_text(json.dumps(sft, indent=2, ensure_ascii=False))
        dpo_path.write_text(json.dumps(dpo, indent=2, ensure_ascii=False))
        print(f"  wrote {sft_path.name} ({len(sft)} items)")
        print(f"  wrote {dpo_path.name} ({len(dpo)} items)")

    # --- Global stats ---
    stats["matched_size_sft"] = min(all_sft_train_counts) if all_sft_train_counts else 0
    stats["matched_size_dpo"] = min(all_dpo_train_counts) if all_dpo_train_counts else 0
    if "8" in stats["per_n"]:
        stats["n8_test_prompts"] = stats["per_n"]["8"].get("n_test_prompts_unique", "?")
    stats["d024_re_extract_commit"] = "TBD"

    print(f"\n{'='*60}")
    print(f"matched_size_sft (min train): {stats['matched_size_sft']}")
    print(f"matched_size_dpo (min train): {stats['matched_size_dpo']}")
    print(f"n8_test_prompts: {stats.get('n8_test_prompts', '?')}")

    if not args.dry_run:
        stats_path = data_dir / "stats_d024.json"
        stats_path.write_text(json.dumps(stats, indent=2))
        print(f"\nwrote {stats_path}")

        # Delete pooled sft_data.json if it exists (stale, from phase1d_generate_data.py)
        pooled = data_dir / "sft_data.json"
        if pooled.exists():
            pooled.unlink()
            print(f"deleted stale pooled {pooled.name}")

    print(f"\n{json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()
