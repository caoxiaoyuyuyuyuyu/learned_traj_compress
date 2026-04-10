"""
Canonical hash-based train/val/test split for exp_006 Phase 1d.

Single source of truth — imported by:
  - scripts/exp_006_stage1_sft.py
  - scripts/exp_006_stage1_dpo.py
  - scripts/re_extract_d024.py

D024: test_frac=0.25 (was 0.15), full prompt text (not [:500]).
"""
import hashlib
import json
import os


def assign_split(prompt_text, seed=42, test_frac=0.25, val_frac=0.10):
    """Deterministic hash-based split. Same prompt always maps to same split.

    Uses full prompt text (not [:500]) because the MEM1 prompt template
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


def split_data(data, prompt_key, seed=42):
    """Split data list into train/val/test by prompt-level hashing.

    Args:
        data: list of dicts
        prompt_key: key name containing the prompt text (e.g. "user_content" or "prompt")
        seed: hash seed (must match across SFT/DPO/eval)
    """
    splits = {"train": [], "val": [], "test": []}
    for item in data:
        text = item[prompt_key]
        s = assign_split(text, seed=seed)
        splits[s].append(item)
    return splits


def verify_split_counts(splits, data_dir, data_type, n_value=None):
    """Verify train/test counts against stats_d024.json. Abort if mismatch.

    Args:
        splits: dict with "train"/"val"/"test" lists
        data_dir: path to phase1d_v2_data dir containing stats_d024.json
        data_type: "sft" or "dpo"
        n_value: N value (required for per-N DPO, None for pooled SFT)
    """
    stats_path = os.path.join(data_dir, "stats_d024.json")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"stats_d024.json not found at {stats_path}. "
            f"Run re_extract_d024.py first to generate split ground truth."
        )

    with open(stats_path) as f:
        stats = json.load(f)

    actual_train = len(splits["train"])
    actual_test = len(splits["test"])

    if data_type == "sft" and n_value is None:
        # Pooled SFT: sum across all N
        expected_train = sum(
            stats["per_n"][str(n)]["n_sft_train"]
            for n in sorted(stats["per_n"].keys(), key=int)
        )
        expected_test = sum(
            stats["per_n"][str(n)]["n_sft_test"]
            for n in sorted(stats["per_n"].keys(), key=int)
        )
        label = "pooled SFT"
    elif data_type == "sft" and n_value is not None:
        n_key = str(n_value)
        expected_train = stats["per_n"][n_key]["n_sft_train"]
        expected_test = stats["per_n"][n_key]["n_sft_test"]
        label = f"SFT N={n_value}"
    elif data_type == "dpo":
        assert n_value is not None, "n_value required for DPO verification"
        n_key = str(n_value)
        expected_train = stats["per_n"][n_key]["n_dpo_train"]
        expected_test = stats["per_n"][n_key]["n_dpo_test"]
        label = f"DPO N={n_value}"
    else:
        raise ValueError(f"Unknown data_type: {data_type}")

    if actual_train != expected_train or actual_test != expected_test:
        raise RuntimeError(
            f"SPLIT MISMATCH ({label}): "
            f"train={actual_train} (expected {expected_train}), "
            f"test={actual_test} (expected {expected_test}). "
            f"This indicates a data leakage risk. "
            f"Check that assign_split uses full prompt text and test_frac=0.25. "
            f"Source of truth: {stats_path}"
        )

    print(f"[split-verify] {label} PASS: train={actual_train}, test={actual_test} "
          f"(matches stats_d024.json)")
