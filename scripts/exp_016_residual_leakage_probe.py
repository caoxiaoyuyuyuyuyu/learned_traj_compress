#!/usr/bin/env python3
"""exp_016 P0 residual-leakage probe.

Reconstruct the model-visible text for each test prompt under each
condition and count how many prompts still contain unredacted
gold-answer substrings. Sanity: oracle_full (no redaction) should
return ~0.98 residual rate (Reviewer audit baseline).

Word-boundary anchored, case-insensitive, whitespace-flexible gold
pattern — identical rules to scripts/exp_016_passage_redaction.py.
"""
import json, os, re, sys
from pathlib import Path

_SCR = Path("/root/autodl-tmp/learned_traj_compress/scripts")
sys.path.insert(0, str(_SCR))
from exp_016_passage_redaction import (
    load_phase1d_v2_oracle, redact_info_block_tag, _flatten_golds,
    _build_gold_pattern, _clean_gold,
)

WORKDIR = Path("/root/autodl-tmp/learned_traj_compress/artifacts/exp_016_workdir")
ORACLE_SRC = Path("/root/autodl-tmp/learned_traj_compress/artifacts/phase1d_v2_data/raw_trajectories_N8.json")
N_TEST = 125
SPLIT_SEED = 42


def gold_in_text(text: str, golds) -> (bool, int, list):
    """Count how many cleaned golds have at least one word-boundary match
    in text (case-insensitive, whitespace-flexible). Skips golds <2 chars
    after cleaning. Returns (hit_flag, n_golds_hit, list_of_hit_golds)."""
    if not text:
        return False, 0, []
    hits = []
    for g in golds or []:
        g_clean = _clean_gold(g)
        if len(g_clean) < 2:
            continue
        pat = _build_gold_pattern(g)
        if pat is None:
            continue
        if pat.search(text):
            hits.append(g_clean)
    return (len(hits) > 0), len(hits), hits


def probe(condition, items, oracle_map_full):
    """For each item, build reconstructed model-visible text for the
    given condition and check for residual unredacted gold substrings."""
    mode = None
    if condition == "oracle_redacted_full":
        mode = "full"
    elif condition == "oracle_redacted_half":
        mode = "half"
    elif condition == "oracle_full":
        mode = None  # no redaction
    else:
        raise ValueError(condition)

    n_total = 0
    n_residual_blocks_only = 0   # gold in redacted blocks (excluding user_content)
    n_residual_including_q = 0   # gold anywhere in model-visible text
    residual_block_ids = []
    residual_multi_block_ids = []
    n_blocks_checked = 0
    n_redacted_total = 0

    for pid, it in enumerate(items):
        user_content = it["user_content"]
        blocks_full = oracle_map_full.get(user_content, [])
        golds = _flatten_golds(it["gold_answers"])

        if mode is None:
            redacted_blocks = list(blocks_full)
            red_here = 0
        else:
            redacted_blocks = []
            red_here = 0
            for b in blocks_full:
                rb, n = redact_info_block_tag(b, golds, mode=mode)
                redacted_blocks.append(rb)
                red_here += n
        n_redacted_total += red_here
        n_blocks_checked += len(redacted_blocks)

        blocks_concat = "\n".join(redacted_blocks)
        full_visible = user_content + "\n" + blocks_concat

        hit_blocks, n_hit_b, hits_b = gold_in_text(blocks_concat, golds)
        hit_full, n_hit_f, hits_f = gold_in_text(full_visible, golds)

        n_total += 1
        if hit_blocks:
            n_residual_blocks_only += 1
            residual_block_ids.append(pid)
            if n_hit_b > 1:
                residual_multi_block_ids.append(pid)
        if hit_full:
            n_residual_including_q += 1

    return {
        "condition": condition,
        "n_total": n_total,
        "n_blocks_checked": n_blocks_checked,
        "n_redacted_substitutions_applied": n_redacted_total,
        "n_residual_blocks_only": n_residual_blocks_only,
        "residual_rate_blocks_only": round(n_residual_blocks_only / n_total, 4) if n_total else 0.0,
        "n_residual_including_user_q": n_residual_including_q,
        "residual_rate_including_user_q": round(n_residual_including_q / n_total, 4) if n_total else 0.0,
        "n_multi_gold_residual": len(residual_multi_block_ids),
        "residual_prompt_ids_blocks_only_first20": residual_block_ids[:20],
    }


def count_in_stored_generated_text(eval_path):
    """Diagnose the 753 vs 1663 gap: count [REDACTED] in the JSON-stored
    `details[*].generated_text` (which is generated[:2000])."""
    d = json.loads(Path(eval_path).read_text())
    details = d.get("details", [])
    total = 0
    for det in details:
        gt = det.get("generated_text", "") or ""
        total += gt.count("[REDACTED]")
    return total


def main():
    items, oracle_map_full = load_phase1d_v2_oracle(str(ORACLE_SRC), split_seed=SPLIT_SEED)
    items = items[:N_TEST]
    print(f"[probe] loaded {len(items)} items, oracle_map_full has {len(oracle_map_full)} prompts")

    results = {}
    for cond in ("oracle_full", "oracle_redacted_full", "oracle_redacted_half"):
        print(f"[probe] running {cond}...", flush=True)
        results[cond] = probe(cond, items, oracle_map_full)

    # Diagnose the 753 vs 1663 gap
    diag = {}
    for cond, fname in (
        ("oracle_redacted_full", "oracle_redacted_full_eval.json"),
        ("oracle_redacted_half",  "oracle_redacted_half_eval.json"),
    ):
        path = WORKDIR / fname
        if path.exists():
            diag[cond] = {
                "n_redacted_in_stored_generated_text_2000char": count_in_stored_generated_text(path),
            }
    # Add note
    diag["_note"] = (
        "eval_sft_with_api.py:397 stores generated_text[:2000] which truncates "
        "the full rollout (~8 info blocks × ~200 chars + model output) to the "
        "first 2000 chars. This is the likely cause of the 1663→753 ~55% drop; "
        "not prompt-side model input truncation."
    )

    out = {
        "sanity_note": "oracle_full residual rate (blocks_only) should be >=0.90 if gold-matching logic is correct",
        "methodology": {
            "text_reconstructed_from": "items[pid].user_content + concat(oracle_map blocks after script's own redact_info_block_tag)",
            "gold_pattern": "scripts/exp_016_passage_redaction._build_gold_pattern (word-boundary, IGNORECASE, whitespace-flexible)",
            "note_on_model_input": "generate_with_retrieval tokenizes full context without max_length/truncation; model sees full text every round",
        },
        "probes": results,
        "redaction_serialization_diagnosis": diag,
    }
    out_path = WORKDIR / "residual_leakage_probe.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[probe] wrote {out_path}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
