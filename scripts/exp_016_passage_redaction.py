#!/usr/bin/env python3
"""exp_016 Passage-Redaction Oracle Replay (C1 blocking, EMNLP Findings rebuttal).

Reuses exp_009 / eval_sft_with_api.py oracle replay logic; the only delta is
that teacher <information> blocks are pre-redacted before injection.

4 conditions (Director APPROVED 2026-04-14, plan.md §2):
  - no_api               : baseline (no oracle injection); model hallucinates retrieval
  - oracle_full          : reproduce exp_009 baseline (sanity gate, no redaction)
  - oracle_redacted_full : 100% of gold-answer substring occurrences masked [REDACTED]
  - oracle_redacted_half : deterministic odd-indexed 50% redaction (dose-response)

Redaction logic follows v3.4 oracle_leakage_audit.py `_norm` spirit:
case-insensitive literal substring, whitespace-flexible.

Per-block statistics emitted per plan.md §3 Evidence requirements:
  {n_prompts, n_blocks_total, n_blocks_with_redaction,
   n_redacted_substitutions_total, n_redacted_per_block,
   fully_unredactable_prompt_ids}

Phases (--phase):
  0              : dry-run (no GPU), write redaction_stats.json + redaction_samples.json
  eval           : single-condition GPU eval (--condition <...>)
  aggregate      : paired bootstrap CI, 10k resamples, seed 42 (plan.md §5 Phase 3)
"""

import argparse
import glob
import json
import os
import random
import re
import string
import sys
import time

# ── Reused from v3.4 oracle_leakage_audit.py ─────────────────────────
INFO_BLOCK_RE = re.compile(r"<information>(.*?)(?:</information>|$)", re.DOTALL)


def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return s


def _flatten_golds(golds):
    if isinstance(golds, str):
        golds = [golds]
    flat = []
    for g in golds or []:
        if isinstance(g, list):
            flat.extend(g)
        else:
            flat.append(g)
    return [str(g) for g in flat if g]


# ── Redaction core ───────────────────────────────────────────────────
REDACTED_TOKEN = "[REDACTED]"


def _build_gold_pattern(gold: str) -> re.Pattern:
    """Build whitespace-flexible, case-insensitive pattern matching gold."""
    # Escape gold then make internal whitespace flexible (\s+) so tokenisation
    # variants ("Albert  Einstein", "Albert\nEinstein") still match. Case-
    # insensitive to approximate _norm's lower().
    parts = [re.escape(tok) for tok in re.split(r"\s+", gold.strip()) if tok]
    if not parts:
        return None
    pat = r"\s+".join(parts)
    return re.compile(pat, re.IGNORECASE)


def redact_block_content(block_text: str, gold_answers, mode: str = "full"):
    """Redact gold-answer substrings in a single info-block body.

    Args:
        block_text: body *without* <information>/</information> tags
        gold_answers: flat list of str
        mode: "full" (redact all) | "half" (redact odd-indexed, 1st/3rd/...)

    Returns:
        (redacted_text, n_substitutions_made)
    """
    # Sort golds longest-first so overlapping matches favour longest span.
    golds_sorted = sorted({g for g in gold_answers if g}, key=len, reverse=True)
    if not golds_sorted:
        return block_text, 0

    # Enumerate non-overlapping matches in document order across all golds.
    spans = []  # list of (start, end, gold)
    occupied = []  # intervals already claimed
    for gold in golds_sorted:
        pat = _build_gold_pattern(gold)
        if pat is None:
            continue
        for m in pat.finditer(block_text):
            s, e = m.span()
            # Skip if overlaps already-claimed region
            if any(not (e <= os_ or s >= oe_) for os_, oe_ in occupied):
                continue
            spans.append((s, e, gold))
            occupied.append((s, e))
    spans.sort(key=lambda x: x[0])

    if not spans:
        return block_text, 0

    if mode == "half":
        # Redact odd-indexed (1st, 3rd, 5th, ...) → positions 0, 2, 4 in 0-index.
        chosen = [sp for i, sp in enumerate(spans) if i % 2 == 0]
    else:
        chosen = spans

    if not chosen:
        return block_text, 0

    out = []
    cursor = 0
    for s, e, _ in chosen:
        out.append(block_text[cursor:s])
        out.append(REDACTED_TOKEN)
        cursor = e
    out.append(block_text[cursor:])
    return "".join(out), len(chosen)


def redact_info_block_tag(full_block: str, gold_answers, mode: str = "full"):
    """Redact a full `<information>...</information>` string, preserving tags.

    Returns (redacted_full_block, n_substitutions).
    """
    m = re.match(r"(<information>)(.*?)(</information>)", full_block, re.DOTALL)
    if not m:
        # Untagged fallback
        red, n = redact_block_content(full_block, gold_answers, mode=mode)
        return red, n
    open_tag, body, close_tag = m.group(1), m.group(2), m.group(3)
    red_body, n = redact_block_content(body, gold_answers, mode=mode)
    return open_tag + red_body + close_tag, n


# ── Oracle map loader (phase1d_v2 N=8) ───────────────────────────────
_INFO_BLOCK_KEEP_TAGS_RE = re.compile(r"<information>.*?</information>", re.DOTALL)


def _extract_information_blocks(text):
    return [m.group(0) for m in _INFO_BLOCK_KEEP_TAGS_RE.finditer(text)]


def _get_all_assistant_texts(item):
    if "responses" in item:
        return [r["full_assistant"] for r in item["responses"] if "full_assistant" in r]
    if "assistant_content" in item:
        return [item["assistant_content"]]
    return []


def _get_prompt_text(item):
    assert "user_content" in item
    return item["user_content"]


def load_phase1d_v2_oracle(raw_traj_path: str, split_seed: int = 42):
    """Load N=8 test-split items + per-prompt oracle info-block lists.

    Returns:
        items: list of dicts with keys user_content, gold_answers
        oracle_map: dict[user_content] -> list[str] (full <information> blocks)
    """
    _scripts_dir = os.path.abspath(os.path.dirname(__file__))
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    from utils.split import assign_split  # noqa: E402
    extract_information_blocks = _extract_information_blocks
    get_prompt_text = _get_prompt_text

    with open(raw_traj_path) as f:
        raw = json.load(f)
    results = raw["results"] if isinstance(raw, dict) and "results" in raw else raw

    items = []
    oracle_map = {}
    for item in results:
        prompt = get_prompt_text(item)
        if assign_split(prompt, seed=split_seed) != "test":
            continue
        # Pick best response by number of info blocks (matches exp_009)
        best_blocks = []
        for assistant_text in _get_all_assistant_texts(item):
            blocks = extract_information_blocks(assistant_text)
            if blocks and (not best_blocks or len(blocks) > len(best_blocks)):
                best_blocks = blocks
        if best_blocks and prompt not in oracle_map:
            oracle_map[prompt] = best_blocks
            items.append({"user_content": prompt, "gold_answers": item["gold_answers"]})
    return items, oracle_map


# ── Dry-run / Phase 0 ────────────────────────────────────────────────
def phase0_dryrun(raw_traj_path, output_dir, n_test=125, split_seed=42):
    items, oracle_map = load_phase1d_v2_oracle(raw_traj_path, split_seed=split_seed)
    items = items[:n_test] if n_test else items

    per_block = []          # [n_substitutions_full, ...]
    n_blocks_total = 0
    n_blocks_with_red = 0
    n_subs_total = 0
    n_prompts_zero = []     # prompt ids with no redactions applied across any block
    per_prompt_detail = []

    for pid, it in enumerate(items):
        blocks = oracle_map.get(it["user_content"], [])
        golds = _flatten_golds(it["gold_answers"])
        total_here = 0
        for b in blocks:
            _, n_full = redact_info_block_tag(b, golds, mode="full")
            n_blocks_total += 1
            if n_full > 0:
                n_blocks_with_red += 1
                per_block.append(n_full)
            else:
                per_block.append(0)
            n_subs_total += n_full
            total_here += n_full
        if total_here == 0:
            n_prompts_zero.append(pid)
        per_prompt_detail.append({
            "prompt_id": pid,
            "n_blocks": len(blocks),
            "n_substitutions_full": total_here,
        })

    mean_per_block = (
        sum(per_block) / len(per_block) if per_block else 0.0
    )
    stats = {
        "n_prompts": len(items),
        "n_blocks_total": n_blocks_total,
        "n_blocks_with_redaction": n_blocks_with_red,
        "n_redacted_substitutions_total": n_subs_total,
        "mean_redactions_per_block": round(mean_per_block, 3),
        "fully_unredactable_prompt_ids": n_prompts_zero,
        "per_prompt": per_prompt_detail,
    }
    os.makedirs(output_dir, exist_ok=True)
    stats_path = os.path.join(output_dir, "redaction_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[dryrun] wrote {stats_path}", flush=True)

    # 3 before/after samples — prefer prompts with ≥2 substitutions so half mode shows
    candidate_ids = [
        i for i, d in enumerate(per_prompt_detail) if d["n_substitutions_full"] >= 2
    ][:3]
    if len(candidate_ids) < 3:
        extra = [
            i for i, d in enumerate(per_prompt_detail) if d["n_substitutions_full"] >= 1
        ]
        for i in extra:
            if i not in candidate_ids:
                candidate_ids.append(i)
            if len(candidate_ids) >= 3:
                break

    samples = []
    for pid in candidate_ids[:5]:
        it = items[pid]
        golds = _flatten_golds(it["gold_answers"])
        blocks = oracle_map.get(it["user_content"], [])
        if not blocks:
            continue
        b0 = blocks[0]
        red_full, n_full = redact_info_block_tag(b0, golds, mode="full")
        red_half, n_half = redact_info_block_tag(b0, golds, mode="half")
        samples.append({
            "prompt_id": pid,
            "gold_answers": golds,
            "info_block_before": b0[:2000],
            "info_block_after_full_redact": red_full[:2000],
            "info_block_after_half_redact": red_half[:2000],
            "n_substitutions_full": n_full,
            "n_substitutions_half": n_half,
        })
    samples_path = os.path.join(output_dir, "redaction_samples.json")
    with open(samples_path, "w") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"[dryrun] wrote {samples_path} ({len(samples)} samples)", flush=True)

    print(json.dumps({
        "n_prompts": stats["n_prompts"],
        "n_blocks_total": stats["n_blocks_total"],
        "n_blocks_with_redaction": stats["n_blocks_with_redaction"],
        "n_substitutions_total": stats["n_redacted_substitutions_total"],
        "n_fully_unredactable": len(stats["fully_unredactable_prompt_ids"]),
    }, indent=2))
    return stats


# ── Non-English predictions check (exp_014 parity) ──────────────────
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]")


def has_cjk(s: str) -> bool:
    return bool(_CJK_RE.search(s or ""))


# ── Eval phase (single condition) ────────────────────────────────────
def run_eval_condition(args):
    # Lazy torch imports so dry-run path stays clean
    import torch  # noqa: F401
    from eval_sft_with_api import (
        load_model_merged, evaluate_with_retrieval,
        load_test_prompts,
    )
    from transformers import AutoTokenizer

    items, oracle_map_full = load_phase1d_v2_oracle(
        args.oracle_source, split_seed=args.split_seed
    )
    items = items[: args.n_test] if args.n_test else items

    # Pre-redact oracle_map if condition calls for it
    redaction_stats = None
    if args.condition in ("oracle_redacted_full", "oracle_redacted_half"):
        mode = "full" if args.condition == "oracle_redacted_full" else "half"
        new_map = {}
        per_block_counts = []
        n_blocks_total = 0
        n_blocks_with_red = 0
        n_subs_total = 0
        zero_ids = []
        for pid, it in enumerate(items):
            blocks = oracle_map_full.get(it["user_content"], [])
            golds = _flatten_golds(it["gold_answers"])
            new_blocks = []
            total_here = 0
            for b in blocks:
                red, n = redact_info_block_tag(b, golds, mode=mode)
                new_blocks.append(red)
                n_blocks_total += 1
                if n > 0:
                    n_blocks_with_red += 1
                n_subs_total += n
                per_block_counts.append(n)
                total_here += n
            new_map[it["user_content"]] = new_blocks
            if total_here == 0:
                zero_ids.append(pid)
        oracle_map = new_map
        redaction_stats = {
            "mode": mode,
            "n_prompts": len(items),
            "n_blocks_total": n_blocks_total,
            "n_blocks_with_redaction": n_blocks_with_red,
            "n_redacted_substitutions_total": n_subs_total,
            "n_redacted_per_block": per_block_counts,
            "fully_unredactable_prompt_ids": zero_ids,
        }
    elif args.condition == "oracle_full":
        oracle_map = oracle_map_full
    elif args.condition == "no_api":
        oracle_map = {}  # empty — retrieval_fn returns None always
    else:
        raise ValueError(f"Unknown condition: {args.condition}")

    # Load model
    print(f"[eval] loading tokenizer from {args.sft_adapter}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.sft_adapter, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model_merged(args.model_path, args.sft_adapter)

    # For no_api we pass retrieval_mode="oracle" with empty map; retrieval_fn
    # returns None for every round → matches "no injection" behaviour.
    print(
        f"[eval] condition={args.condition} n_items={len(items)} "
        f"redacted={'yes' if redaction_stats else 'no'}",
        flush=True,
    )
    t0 = time.time()
    results = evaluate_with_retrieval(
        model, tokenizer, items,
        retrieval_mode="oracle",
        oracle_map=oracle_map,
        max_new_tokens=args.max_new_tokens,
        max_search_rounds=args.max_search_rounds,
        save_partial_path=args.output_json + ".partial",
    )
    elapsed = time.time() - t0

    em_full = sum(r["em_full"] for r in results) / len(results)
    em_partial = sum(r["em_partial"] for r in results) / len(results)
    trunc = sum(r["truncated"] for r in results) / len(results)
    avg_s = sum(r["n_searches"] for r in results) / len(results)
    n_cjk = sum(1 for r in results if has_cjk(r.get("predicted_answer", "")))
    pct_cjk = n_cjk / len(results)

    out = {
        "condition": args.condition,
        "model_path": args.model_path,
        "sft_adapter": args.sft_adapter,
        "split": args.split,
        "n_prompts": len(results),
        "em_full": round(em_full, 4),
        "em_partial": round(em_partial, 4),
        "truncated_rate": round(trunc, 4),
        "avg_searches": round(avg_s, 2),
        "pct_non_english_preds": round(pct_cjk, 4),
        "elapsed_seconds": round(elapsed, 1),
        "details": results,
    }
    if redaction_stats is not None:
        out["redaction_stats"] = redaction_stats

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(
        f"[eval] done: em_partial={em_partial:.4f} em_full={em_full:.4f} "
        f"trunc={trunc:.2%} avg_searches={avg_s:.2f} pct_cjk={pct_cjk:.2%} "
        f"elapsed={elapsed:.0f}s",
        flush=True,
    )
    print(f"[eval] saved → {args.output_json}", flush=True)


# ── Aggregate phase: paired bootstrap ───────────────────────────────
def phase3_aggregate(workdir):
    def load_em_partial(path):
        with open(path) as f:
            d = json.load(f)
        return [r["em_partial"] for r in d["details"]], d.get("em_partial")

    paths = {
        "no_api": os.path.join(workdir, "no_api_eval.json"),
        "oracle_full": os.path.join(workdir, "oracle_full_eval.json"),
        "oracle_redacted_full": os.path.join(workdir, "oracle_redacted_full_eval.json"),
        "oracle_redacted_half": os.path.join(workdir, "oracle_redacted_half_eval.json"),
    }
    ems = {}
    agg = {}
    for k, p in paths.items():
        if os.path.exists(p):
            per, mean = load_em_partial(p)
            ems[k] = per
            agg[k] = mean

    if "no_api" not in ems:
        print("[aggregate] no_api_eval.json missing → skipping paired bootstrap")
        return

    base = ems["no_api"]
    n = len(base)
    rng = random.Random(42)
    n_resamples = 10000

    import numpy as np
    base_a = np.array(base)
    result = {"mean_em_partial": agg, "n": n, "n_resamples": n_resamples,
              "seed": 42, "deltas": {}}
    for cond in ["oracle_full", "oracle_redacted_full", "oracle_redacted_half"]:
        if cond not in ems:
            continue
        arr = np.array(ems[cond])
        diffs = arr - base_a
        # paired bootstrap
        point = diffs.mean()
        boots = np.empty(n_resamples)
        for i in range(n_resamples):
            idx = np.random.default_rng(42 + i).integers(0, n, n)
            boots[i] = diffs[idx].mean()
        lo, hi = np.percentile(boots, [2.5, 97.5])
        p_pos = float((boots > 0).mean())
        result["deltas"][cond] = {
            "delta_em_partial": round(float(point), 4),
            "ci95_lo": round(float(lo), 4),
            "ci95_hi": round(float(hi), 4),
            "p_boot_gt0": round(p_pos, 4),
        }

    out_path = os.path.join(workdir, "bootstrap_redaction_ci.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[aggregate] wrote {out_path}")
    print(json.dumps(result, indent=2))


# ── CLI ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["0", "eval", "aggregate"], default="eval")
    ap.add_argument("--condition",
                    choices=["no_api", "oracle_full", "oracle_redacted_full",
                             "oracle_redacted_half"],
                    default=None)
    ap.add_argument("--model_path", default="/root/autodl-tmp/models/Qwen2.5-3B")
    ap.add_argument("--sft_adapter",
                    default="checkpoints/exp_006_sft_shared",
                    help="LoRA adapter path")
    ap.add_argument("--oracle_source",
                    default="artifacts/phase1d_v2_data/raw_trajectories_N8.json")
    ap.add_argument("--output_json", default=None)
    ap.add_argument("--output_dir", default="artifacts/exp_016_workdir")
    ap.add_argument("--n_test", type=int, default=125)
    ap.add_argument("--split", default="phase1d_v2")
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    ap.add_argument("--max_search_rounds", type=int, default=10)
    args = ap.parse_args()

    if args.phase == "0":
        phase0_dryrun(args.oracle_source, args.output_dir,
                      n_test=args.n_test, split_seed=args.split_seed)
    elif args.phase == "eval":
        if not args.condition:
            raise SystemExit("--condition required for --phase eval")
        if not args.output_json:
            args.output_json = os.path.join(
                args.output_dir, f"{args.condition}_eval.json")
        run_eval_condition(args)
    elif args.phase == "aggregate":
        phase3_aggregate(args.output_dir)


if __name__ == "__main__":
    main()
