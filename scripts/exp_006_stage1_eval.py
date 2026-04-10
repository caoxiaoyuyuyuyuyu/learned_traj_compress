#!/usr/bin/env python3
"""
Exp 006 Stage 1 — Evaluation Script (D019).

Evaluates trained models on held-out test prompts. Loads model, generates
responses, extracts answers from <answer> tags, computes per-objective EM.

Evaluates: shared-SFT baseline + SFT+DPO_N2/N4/N8 on test prompts from each N.
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse the project's canonical EM utilities (matches MEM1 training reward
# semantics; resolves Reviewer BUG 2 — formerly eval.py used a divergent
# normalize_answer + treated multi-answer gold as a single string).
_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)
from src.utils.metrics import em_check  # noqa: E402


# ── Answer extraction (MEM1 compatible) ──────────────────────────────

def extract_answer(text):
    """Extract answer from last <answer> tag (MEM1 convention).

    Mirrors scripts/eval_collapse_curve.py:extract_answer — returns the inner
    text of the last <answer>...</answer> tag, or empty string if none.
    """
    matches = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL))
    if not matches:
        return ""
    return matches[-1].group(1).strip()


# ── Split key (pinned to user_content; no fallback chain) ────────────

def get_prompt_text(item):
    """Extract prompt text for split hashing.

    Pinned to `user_content` (raw trajectory source of truth). Fallback chain
    removed to prevent silent split drift if schema fields are added/renamed.
    """
    assert "user_content" in item, (
        f"missing 'user_content' key for eval split hashing; "
        f"got keys: {list(item.keys())[:5]}"
    )
    return item["user_content"]


# ── Model loading ────────────────────────────────────────────────────

def load_model(model_path, adapter_path=None, dpo_adapter_path=None):
    """Load model with optional LoRA adapter(s).

    For SFT-only: model_path + adapter_path
    For SFT+DPO: model_path + adapter_path (SFT, merged) + dpo_adapter_path
    """
    print(f"Loading base model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_path:
        print(f"Loading SFT adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        if dpo_adapter_path:
            print("Merging SFT adapter...")
            model = model.merge_and_unload()
            print(f"Loading DPO adapter: {dpo_adapter_path}")
            model = PeftModel.from_pretrained(model, dpo_adapter_path)

    model.eval()
    return model


def load_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# ── Test data loading ────────────────────────────────────────────────

def load_test_prompts(data_dir, split_seed=42):
    """Load test prompts from raw_trajectories, filtered by split.

    Returns dict: {N: [{user_content, gold_answers, questions, ...}, ...]}
    """
    import glob

    # Use canonical split (D024: full text hash + test_frac=0.25)
    _scripts_dir = os.path.dirname(__file__)
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    from utils.split import assign_split  # noqa: E402

    test_prompts = {}
    pattern = os.path.join(data_dir, "raw_trajectories_N*.json")
    for fpath in sorted(glob.glob(pattern)):
        with open(fpath) as f:
            raw = json.load(f)

        # raw is {stats, results}
        results = raw.get("results", raw) if isinstance(raw, dict) else raw
        if isinstance(raw, dict) and "results" in raw:
            results = raw["results"]

        n = raw.get("stats", {}).get("n_objectives")
        if n is None:
            # Try to extract from filename
            m = re.search(r"N(\d+)", os.path.basename(fpath))
            n = int(m.group(1)) if m else None

        if n is None:
            print(f"WARNING: Cannot determine N for {fpath}, skipping")
            continue

        test_items = []
        for item in results:
            # Hardened: get_prompt_text asserts `user_content` exists.
            prompt_text = get_prompt_text(item)
            if assign_split(prompt_text, seed=split_seed) == "test":
                test_items.append(item)

        test_prompts[n] = test_items
        print(f"  N={n}: {len(test_items)} test prompts (from {len(results)} total)")

    return test_prompts


# ── Generation & evaluation ──────────────────────────────────────────

def score_item(generated_text, gold_answers, n_generated_tokens, max_new_tokens):
    """Score a single generation against multi-objective gold.

    Resolves Reviewer BUG 2:
      (a) gold_answers is List[List[str]] (each sub-question has an equivalence
          set of acceptable answers); we route to em_check which accepts a list.
      (b) Predicted answer is split on `;` (MEM1 prompt convention `a1; a2; ...`);
          per-objective comparison aligns position i with gold_answers[i].

    Returns dict with: pred_raw, pred_parts, em_per_obj, em_full, em_partial,
    truncated.
    """
    pred_raw = extract_answer(generated_text)
    pred_parts = [p.strip() for p in pred_raw.split(";")] if pred_raw else []

    em_per_obj = []
    for i, gold_set in enumerate(gold_answers):
        pred_i = pred_parts[i] if i < len(pred_parts) else ""
        em_per_obj.append(int(em_check(pred_i, gold_set)))

    em_full = float(len(em_per_obj) > 0 and all(e == 1 for e in em_per_obj))
    em_partial = (sum(em_per_obj) / len(gold_answers)) if gold_answers else 0.0
    truncated = bool(n_generated_tokens >= max_new_tokens)

    return {
        "pred_raw": pred_raw,
        "pred_parts": pred_parts,
        "em_per_obj": em_per_obj,
        "em_full": em_full,
        "em_partial": em_partial,
        "truncated": truncated,
    }


@torch.no_grad()
def evaluate_model(model, tokenizer, test_items, max_new_tokens=2048, batch_size=1):
    """Generate responses and compute EM for each test item."""
    results = []
    for i, item in enumerate(test_items):
        user_content = get_prompt_text(item)
        gold_answers = item.get("gold_answers", [])
        questions = item.get("questions", [])

        # Format as chat
        messages = [{"role": "user", "content": user_content}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True,
                           max_length=4096).to(model.device)
        input_len = inputs["input_ids"].shape[1]

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )
        generated_ids = outputs[0][input_len:]
        n_generated = int(generated_ids.shape[0])
        generated = tokenizer.decode(generated_ids, skip_special_tokens=True)

        scored = score_item(generated, gold_answers, n_generated, max_new_tokens)

        results.append({
            "prompt_idx": item.get("prompt_idx", i),
            "n_objectives": item.get("n_objectives", len(gold_answers)),
            "gold_answers": gold_answers,
            "predicted_answer": scored["pred_raw"],
            "predicted_parts": scored["pred_parts"],
            "generated_text": generated[:500],
            "per_q_em": scored["em_per_obj"],
            "em_full": scored["em_full"],
            "em_partial": scored["em_partial"],
            "truncated": scored["truncated"],
            "n_generated_tokens": n_generated,
        })

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(test_items)}] "
                  f"em_full={scored['em_full']:.0f} "
                  f"em_partial={scored['em_partial']:.3f} "
                  f"trunc={scored['truncated']}")

    return results


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Exp 006 Stage 1: Evaluation")
    parser.add_argument("--model_path", default="/root/autodl-tmp/models/Qwen2.5-3B",
                        help="Base model path")
    parser.add_argument("--sft_adapter", default=None,
                        help="SFT LoRA adapter path (omit for base-only eval)")
    parser.add_argument("--dpo_adapter", default=None,
                        help="DPO LoRA adapter path (requires --sft_adapter)")
    parser.add_argument("--model_name", required=True,
                        help="Name for this model in results (e.g. 'sft_shared', 'dpo_N4')")
    parser.add_argument("--data_dir", default="/root/autodl-tmp/learned_traj_compress/artifacts/phase1d_v2_data")
    parser.add_argument("--output_dir", default="/root/autodl-tmp/learned_traj_compress/artifacts/exp_006_eval")
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--n_values", type=int, nargs="+", default=[2, 4, 8],
                        help="Which N values to evaluate on")
    args = parser.parse_args()

    t0 = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load test prompts
    print("=== Loading test prompts ===")
    test_prompts = load_test_prompts(args.data_dir, split_seed=args.split_seed)

    # Load model
    print(f"\n=== Loading model: {args.model_name} ===")
    tokenizer_path = args.dpo_adapter or args.sft_adapter or args.model_path
    tokenizer = load_tokenizer(tokenizer_path)
    model = load_model(args.model_path, args.sft_adapter, args.dpo_adapter)

    # Evaluate on each N
    all_results = {}
    summary = {}
    for n in args.n_values:
        if n not in test_prompts or len(test_prompts[n]) == 0:
            print(f"\n[SKIP] N={n}: no test prompts available")
            continue

        print(f"\n=== Evaluating on N={n} ({len(test_prompts[n])} prompts) ===")
        results = evaluate_model(model, tokenizer, test_prompts[n],
                                 max_new_tokens=args.max_new_tokens)

        em_full_list = [r["em_full"] for r in results]
        em_partial_list = [r["em_partial"] for r in results]
        truncated_list = [r["truncated"] for r in results]
        em_full_mean = sum(em_full_list) / len(em_full_list) if em_full_list else 0.0
        em_partial_mean = sum(em_partial_list) / len(em_partial_list) if em_partial_list else 0.0
        trunc_rate = sum(truncated_list) / len(truncated_list) if truncated_list else 0.0
        all_results[f"N{n}"] = results
        summary[f"N{n}"] = {
            "n_prompts": len(results),
            "em_full": round(em_full_mean, 4),
            "em_partial": round(em_partial_mean, 4),
            "truncated_rate": round(trunc_rate, 4),
        }
        print(f"  N={n} em_full={em_full_mean:.4f} "
              f"em_partial={em_partial_mean:.4f} "
              f"truncated={trunc_rate:.1%}")

    # Save results
    output = {
        "model_name": args.model_name,
        "model_path": args.model_path,
        "sft_adapter": args.sft_adapter,
        "dpo_adapter": args.dpo_adapter,
        "split_seed": args.split_seed,
        "summary": summary,
        "details": all_results,
        "elapsed_seconds": time.time() - t0,
    }

    result_path = os.path.join(args.output_dir, f"eval_{args.model_name}.json")
    with open(result_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"\n=== Evaluation complete in {elapsed:.0f}s ===")
    print(f"Results saved to: {result_path}")
    print("\nSummary:")
    for n_key, s in summary.items():
        print(f"  {n_key}: em_full={s['em_full']:.4f} em_partial={s['em_partial']:.4f} "
              f"truncated={s['truncated_rate']:.1%} ({s['n_prompts']} prompts)")


if __name__ == "__main__":
    main()
