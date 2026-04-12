#!/usr/bin/env python3
"""Exp 009: Evaluate SFT student with real retrieval API.

Tests the behavioral mismatch hypothesis causally: give the SFT student a real
retrieval backend so its <search>query</search> tags produce real <information>
blocks instead of hallucinated ones. If EM rises significantly, mismatch is
confirmed as causal.

Two retrieval modes:
  - oracle: inject the exact <information> blocks from the teacher trajectory
            (upper bound — tests whether the strategy itself is sound)
  - bm25:   build a BM25 index from all training trajectory documents and
            retrieve against it (realistic — tests end-to-end with imperfect retrieval)

Generation is iterative (not batched) since we must intercept <search> tags:
  1. Generate until </search> or </answer> or max_tokens
  2. If </search>: extract query → retrieve → inject <information> → continue
  3. Repeat up to --max_search_rounds
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)
from src.utils.metrics import em_check  # noqa: E402


# ── Git SHA ──────────────────────────────────────────────────────────

def _git_commit_sha():
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)) + "/..",
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


# ── Answer / search extraction ───────────────────────────────────────

def extract_answer(text):
    matches = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL))
    return matches[-1].group(1).strip() if matches else ""


def extract_search_query(text):
    """Extract the last <search>...</search> query from text."""
    matches = list(re.finditer(r"<search>(.*?)</search>", text, re.DOTALL))
    return matches[-1].group(1).strip() if matches else None


def extract_information_blocks(text):
    """Extract all <information>...</information> blocks in order."""
    return [m.group(0) for m in re.finditer(
        r"<information>.*?</information>", text, re.DOTALL)]


# ── Split ────────────────────────────────────────────────────────────

def get_prompt_text(item):
    assert "user_content" in item
    return item["user_content"]


_scripts_dir = os.path.dirname(__file__)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from utils.split import assign_split  # noqa: E402


# ── BM25 corpus builder ─────────────────────────────────────────────

def build_bm25_corpus(data_dir):
    """Extract all documents from <information> blocks across all trajectories.

    Returns (corpus_texts, corpus_titles) where each entry is a single Doc.
    """
    doc_pattern = re.compile(
        r"Doc \d+\(Title: ([^)]+)\)\s*(.*?)(?=Doc \d+\(|$)", re.DOTALL
    )
    corpus_texts = []
    corpus_titles = []
    seen = set()

    for fpath in sorted(glob.glob(os.path.join(data_dir, "raw_trajectories_N*.json"))):
        with open(fpath) as f:
            raw = json.load(f)
        results = raw.get("results", raw) if isinstance(raw, dict) else raw
        if isinstance(raw, dict) and "results" in raw:
            results = raw["results"]

        for item in results:
            assistant = item.get("assistant_content", "")
            for info_block in extract_information_blocks(assistant):
                for m in doc_pattern.finditer(info_block):
                    title = m.group(1).strip()
                    body = m.group(2).strip()
                    doc_key = f"{title}|{body[:100]}"
                    if doc_key not in seen:
                        seen.add(doc_key)
                        corpus_texts.append(f"{title}. {body}")
                        corpus_titles.append(title)

    print(f"[BM25] Built corpus: {len(corpus_texts)} unique documents "
          f"from {data_dir}", flush=True)
    return corpus_texts, corpus_titles


def build_bm25_index(corpus_texts):
    """Build BM25 index from corpus texts."""
    from rank_bm25 import BM25Okapi
    tokenized = [doc.lower().split() for doc in corpus_texts]
    return BM25Okapi(tokenized)


def bm25_search(bm25, corpus_texts, corpus_titles, query, top_k=4):
    """Search BM25 index and format results as <information> block."""
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = scores.argsort()[-top_k:][::-1]

    docs = []
    for rank, idx in enumerate(top_indices, 1):
        title = corpus_titles[idx]
        # Extract body (remove title prefix)
        text = corpus_texts[idx]
        body = text[len(corpus_titles[idx]) + 2:]  # skip "title. "
        docs.append(f"Doc {rank}(Title: {title}) {body}")

    return "<information>\n" + "\n".join(docs) + "\n</information>"


# ── Oracle retrieval ─────────────────────────────────────────────────

def build_oracle_map(data_dir, split_seed=42):
    """Build a map from user_content -> ordered list of <information> blocks.

    Only includes test-split items.
    """
    oracle = {}
    for fpath in sorted(glob.glob(os.path.join(data_dir, "raw_trajectories_N*.json"))):
        with open(fpath) as f:
            raw = json.load(f)
        results = raw.get("results", raw) if isinstance(raw, dict) else raw
        if isinstance(raw, dict) and "results" in raw:
            results = raw["results"]

        for item in results:
            prompt = get_prompt_text(item)
            if assign_split(prompt, seed=split_seed) != "test":
                continue
            assistant = item.get("assistant_content", "")
            blocks = extract_information_blocks(assistant)
            if blocks:
                oracle[prompt] = blocks

    print(f"[Oracle] Built oracle map: {len(oracle)} test prompts with "
          f"information blocks", flush=True)
    return oracle


# ── Model loading ────────────────────────────────────────────────────

def load_model_merged(model_path, sft_adapter=None):
    print(f"Loading base: {model_path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    if sft_adapter:
        print(f"Loading & merging SFT: {sft_adapter}", flush=True)
        model = PeftModel.from_pretrained(model, sft_adapter)
        model = model.merge_and_unload()
    model.eval()
    return model


# ── Test data loading ────────────────────────────────────────────────

def load_test_prompts(data_dir, split_seed=42):
    test_prompts = {}
    for fpath in sorted(glob.glob(os.path.join(data_dir, "raw_trajectories_N*.json"))):
        with open(fpath) as f:
            raw = json.load(f)
        results = raw.get("results", raw) if isinstance(raw, dict) else raw
        if isinstance(raw, dict) and "results" in raw:
            results = raw["results"]
        fname = os.path.basename(fpath)
        m = re.search(r"N(\d+)", fname)
        if not m:
            continue
        n = int(m.group(1))
        test_items = [
            item for item in results
            if assign_split(get_prompt_text(item), seed=split_seed) == "test"
        ]
        test_prompts[n] = test_items
        print(f"  N={n}: {len(test_items)} test prompts", flush=True)
    return test_prompts


# ── Scoring ──────────────────────────────────────────────────────────

def score_item(generated_text, gold_answers, n_generated_tokens, max_new_tokens):
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
        "pred_raw": pred_raw, "pred_parts": pred_parts,
        "em_per_obj": em_per_obj, "em_full": em_full,
        "em_partial": em_partial, "truncated": truncated,
    }


# ── Iterative generation with retrieval ──────────────────────────────

@torch.no_grad()
def generate_with_retrieval(
    model, tokenizer, prompt_text, *,
    retrieval_fn, max_new_tokens=4096, max_search_rounds=10,
):
    """Generate iteratively, intercepting <search> tags for real retrieval.

    Args:
        retrieval_fn: callable(query, round_idx) -> str (information block)
            Returns None if no retrieval available for this round.
    """
    msgs = [{"role": "user", "content": prompt_text}]
    full_input = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True)

    total_generated = 0
    n_searches = 0
    current_text = full_input

    for round_idx in range(max_search_rounds + 1):
        remaining_tokens = max_new_tokens - total_generated
        if remaining_tokens <= 0:
            break

        inputs = tokenizer(current_text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        outputs = model.generate(
            **inputs,
            max_new_tokens=min(remaining_tokens, 2048),
            do_sample=False,
            stop_strings=["</search>", "</answer>"],
            tokenizer=tokenizer,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

        new_tokens = outputs[0][input_len:]
        n_new = len(new_tokens)
        total_generated += n_new
        generated_chunk = tokenizer.decode(new_tokens, skip_special_tokens=False)

        # stop_strings are not included in output tokens; re-check what stopped us
        # by looking at what the model was about to produce
        current_text += generated_chunk

        # Check if we stopped at </search>
        # transformers stop_strings strips the stop string from output,
        # so we need to check if the text ends with <search>...</search> pattern
        # Actually, stop_strings in HF may or may not include the stop string.
        # Let's check if the last content looks like it was cut at a search tag.

        # Heuristic: if text ends with content after <search> but no </search>,
        # the stop_string fired. Append </search> to complete the tag.
        last_search_open = current_text.rfind("<search>")
        last_search_close = current_text.rfind("</search>")
        if last_search_open > last_search_close:
            # Stop string fired on </search> — add it back
            current_text += "</search>"
            query = extract_search_query(current_text)
            if query:
                n_searches += 1
                info_block = retrieval_fn(query, round_idx)
                if info_block is not None:
                    current_text += "\n" + info_block + "\n"
                    continue
                else:
                    # No retrieval available, let model continue without injection
                    continue
            else:
                break
        else:
            # Either hit </answer> or max_tokens — done
            # Check if </answer> stop string fired
            last_answer_open = current_text.rfind("<answer>")
            last_answer_close = current_text.rfind("</answer>")
            if last_answer_open > last_answer_close:
                current_text += "</answer>"
            break

    # Extract only the generated part (after the initial prompt)
    generated_text = current_text[len(full_input):]
    return generated_text, total_generated, n_searches


# ── Main evaluation loop ─────────────────────────────────────────────

def evaluate_with_retrieval(
    model, tokenizer, test_items, *,
    retrieval_mode, oracle_map=None, bm25_index=None,
    bm25_corpus_texts=None, bm25_corpus_titles=None,
    max_new_tokens=4096, max_search_rounds=10,
    save_partial_path=None,
):
    results = []
    n_items = len(test_items)

    for idx, item in enumerate(test_items):
        prompt = get_prompt_text(item)
        gold_answers = item["gold_answers"]

        # Build retrieval function
        if retrieval_mode == "oracle":
            oracle_blocks = oracle_map.get(prompt, [])
            def retrieval_fn(query, round_idx, _blocks=oracle_blocks):
                if round_idx < len(_blocks):
                    return _blocks[round_idx]
                return None
        elif retrieval_mode == "bm25":
            def retrieval_fn(query, round_idx):
                return bm25_search(
                    bm25_index, bm25_corpus_texts, bm25_corpus_titles,
                    query, top_k=4)
        else:
            raise ValueError(f"Unknown retrieval_mode: {retrieval_mode}")

        generated, n_tokens, n_searches = generate_with_retrieval(
            model, tokenizer, prompt,
            retrieval_fn=retrieval_fn,
            max_new_tokens=max_new_tokens,
            max_search_rounds=max_search_rounds,
        )

        scored = score_item(generated, gold_answers, n_tokens, max_new_tokens)

        results.append({
            "gold_answers": gold_answers,
            "predicted_answer": scored["pred_raw"],
            "predicted_parts": scored["pred_parts"],
            "generated_text": generated[:2000],
            "per_q_em": scored["em_per_obj"],
            "em_full": scored["em_full"],
            "em_partial": scored["em_partial"],
            "truncated": scored["truncated"],
            "n_generated_tokens": n_tokens,
            "n_searches": n_searches,
        })

        # Progress
        em_so_far = sum(r["em_partial"] for r in results) / len(results)
        print(f"  [{idx+1}/{n_items}] em_partial={scored['em_partial']:.2f} "
              f"searches={n_searches} running_avg={em_so_far:.4f}",
              flush=True)

        # Partial save
        if save_partial_path and (idx + 1) % 10 == 0:
            with open(save_partial_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    return results


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Exp 009: Evaluate SFT student with real retrieval API")
    parser.add_argument("--model_path", required=True,
                        help="Base model path (e.g. Qwen2.5-3B)")
    parser.add_argument("--sft_adapter", required=True,
                        help="SFT LoRA adapter path")
    parser.add_argument("--retrieval_mode", required=True,
                        choices=["oracle", "bm25"],
                        help="oracle: use teacher's retrieval; "
                             "bm25: BM25 over training docs")
    parser.add_argument("--data_dir",
                        default="artifacts/phase1d_v2_data",
                        help="Directory with raw_trajectories_N*.json")
    parser.add_argument("--output_dir",
                        default="artifacts/exp_009_eval")
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--max_search_rounds", type=int, default=10)
    parser.add_argument("--n_values", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of test prompts per N (for debug)")
    args = parser.parse_args()

    t0 = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load test prompts ──
    print("=== Loading test prompts ===", flush=True)
    test_prompts = load_test_prompts(args.data_dir, split_seed=args.split_seed)

    # ── Build retrieval backend ──
    oracle_map = None
    bm25_index = None
    bm25_corpus_texts = None
    bm25_corpus_titles = None

    if args.retrieval_mode == "oracle":
        print("\n=== Building oracle retrieval map ===", flush=True)
        oracle_map = build_oracle_map(args.data_dir, split_seed=args.split_seed)
    elif args.retrieval_mode == "bm25":
        print("\n=== Building BM25 index ===", flush=True)
        bm25_corpus_texts, bm25_corpus_titles = build_bm25_corpus(args.data_dir)
        bm25_index = build_bm25_index(bm25_corpus_texts)

    # ── Load model ──
    print(f"\n=== Loading model (merged) ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.sft_adapter, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model_merged(args.model_path, args.sft_adapter)

    # ── Evaluate ──
    model_name = f"sft_shared_api_{args.retrieval_mode}"
    all_results = {}
    summary = {}

    for n in args.n_values:
        if n not in test_prompts or not test_prompts[n]:
            print(f"\n[SKIP] N={n}: no test prompts", flush=True)
            continue

        items = test_prompts[n]
        if args.limit:
            items = items[:args.limit]

        print(f"\n=== Evaluating N={n} ({len(items)} prompts, "
              f"mode={args.retrieval_mode}) ===", flush=True)

        partial_path = os.path.join(
            args.output_dir, f"eval_{model_name}_N{n}_partial.json")

        results = evaluate_with_retrieval(
            model, tokenizer, items,
            retrieval_mode=args.retrieval_mode,
            oracle_map=oracle_map,
            bm25_index=bm25_index,
            bm25_corpus_texts=bm25_corpus_texts,
            bm25_corpus_titles=bm25_corpus_titles,
            max_new_tokens=args.max_new_tokens,
            max_search_rounds=args.max_search_rounds,
            save_partial_path=partial_path,
        )

        em_full_mean = sum(r["em_full"] for r in results) / len(results)
        em_partial_mean = sum(r["em_partial"] for r in results) / len(results)
        trunc_rate = sum(r["truncated"] for r in results) / len(results)
        avg_searches = sum(r["n_searches"] for r in results) / len(results)

        all_results[f"N{n}"] = results
        summary[f"N{n}"] = {
            "n_prompts": len(results),
            "em_full": round(em_full_mean, 4),
            "em_partial": round(em_partial_mean, 4),
            "truncated_rate": round(trunc_rate, 4),
            "avg_searches": round(avg_searches, 2),
        }
        print(f"  N={n} em_full={em_full_mean:.4f} "
              f"em_partial={em_partial_mean:.4f} "
              f"trunc={trunc_rate:.1%} "
              f"avg_searches={avg_searches:.1f}", flush=True)

    # ── Save ──
    output = {
        "model_name": model_name,
        "model_path": args.model_path,
        "sft_adapter": args.sft_adapter,
        "retrieval_mode": args.retrieval_mode,
        "split_seed": args.split_seed,
        "metadata": {
            "max_new_tokens": args.max_new_tokens,
            "max_search_rounds": args.max_search_rounds,
            "stop_strings": ["</search>", "</answer>"],
            "do_sample": False,
            "commit": _git_commit_sha(),
        },
        "summary": summary,
        "details": all_results,
        "elapsed_seconds": time.time() - t0,
    }

    result_path = os.path.join(args.output_dir, f"eval_{model_name}.json")
    with open(result_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"\n=== Done in {elapsed:.0f}s. Saved: {result_path} ===", flush=True)
    for k, s in summary.items():
        print(f"  {k}: em_full={s['em_full']:.4f} "
              f"em_partial={s['em_partial']:.4f} "
              f"trunc={s['truncated_rate']:.1%} "
              f"avg_searches={s['avg_searches']:.1f}", flush=True)


if __name__ == "__main__":
    main()
