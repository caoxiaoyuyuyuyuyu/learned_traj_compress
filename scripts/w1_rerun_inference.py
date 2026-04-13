#!/usr/bin/env python3
"""W1: Re-run SFT student inference on N=8 test set, saving FULL generated text.

The original exp_006_eval truncated generated_text to 500 chars.
This script saves the complete output for surface metric analysis.
"""

import argparse, json, os, re, sys, time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)
from scripts.utils.split import assign_split


def get_prompt_text(item):
    return item["user_content"]


def load_test_prompts_n8(data_dir, split_seed=42):
    fpath = os.path.join(data_dir, "raw_trajectories_N8.json")
    with open(fpath) as f:
        raw = json.load(f)
    results = raw["results"]
    test_items = [
        item for item in results
        if assign_split(get_prompt_text(item), seed=split_seed) == "test"
    ]
    print(f"N=8: {len(test_items)} test prompts (from {len(results)} total)", flush=True)
    return test_items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/root/autodl-tmp/models/Qwen2.5-3B")
    parser.add_argument("--sft_adapter", default="checkpoints/exp_006_sft_shared")
    parser.add_argument("--data_dir", default="artifacts/phase1d_v2_data")
    parser.add_argument("--output_path", default="artifacts/w1_surface_metrics/student_full_outputs.json")
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    t0 = time.time()

    # Load test prompts
    test_items = load_test_prompts_n8(args.data_dir, args.split_seed)

    # Load model
    print(f"Loading base model: {args.model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.sft_adapter, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    print(f"Loading & merging SFT adapter: {args.sft_adapter}", flush=True)
    model = PeftModel.from_pretrained(model, args.sft_adapter)
    model = model.merge_and_unload()
    model.eval()

    # Run inference
    results = []
    n_items = len(test_items)
    for batch_start in range(0, n_items, args.batch_size):
        batch_end = min(batch_start + args.batch_size, n_items)
        batch_items = test_items[batch_start:batch_end]

        input_texts = []
        for item in batch_items:
            msgs = [{"role": "user", "content": get_prompt_text(item)}]
            input_texts.append(tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True))

        tokenizer.padding_side = "left"
        inputs = tokenizer(input_texts, return_tensors="pt", padding=True,
                          truncation=True, max_length=4096).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens,
                do_sample=False, temperature=None, top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )

        for j, item in enumerate(batch_items):
            generated_ids = outputs[j][inputs["input_ids"].shape[1]:]
            generated_ids = generated_ids[generated_ids != tokenizer.pad_token_id]
            n_generated = len(generated_ids)
            generated = tokenizer.decode(generated_ids, skip_special_tokens=True)

            results.append({
                "prompt_idx": item.get("prompt_idx", batch_start + j),
                "n_objectives": item.get("n_objectives", 8),
                "gold_answers": item.get("gold_answers", []),
                "generated_text": generated,  # FULL text, no truncation
                "n_generated_tokens": n_generated,
                "truncated": n_generated >= args.max_new_tokens,
            })

        print(f"  [{batch_end}/{n_items}] "
              f"avg_len={sum(len(r['generated_text']) for r in results[-len(batch_items):])/len(batch_items):.0f} chars",
              flush=True)

    # Save
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    output = {
        "model_path": args.model_path,
        "sft_adapter": args.sft_adapter,
        "n": 8,
        "n_samples": len(results),
        "max_new_tokens": args.max_new_tokens,
        "results": results,
        "elapsed_seconds": time.time() - t0,
    }
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(results)} results to {args.output_path} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
