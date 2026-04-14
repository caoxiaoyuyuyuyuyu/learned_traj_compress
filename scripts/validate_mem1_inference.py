"""MEM1 Inference Validation — Phase 0 final check.

Validates:
1. Model loads without error (MEM1 7B RL + Qwen2.5-7B base)
2. Generation works with MEM1 RAG memory consolidation prompt format
3. Speed benchmarks: load time, tokens/s
4. RL vs base output comparison

Usage:
    python scripts/validate_mem1_inference.py \
        --mem1_path /root/autodl-tmp/models/MEM1 \
        --base_path /root/autodl-tmp/models/Qwen2.5-7B \
        --wandb_project learned_traj_compress
"""

import argparse
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# MEM1 RAG memory consolidation prompt format (from MEM1 paper)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based on given passages. "
    "For each question, read the passages carefully and provide a concise answer. "
    "Wrap your final answer in <answer> tags."
)

# Test examples: simple QA pairs to verify generation
TEST_EXAMPLES = [
    {
        "passages": (
            "Passage 1: The Eiffel Tower is a wrought-iron lattice tower on the "
            "Champ de Mars in Paris, France. It was constructed from 1887 to 1889 "
            "as the centerpiece of the 1889 World's Fair. The tower is 330 metres "
            "(1,083 ft) tall."
        ),
        "questions": ["How tall is the Eiffel Tower?"],
        "answers": [["330 metres", "330 meters", "1,083 ft", "330"]],
    },
    {
        "passages": (
            "Passage 1: Python was created by Guido van Rossum and first released "
            "in 1991. It emphasizes code readability with significant use of whitespace. "
            "Passage 2: PyTorch is an open-source machine learning library developed "
            "by Meta AI. It was released in 2016."
        ),
        "questions": [
            "Who created Python?",
            "When was PyTorch released?",
        ],
        "answers": [
            ["Guido van Rossum"],
            ["2016"],
        ],
    },
    {
        "passages": (
            "Passage 1: The mitochondria are membrane-bound organelles found in the "
            "cytoplasm of eukaryotic cells. They generate most of the cell's supply "
            "of adenosine triphosphate (ATP), used as a source of chemical energy. "
            "Passage 2: Chloroplasts are organelles that conduct photosynthesis in "
            "plant cells. They contain chlorophyll, which absorbs light energy."
        ),
        "questions": [
            "What do mitochondria generate?",
            "What pigment do chloroplasts contain?",
        ],
        "answers": [
            ["adenosine triphosphate", "ATP"],
            ["chlorophyll"],
        ],
    },
]


def build_prompt(example: dict) -> str:
    """Build MEM1-style prompt from example."""
    parts = [example["passages"], ""]
    for i, q in enumerate(example["questions"], 1):
        parts.append(f"Question {i}: {q}")
    parts.append("")
    parts.append(
        "Please read the passages and answer each question. "
        "Provide your final answers separated by semicolons in <answer> tags."
    )
    return "\n".join(parts)


def apply_chat_template(tokenizer, prompt: str) -> str:
    """Apply chat template if available, otherwise return raw prompt."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return f"{SYSTEM_PROMPT}\n\n{prompt}\n\nAssistant:"


def load_and_benchmark(model_path: str, label: str, dtype: str = "bfloat16"):
    """Load model and return (model, tokenizer, load_time_s)."""
    print(f"\n{'='*60}")
    print(f"Loading {label}: {model_path}")
    print(f"{'='*60}")

    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch_dtype, device_map="auto"
    )
    model.eval()
    load_time = time.time() - t0

    # Model info
    n_params = sum(p.numel() for p in model.parameters())
    mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0

    print(f"  Load time: {load_time:.1f}s")
    print(f"  Parameters: {n_params/1e9:.2f}B")
    print(f"  GPU memory: {mem_gb:.1f}GB")
    print(f"  Dtype: {next(model.parameters()).dtype}")

    return model, tokenizer, load_time, n_params, mem_gb


@torch.no_grad()
def generate_and_benchmark(model, tokenizer, prompt_text: str, max_new_tokens: int = 512):
    """Generate text and return (output_text, gen_time_s, n_tokens)."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    t0 = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen_time = time.time() - t0

    new_tokens = outputs[0][input_len:]
    n_tokens = len(new_tokens)
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return output_text, gen_time, n_tokens


def run_validation(args):
    """Main validation loop."""
    results = {}

    for model_path, label in [
        (args.mem1_path, "MEM1_RL_7B"),
        (args.base_path, "Qwen2.5-7B_base"),
    ]:
        if model_path is None:
            continue

        model, tokenizer, load_time, n_params, mem_gb = load_and_benchmark(
            model_path, label
        )

        model_results = {
            "load_time_s": load_time,
            "n_params": n_params,
            "gpu_mem_gb": mem_gb,
            "examples": [],
        }

        total_tokens = 0
        total_gen_time = 0

        for i, example in enumerate(TEST_EXAMPLES):
            prompt = build_prompt(example)
            prompt_text = apply_chat_template(tokenizer, prompt)

            output, gen_time, n_tokens = generate_and_benchmark(
                model, tokenizer, prompt_text
            )

            tokens_per_sec = n_tokens / gen_time if gen_time > 0 else 0
            total_tokens += n_tokens
            total_gen_time += gen_time

            print(f"\n--- Example {i+1} ({label}) ---")
            print(f"Questions: {example['questions']}")
            print(f"Output ({n_tokens} tokens, {tokens_per_sec:.1f} tok/s):")
            print(f"  {output[:500]}")
            print(f"Expected: {example['answers']}")

            model_results["examples"].append({
                "questions": example["questions"],
                "output": output,
                "expected": example["answers"],
                "n_tokens": n_tokens,
                "gen_time_s": gen_time,
                "tokens_per_sec": tokens_per_sec,
            })

        avg_tps = total_tokens / total_gen_time if total_gen_time > 0 else 0
        model_results["avg_tokens_per_sec"] = avg_tps
        model_results["total_tokens"] = total_tokens
        model_results["total_gen_time_s"] = total_gen_time

        print(f"\n{'='*60}")
        print(f"{label} Summary:")
        print(f"  Avg tokens/s: {avg_tps:.1f}")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Load time: {load_time:.1f}s")
        print(f"{'='*60}")

        results[label] = model_results

        # Free GPU memory before loading next model
        del model
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    # W&B logging
    if args.wandb_project:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name="phase0_inference_validation",
                config=vars(args),
            )
            for label, mr in results.items():
                prefix = label.lower().replace(".", "_")
                wandb.log({
                    f"{prefix}/load_time_s": mr["load_time_s"],
                    f"{prefix}/avg_tokens_per_sec": mr["avg_tokens_per_sec"],
                    f"{prefix}/gpu_mem_gb": mr["gpu_mem_gb"],
                    f"{prefix}/n_params": mr["n_params"],
                })
            wandb.finish()
            print("\nW&B logging complete.")
        except Exception as e:
            print(f"\nW&B logging failed: {e}")

    # Save results
    out_path = "/root/autodl-tmp/learned_traj_compress/inference_validation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mem1_path", default="/root/autodl-tmp/models/MEM1")
    parser.add_argument("--base_path", default="/root/autodl-tmp/models/Qwen2.5-7B")
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    run_validation(args)
