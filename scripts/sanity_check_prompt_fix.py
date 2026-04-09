#!/usr/bin/env python3
"""Sanity check: MEM1 prompt format fix.

Tests 10 N=1 HotpotQA samples with corrected MEM1 prompt format:
- No system prompt (all instructions in user message)
- MEM1 original multi-question prompt template
- Passages in <information> tag with Doc N(Title: T) format
- [HINT]You have 1 turn left.[/HINT]
- Assistant prefix to simulate completed search round
"""

import json
import os
import random
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.metrics import normalize_answer, em_check

os.environ["HF_HOME"] = "/root/autodl-tmp/.hf_cache"

# MEM1 original multi-question prompt template
MEM1_MULTI_PROMPT = """You will answer multiple complex questions using iterative reasoning, summarization, and web search.

At each step, you will see the questions, a cumulative summary of relevant information, the current search query, and search results (except in the first step, where only the questions are provided). Your task is to:

1. Perform reasoning and update a cumulative, concise summary within <think> ... </think>. This acts as persistent memory and must include all essential information from previous <think> and <information> tags.

2. Then choose one of the following actions:
   - If any question remains unanswered, issue a single query for one question inside <search> ... </search>. The query should consist of keywords or a short phrase. Only search one question at a time.
   - If all questions are answered, provide the final answers—separated by semicolons—within <answer> answer1; answer2; ... </answer>. The answers must be concise, contain only essential words, and avoid any explanations.

Important:
- Always follow this structure after <information> or the initial questions: <think> ... </think><search> ... </search> or <think> ... </think><answer> ... </answer>.
- Do not search multiple queries or questions simultaneously.

Answer the following questions: {questions}"""


def build_mem1_prompt(questions: list[str], passages: list[dict], tokenizer) -> str:
    """Build prompt matching MEM1 training format.

    Simulates the state where model has already searched and received results,
    now on the last turn and must answer.
    """
    # Questions in semicolon-separated format
    questions_str = "; ".join(questions)
    user_content = MEM1_MULTI_PROMPT.format(questions=questions_str)

    # Build <information> block with MEM1's Doc format
    info_lines = [f"[HINT]You have 1 turn left. You must answer the question now.[/HINT]"]
    for i, p in enumerate(passages, 1):
        info_lines.append(f"Doc {i}(Title: {p['title']}) {p['text']}")
    info_block = "\n".join(info_lines)

    # Assistant prefix: simulate one search round completed
    search_query = questions[0].split()[:5]  # first few words as search query
    assistant_prefix = (
        f"<think>Let me search for information about these questions.</think>"
        f"<search>{' '.join(search_query)}</search>\n\n"
        f"<information>\n{info_block}\n</information>\n"
    )

    # Build chat messages: no system prompt, user + partial assistant
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_prefix},
    ]

    # Apply chat template, then strip trailing <|im_end|> to let model continue
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    # Remove the last <|im_end|> so model continues from assistant turn
    if prompt.endswith("<|im_end|>\n"):
        prompt = prompt[: -len("<|im_end|>\n")]
    elif prompt.endswith("<|im_end|>"):
        prompt = prompt[: -len("<|im_end|>")]

    return prompt


def extract_answer(text: str) -> str | None:
    """Extract answer from MEM1-style output (after <think>, get <answer>)."""
    matches = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def main():
    seed = 42
    n_samples = 10
    random.seed(seed)

    # Load data
    print("Loading HotpotQA...")
    hotpot = load_dataset("hotpot_qa", "distractor", split="validation",
                          cache_dir="/root/autodl-tmp/.hf_cache")

    # Build QA pool with structured passage info
    pool = []
    for item in hotpot:
        sup_titles = set(item["supporting_facts"]["title"])
        passages = []
        for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
            if title in sup_titles:
                passages.append({"title": title, "text": " ".join(sentences)})
        if passages:
            pool.append({
                "question": item["question"],
                "answer": item["answer"],
                "passages": passages,
            })

    random.shuffle(pool)
    samples = pool[:n_samples]
    print(f"Selected {len(samples)} samples")

    # Load model
    print("Loading MEM1...")
    tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/models/MEM1")
    model = AutoModelForCausalLM.from_pretrained(
        "/root/autodl-tmp/models/MEM1",
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()

    results = []
    correct = 0

    for i, sample in enumerate(samples):
        prompt = build_mem1_prompt(
            questions=[sample["question"]],
            passages=sample["passages"],
            tokenizer=tokenizer,
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=512, do_sample=False,
                temperature=None, top_p=None,
            )
        elapsed = time.time() - t0
        n_tokens = out.shape[1] - input_len

        output = tokenizer.decode(out[0, input_len:], skip_special_tokens=True)
        answer = extract_answer(output)
        gold = sample["answer"]

        # For single question, answer may contain semicolons from model habit
        pred = answer.split(";")[0].strip() if answer else ""
        em = em_check(pred, [gold])
        correct += em

        results.append({
            "question": sample["question"],
            "gold": gold,
            "predicted": pred,
            "raw_answer_tag": answer,
            "em": em,
            "output_text": output[:500],
            "elapsed_s": round(elapsed, 2),
            "n_tokens": n_tokens,
        })

        status = "✓" if em else "✗"
        print(f"\n[{i+1}/{n_samples}] {status} EM={em}")
        print(f"  Q: {sample['question']}")
        print(f"  Gold: {gold}")
        print(f"  Pred: {pred}")
        print(f"  Raw output: {output[:200]}")

    em_score = correct / n_samples
    print(f"\n{'='*60}")
    print(f"RESULT: EM = {correct}/{n_samples} = {em_score:.1%}")
    print(f"Threshold: ≥0.5 → prompt is main cause, proceed with fix")
    print(f"           <0.3 → need further diagnosis")
    print(f"{'='*60}")

    # Save results
    output_path = Path("/root/autodl-tmp/learned_traj_compress/artifacts/sanity_check_prompt_fix.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"em": em_score, "correct": correct, "total": n_samples,
                    "results": results}, f, indent=2, ensure_ascii=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
