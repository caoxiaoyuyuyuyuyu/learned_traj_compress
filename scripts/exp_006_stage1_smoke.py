#!/usr/bin/env python3
"""
Exp 006 Stage 1 — Smoke Test (CPU, tiny model).

Purpose: empirically verify the three Reviewer fixes BEFORE running the real
training pipeline. Two independent smokes:

  1. SFT-init smoke
       - Loads `Qwen/Qwen2.5-0.5B-Instruct` on CPU.
       - Builds a 10-sample mini SFT dataset spanning N=2 and N=4 (so we can
         observe per-N train loss logging).
       - Runs PerNLossSFTTrainer for a handful of optimizer steps with one
         eval step in the middle.
       - Validates BUG 1 fix (SFTTrainer now instantiates without
         load_best_model_at_end conflict) and confirms `train/loss_N{2,4}`
         keys appear in trainer.state.log_history.

  2. Eval-EM smoke
       - Constructs synthetic <answer>...</answer> generations against the
         List[List[str]] gold structure used by raw_trajectories.
       - Calls score_item() directly (no model needed) — this is the function
         that contains the BUG 2 fix (gold list-of-lists, ; split, em_check).
       - Prints per-sample pred/gold/em_per_obj/em_full so the reviewer can
         see the EM metric is no longer systematically depressed.

This smoke is intentionally tiny and CPU-only so it can run on the AutoDL
machine while the Stage 0 datagen process is still using cuda:0.

Usage:
    CUDA_VISIBLE_DEVICES="" python scripts/exp_006_stage1_smoke.py
"""

import argparse
import json
import os
import sys
import tempfile
import traceback

# Force CPU before importing torch
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")

# Allow `from src.utils.metrics import em_check` from inside scripts/.
_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

# Add scripts/ so we can import the stage1 modules
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)


# ── Smoke 1: SFT init ────────────────────────────────────────────────

def smoke_sft_init(data_dir: str, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    print("=" * 60)
    print("SMOKE 1 — SFT init (verifies BUG 1 fix + per-N loss logging)")
    print("=" * 60)

    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig

    # Import the stage1 sft module directly so we share the same
    # PerNDataCollator + PerNLossSFTTrainer + format_for_sft logic that
    # the real training pipeline uses.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "exp_006_stage1_sft",
        os.path.join(_SCRIPTS_DIR, "exp_006_stage1_sft.py"),
    )
    sft_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sft_mod)

    # Build a tiny mixed-N SFT dataset from real data files
    raw = []
    for n in (2, 4):
        path = os.path.join(data_dir, f"sft_data_N{n}.json")
        with open(path) as f:
            items = json.load(f)
        for it in items[:5]:                # 5 from each N
            it["_n_value"] = n
            raw.append(it)
    print(f"[smoke1] loaded {len(raw)} mini SFT samples (N=2 + N=4)")

    print(f"[smoke1] loading tiny model on CPU: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,        # CPU → fp32
    )
    model.to("cpu")

    train_ds = sft_mod.format_for_sft(raw, tokenizer, max_seq_length=512)
    val_ds = sft_mod.format_for_sft(raw[:4], tokenizer, max_seq_length=512)
    print(f"[smoke1] formatted train={len(train_ds)} val={len(val_ds)}")

    out_dir = tempfile.mkdtemp(prefix="exp006_smoke_sft_")
    print(f"[smoke1] output_dir = {out_dir}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
    )

    training_args = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=1,
        max_steps=4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=2,
        save_strategy="steps",
        save_steps=2,
        save_total_limit=3,
        bf16=False,
        fp16=False,
        gradient_checkpointing=False,
        max_length=512,
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        report_to=[],                       # no wandb
        run_name="exp_006_smoke",
        seed=42,
        dataloader_pin_memory=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        use_cpu=True,
    )

    collator = sft_mod.PerNDataCollator(tokenizer)
    print("[smoke1] instantiating PerNLossSFTTrainer...")
    trainer = sft_mod.PerNLossSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_config,
        processing_class=tokenizer,
        data_collator=collator,
    )
    print("[smoke1] trainer instantiated OK (BUG 1 fix verified)")

    print("[smoke1] starting trainer.train() ...")
    trainer.train()
    print("[smoke1] trainer.train() returned OK")

    # Inspect log history for per-N loss keys
    log_hist = trainer.state.log_history
    print(f"[smoke1] log_history length = {len(log_hist)}")
    print("[smoke1] last 6 log_history entries:")
    for entry in log_hist[-6:]:
        print(f"    {entry}")

    has_n2 = any("train/loss_N2" in e for e in log_hist)
    has_n4 = any("train/loss_N4" in e for e in log_hist)
    has_eval = any(any(k.startswith("eval_") for k in e) for e in log_hist)
    print(f"[smoke1] saw train/loss_N2={has_n2}  train/loss_N4={has_n4}  eval_step={has_eval}")
    assert has_n2 or has_n4, "expected at least one per-N loss key in log history"
    assert has_eval, "expected at least one eval log entry"
    print("[smoke1] PASS")


# ── Smoke 2: Eval EM ─────────────────────────────────────────────────

def smoke_eval_em():
    print("=" * 60)
    print("SMOKE 2 — Eval EM (verifies BUG 2 fix: list-of-list gold + ; split)")
    print("=" * 60)

    # Import the stage1 eval module's score_item
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "exp_006_stage1_eval",
        os.path.join(_SCRIPTS_DIR, "exp_006_stage1_eval.py"),
    )
    eval_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_mod)

    # Synthetic gold matching raw_trajectories schema:
    #   gold_answers: List[List[str]]   (each inner list = equivalence set)
    cases = [
        # (description, generated_text, gold_answers, expected_em_full)
        (
            "N=2 fully correct (with article + casing variation)",
            "<think>thinking</think>\n<answer> Albert Einstein; The eiffel Tower </answer>",
            [["Albert Einstein", "Einstein"], ["Eiffel Tower", "tour eiffel"]],
            1.0,
        ),
        (
            "N=2 partial (q1 correct, q2 wrong)",
            "<think>...</think><answer>Einstein; Big Ben</answer>",
            [["Albert Einstein"], ["Eiffel Tower"]],
            0.0,
        ),
        (
            "N=4 fully correct via ; split",
            "<answer>paris; london; berlin; rome</answer>",
            [["Paris"], ["London"], ["Berlin"], ["Rome"]],
            1.0,
        ),
        (
            "N=4 with multi-gold equivalence (alt name in second slot)",
            "<answer>paris; usa; berlin; rome</answer>",
            [["Paris"], ["United States", "USA", "America"], ["Berlin"], ["Rome"]],
            1.0,
        ),
        (
            "N=4 missing trailing slot",
            "<answer>paris; london; berlin</answer>",
            [["Paris"], ["London"], ["Berlin"], ["Rome"]],
            0.0,
        ),
        (
            "N=8 truncated mid-answer (truncated=True)",
            "<answer>a; b; c; d; e; f; g; h</answer>",
            [["a"], ["b"], ["c"], ["d"], ["e"], ["f"], ["g"], ["h"]],
            1.0,
        ),
        (
            "no <answer> tag (extraction fail → all zero)",
            "<think>I have no idea</think>",
            [["x"], ["y"]],
            0.0,
        ),
    ]

    n_truncated = 0
    n_correct_em_full = 0
    print(f"\n[smoke2] running {len(cases)} synthetic eval cases\n")
    for i, (desc, gen, gold, expected_full) in enumerate(cases):
        # n_generated_tokens / max_new_tokens chosen so case 6 is "truncated"
        n_gen = 2048 if "truncated" in desc.lower() else 64
        max_new = 2048
        scored = eval_mod.score_item(gen, gold, n_gen, max_new)
        ok = (scored["em_full"] == expected_full)
        n_correct_em_full += int(ok)
        n_truncated += int(scored["truncated"])
        print(f"[case {i+1}] {desc}")
        print(f"    pred_raw     : {scored['pred_raw']!r}")
        print(f"    pred_parts   : {scored['pred_parts']}")
        print(f"    gold         : {gold}")
        print(f"    em_per_obj   : {scored['em_per_obj']}")
        print(f"    em_full      : {scored['em_full']:.0f}  (expected {expected_full:.0f}) {'OK' if ok else 'FAIL'}")
        print(f"    em_partial   : {scored['em_partial']:.3f}")
        print(f"    truncated    : {scored['truncated']}")
        print()

    trunc_rate = n_truncated / len(cases)
    print(f"[smoke2] em_full agreement vs expected: {n_correct_em_full}/{len(cases)}")
    print(f"[smoke2] truncated_rate: {trunc_rate:.1%}")

    assert n_correct_em_full == len(cases), (
        f"em_full mismatch on {len(cases) - n_correct_em_full} cases — BUG 2 fix incomplete"
    )
    print("[smoke2] PASS")


# ── Smoke 3: DPO init ───────────────────────────────────────────────

def smoke_dpo_init(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    """Verify DPOConfig + DPOTrainer instantiate and run 5 steps on CPU.

    Uses 8 synthetic DPO pairs (RAG-style prompts) with LoRA r=8.
    On 0.5B, loss ≈ log(2) ≈ 0.693 is expected (model can't distinguish
    chosen/rejected in 5 steps).  We just check it runs without error.
    """
    print("=" * 60)
    print("SMOKE 3 — DPO init (DPOConfig + DPOTrainer + 5 steps)")
    print("=" * 60)

    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    # --- Synthetic DPO data (8 pairs, RAG-style prompts) ---
    synthetic_pairs = []
    for i in range(8):
        synthetic_pairs.append({
            "prompt": (
                f"You will answer complex questions using search.\n"
                f"Answer the following questions: question_{i}_a; question_{i}_b\n"
                f"<information>\nDoc 1(Title: Topic {i}) Some relevant passage about topic {i}.\n</information>"
            ),
            "chosen": (
                f"<think>Based on the information, I can answer both questions.</think>"
                f"<answer>correct_answer_{i}_a; correct_answer_{i}_b</answer>"
            ),
            "rejected": (
                f"<think>I'm not sure about these questions.</think>"
                f"<answer>wrong_answer_{i}_a; wrong_answer_{i}_b</answer>"
            ),
        })

    dpo_ds = Dataset.from_list(synthetic_pairs)
    print(f"[smoke3] synthetic DPO dataset: {len(dpo_ds)} pairs")

    print(f"[smoke3] loading tiny model on CPU: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # CPU → fp32
    )
    model.to("cpu")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
    )

    out_dir = tempfile.mkdtemp(prefix="exp006_smoke_dpo_")
    print(f"[smoke3] output_dir = {out_dir}")

    # Detect bf16 support (CPU usually doesn't have it)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    dpo_config = DPOConfig(
        output_dir=out_dir,
        num_train_epochs=1,
        max_steps=5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        logging_steps=1,
        save_strategy="no",
        bf16=use_bf16,
        fp16=False,
        gradient_checkpointing=False,
        report_to=[],
        run_name="exp_006_dpo_smoke",
        seed=42,
        dataloader_pin_memory=False,
        use_cpu=not torch.cuda.is_available(),
        remove_unused_columns=False,
        max_length=512,
    )
    print("[smoke3] DPOConfig instantiated OK")

    print("[smoke3] instantiating DPOTrainer...")
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dpo_ds,
        peft_config=lora_config,
        processing_class=tokenizer,
    )
    print("[smoke3] DPOTrainer instantiated OK")

    print("[smoke3] starting trainer.train() (5 steps)...")
    trainer.train()
    print("[smoke3] trainer.train() returned OK")

    # Check log history
    log_hist = trainer.state.log_history
    print(f"[smoke3] log_history length = {len(log_hist)}")
    for entry in log_hist[-5:]:
        print(f"    {entry}")

    has_loss = any("loss" in e for e in log_hist)
    assert has_loss, "expected loss entries in log history"
    print("[smoke3] PASS")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="/root/autodl-tmp/learned_traj_compress/artifacts/phase1d_v2_data",
        help="Where sft_data_N{2,4}.json live",
    )
    parser.add_argument(
        "--model_name",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Tiny model for smoke tests (CPU)",
    )
    parser.add_argument(
        "--part",
        choices=["all", "sft", "eval", "dpo"],
        default="all",
        help="Which smoke to run: all, sft, eval, or dpo",
    )
    # Legacy flags (kept for backwards compat)
    parser.add_argument("--skip_sft", action="store_true",
                        help="Skip SFT smoke (e.g. if HF download blocked)")
    parser.add_argument("--skip_eval", action="store_true")
    args = parser.parse_args()

    # Determine which smokes to run
    run_eval = (args.part in ("all", "eval")) and not args.skip_eval
    run_sft = (args.part in ("all", "sft")) and not args.skip_sft
    run_dpo = args.part in ("all", "dpo")

    failures = []
    if run_eval:
        try:
            smoke_eval_em()
        except Exception:
            failures.append("smoke_eval_em")
            traceback.print_exc()
    if run_sft:
        try:
            smoke_sft_init(args.data_dir, args.model_name)
        except Exception:
            failures.append("smoke_sft_init")
            traceback.print_exc()
    if run_dpo:
        try:
            smoke_dpo_init(args.model_name)
        except Exception:
            failures.append("smoke_dpo_init")
            traceback.print_exc()

    print()
    print("=" * 60)
    if failures:
        print(f"SMOKE FAILED: {failures}")
        sys.exit(1)
    parts_run = [x for x, r in [("eval", run_eval), ("sft", run_sft), ("dpo", run_dpo)] if r]
    print(f"ALL SMOKES PASSED ({', '.join(parts_run)})")
    print("=" * 60)


if __name__ == "__main__":
    main()
