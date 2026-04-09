#!/usr/bin/env python3
"""
Phase 1c: Per-objective Probing — Principal Direction Activation Shift

Verify that RL's supra-random alignment with base principal directions
(2-6x random, from Phase 1b corrected) encodes multi-objective capability.

Hypothesis (revised after exp_003b):
"RL在base主方向上的精细调整编码了多目标consolidation能力"

Method:
1. Run MEM1 RL + Qwen2.5-7B base on Phase 1a HotpotQA samples → collect
   hidden states at selected layers (last token before generation)
2. Compute activation delta: Δh = h_RL - h_base
3. Compute base activation principal subspace via SVD (k=64 and k=256)
4. Project Δh into: base principal subspace / random subspace (control)
5. Train linear probes on each projection to predict per-objective EM
6. If principal Δh probe >> random Δh probe → RL's fine-tuning along
   principal directions encodes task capability

Usage:
    python scripts/probe_orthogonal_subspace.py [--dry_run]
"""

import argparse
import gc
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Reuse Phase 1a data loading and prompt construction
# ---------------------------------------------------------------------------

MAX_INPUT_TOKENS = 4096

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


def load_qa_pool(cache_dir, seed):
    rng = random.Random(seed)
    pool = []
    print("Loading HotpotQA validation split...")
    hotpot = load_dataset("hotpot_qa", "distractor", split="validation", cache_dir=cache_dir)
    for item in hotpot:
        sup_titles = set(item["supporting_facts"]["title"])
        passages = []
        for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
            if title in sup_titles:
                passages.append({"title": title, "text": " ".join(sentences)})
        if passages:
            pool.append({
                "question": item["question"],
                "answers": [item["answer"]],
                "passages": passages,
            })
    rng.shuffle(pool)
    print(f"QA pool size: {len(pool)}")
    return pool


def build_samples(pool, n, num_samples, seed):
    rng = random.Random(seed + n)
    needed = n * num_samples
    if needed > len(pool):
        items = [rng.choice(pool) for _ in range(needed)]
    else:
        items = rng.sample(pool, needed)
    return [items[i * n:(i + 1) * n] for i in range(num_samples)]


def build_prompt(sample, tokenizer, max_tokens=MAX_INPUT_TOKENS):
    questions = [item["question"] for item in sample]
    questions_str = "; ".join(questions)
    user_content = MEM1_MULTI_PROMPT.format(questions=questions_str)

    all_passages = []
    for item in sample:
        for p in item["passages"]:
            all_passages.append(p)

    info_lines = ["[HINT]You have 1 turn left. You must answer the question now.[/HINT]"]
    for i, p in enumerate(all_passages, 1):
        info_lines.append(f"Doc {i}(Title: {p['title']}) {p['text']}")
    info_block = "\n".join(info_lines)

    search_query = questions[0].split()[:5]
    assistant_prefix = (
        f"<think>Let me search for information about these questions.</think>"
        f"<search>{' '.join(search_query)}</search>\n\n"
        f"<information>\n{info_block}\n</information>\n"
    )

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_prefix},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    if prompt.endswith("<|im_end|>\n"):
        prompt = prompt[:-len("<|im_end|>\n")]
    elif prompt.endswith("<|im_end|>"):
        prompt = prompt[:-len("<|im_end|>")]

    token_count = len(tokenizer.encode(prompt))
    if token_count > max_tokens:
        overflow = token_count - max_tokens
        chars_to_cut = overflow * 4
        cut_per = chars_to_cut // len(all_passages) + 1
        for p in all_passages:
            if len(p["text"]) > cut_per + 50:
                p["text"] = p["text"][:-(cut_per)] + "..."
        info_lines = ["[HINT]You have 1 turn left. You must answer the question now.[/HINT]"]
        for i, p in enumerate(all_passages, 1):
            info_lines.append(f"Doc {i}(Title: {p['title']}) {p['text']}")
        info_block = "\n".join(info_lines)
        assistant_prefix = (
            f"<think>Let me search for information about these questions.</think>"
            f"<search>{' '.join(search_query)}</search>\n\n"
            f"<information>\n{info_block}\n</information>\n"
        )
        messages[1]["content"] = assistant_prefix
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        if prompt.endswith("<|im_end|>\n"):
            prompt = prompt[:-len("<|im_end|>\n")]
        elif prompt.endswith("<|im_end|>"):
            prompt = prompt[:-len("<|im_end|>")]

    return prompt


# ---------------------------------------------------------------------------
# EM evaluation (from Phase 1a)
# ---------------------------------------------------------------------------

def normalize_answer(s):
    import string
    s = s.lower().strip()
    for art in ["a ", "an ", "the "]:
        while s.startswith(art):
            s = s[len(art):]
    s = "".join(c for c in s if c not in string.punctuation)
    s = " ".join(s.split())
    return s


def em_check(prediction, gold_answers):
    pred_norm = normalize_answer(prediction)
    return any(normalize_answer(g) == pred_norm for g in gold_answers)


def extract_answer(text):
    matches = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()


# ---------------------------------------------------------------------------
# Hidden state collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_hidden_states(model, tokenizer, prompts, layer_indices, device="cuda",
                          batch_size=1):
    """Run forward pass and collect hidden states at specified layers.

    Returns dict: {layer_idx: np.array of shape [n_samples, hidden_dim]}
    Hidden state = residual stream at last token position of each prompt.
    """
    hidden_states = {l: [] for l in layer_indices}
    model.eval()

    for i in tqdm(range(0, len(prompts), batch_size), desc="Collecting hidden states"):
        batch_prompts = prompts[i:i + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True,
                           truncation=True, max_length=MAX_INPUT_TOKENS).to(device)

        outputs = model(**inputs, output_hidden_states=True)

        for l in layer_indices:
            hs = outputs.hidden_states[l + 1]  # after layer l
            for b in range(hs.shape[0]):
                attn_mask = inputs["attention_mask"][b]
                last_pos = attn_mask.sum().item() - 1
                hidden_states[l].append(hs[b, last_pos].cpu().float().numpy())

        del outputs, inputs
        torch.cuda.empty_cache()

    return {l: np.stack(vecs) for l, vecs in hidden_states.items()}


# ---------------------------------------------------------------------------
# Subspace operations
# ---------------------------------------------------------------------------

def compute_activation_subspace(H, k):
    """Compute top-k principal directions of activation matrix H [n_samples, hidden_dim].

    Returns V_k: [hidden_dim, k] — the top-k right singular vectors.
    """
    H_centered = H - H.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(H_centered, full_matrices=False)
    V_k = Vt[:k].T  # [hidden_dim, k]
    explained_var = (S[:k] ** 2).sum() / (S ** 2).sum()
    return V_k, float(explained_var), S


def project_to_subspace(H, V_k):
    """Project H onto subspace spanned by columns of V_k.
    Returns H_proj: [n_samples, k] — coordinates in the subspace.
    """
    return H @ V_k  # [n, k]


def random_subspace(hidden_dim, k, seed=42):
    """Generate a random orthonormal subspace of dimension k."""
    rng = np.random.RandomState(seed)
    A = rng.randn(hidden_dim, k)
    Q, _ = np.linalg.qr(A)
    return Q  # [hidden_dim, k]


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------

def run_probes(features_dict, targets, target_type="binary", n_folds=5, seed=42):
    """Train linear probes on different feature sets.

    Args:
        features_dict: {name: np.array [n_samples, dim]}
        targets: np.array [n_samples]
        target_type: "binary" (LogisticRegression) or "regression" (Ridge)
        n_folds: cross-validation folds

    Returns: {name: {metric_name: value}}
    """
    results = {}

    for name, X in features_dict.items():
        if X.shape[0] < n_folds * 2:
            results[name] = {"error": "too few samples"}
            continue

        scaler = StandardScaler()

        if target_type == "binary":
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            accs, aucs = [], []
            y = targets.astype(int)

            # Check if we have both classes
            if len(np.unique(y)) < 2:
                results[name] = {"error": "single class", "dim": int(X.shape[1])}
                continue

            for train_idx, test_idx in kf.split(X, y):
                X_train = scaler.fit_transform(X[train_idx])
                X_test = scaler.transform(X[test_idx])
                y_train, y_test = y[train_idx], y[test_idx]

                clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                y_prob = clf.predict_proba(X_test)[:, 1] if len(np.unique(y_train)) > 1 else np.zeros(len(y_test))

                accs.append(accuracy_score(y_test, y_pred))
                if len(np.unique(y_test)) > 1 and len(np.unique(y_train)) > 1:
                    aucs.append(roc_auc_score(y_test, y_prob))

            results[name] = {
                "accuracy_mean": float(np.mean(accs)),
                "accuracy_std": float(np.std(accs)),
                "auc_mean": float(np.mean(aucs)) if aucs else None,
                "auc_std": float(np.std(aucs)) if aucs else None,
                "dim": int(X.shape[1]),
            }

        else:  # regression
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            r2s, corrs = [], []
            y = targets.astype(float)

            for train_idx, test_idx in kf.split(X):
                X_train = scaler.fit_transform(X[train_idx])
                X_test = scaler.transform(X[test_idx])
                y_train, y_test = y[train_idx], y[test_idx]

                reg = Ridge(alpha=1.0)
                reg.fit(X_train, y_train)
                y_pred = reg.predict(X_test)

                if np.std(y_test) > 1e-10:
                    r2s.append(r2_score(y_test, y_pred))
                    corrs.append(float(np.corrcoef(y_test, y_pred)[0, 1]))

            results[name] = {
                "r2_mean": float(np.mean(r2s)) if r2s else None,
                "r2_std": float(np.std(r2s)) if r2s else None,
                "corr_mean": float(np.mean(corrs)) if corrs else None,
                "corr_std": float(np.std(corrs)) if corrs else None,
                "dim": int(X.shape[1]),
            }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 1c: Principal Direction Activation Shift Probing")
    parser.add_argument("--rl_dir", default="/root/autodl-tmp/models/MEM1")
    parser.add_argument("--base_dir", default="/root/autodl-tmp/models/Qwen2.5-7B")
    parser.add_argument("--output_path", default="/root/autodl-tmp/learned_traj_compress/artifacts/exp_004_probe_results.json")
    parser.add_argument("--cache_dir", default="/root/autodl-tmp/.hf_cache")
    parser.add_argument("--samples_per_n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true",
                        help="Use 10 samples per N, 3 N values, 3 layers")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    N_VALUES = [1, 2, 4, 8, 12, 16]
    LAYER_INDICES = [0, 3, 7, 11, 15, 19, 23, 27]
    K_VALUES = [64, 256]  # Match exp_003b k-sensitivity findings
    N_RANDOM_SEEDS = 5

    if args.dry_run:
        args.samples_per_n = 10
        N_VALUES = [1, 4, 8]
        LAYER_INDICES = [0, 15, 27]
        print("*** DRY RUN: 10 samples/N, N=[1,4,8], layers=[0,15,27] ***")

    print(f"=== Phase 1c: Principal Direction Activation Shift Probing ===")
    print(f"RL model: {args.rl_dir}")
    print(f"Base model: {args.base_dir}")
    print(f"Subspace k values: {K_VALUES}")
    print(f"N values: {N_VALUES}")
    print(f"Layers: {LAYER_INDICES}")
    print(f"Samples per N: {args.samples_per_n}")
    print()

    t0 = time.time()

    # --- Load data ---
    print("[1/6] Loading QA data...")
    pool = load_qa_pool(args.cache_dir, args.seed)

    all_samples = []
    for n in N_VALUES:
        samples = build_samples(pool, n, args.samples_per_n, args.seed)
        for sample in samples:
            all_samples.append((n, sample))
    print(f"  Total samples: {len(all_samples)}")

    # --- Load tokenizer ---
    print("[2/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.rl_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = []
    for n, sample in all_samples:
        prompt = build_prompt(sample, tokenizer)
        prompts.append(prompt)
    print(f"  Built {len(prompts)} prompts")

    # --- Collect base model hidden states ---
    print("[3/6] Collecting base model hidden states...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_dir, torch_dtype=torch.float16, device_map=args.device,
        trust_remote_code=True)
    hs_base = collect_hidden_states(base_model, tokenizer, prompts, LAYER_INDICES,
                                    device=args.device)
    del base_model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Collected: {', '.join(f'L{l}: {v.shape}' for l, v in hs_base.items())}")

    # --- Collect RL model hidden states + inference for labels ---
    print("[4/6] Collecting RL model hidden states + running inference...")
    rl_model = AutoModelForCausalLM.from_pretrained(
        args.rl_dir, torch_dtype=torch.float16, device_map=args.device,
        trust_remote_code=True)

    hs_rl = collect_hidden_states(rl_model, tokenizer, prompts, LAYER_INDICES,
                                  device=args.device)

    # Run inference for EM labels
    print("  Running RL inference for EM labels...")
    sample_labels = []
    for idx, (n, sample) in enumerate(tqdm(all_samples, desc="RL inference")):
        prompt = prompts[idx]
        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
        with torch.no_grad():
            output_ids = rl_model.generate(
                **inputs, max_new_tokens=512,
                do_sample=False, temperature=1.0,
            )
        generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)

        raw_answer = extract_answer(generated)
        per_obj_em = []
        if raw_answer is not None:
            answer_parts = [a.strip() for a in raw_answer.split(";")]
            for q_idx, item in enumerate(sample):
                if q_idx < len(answer_parts):
                    em = 1 if em_check(answer_parts[q_idx], item["answers"]) else 0
                else:
                    em = 0
                per_obj_em.append(em)
        else:
            per_obj_em = [0] * n

        mean_em = sum(per_obj_em) / len(per_obj_em) if per_obj_em else 0.0
        sample_labels.append({
            "n": n, "mean_em": float(mean_em),
            "per_obj_em": per_obj_em,
            "extraction_failed": raw_answer is None,
        })

    del rl_model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Collected RL hidden states + labels for {len(sample_labels)} samples")

    # --- Compute Δh and run probes ---
    print("[5/6] Computing activation deltas and running probes...")

    mean_ems = np.array([s["mean_em"] for s in sample_labels])
    binary_labels = (mean_ems > 0.5).astype(int)
    n_values_arr = np.array([s["n"] for s in sample_labels])
    hidden_dim = list(hs_base.values())[0].shape[1]

    print(f"  Label stats: mean_em={mean_ems.mean():.3f}, binary_pos_rate={binary_labels.mean():.3f}")
    print(f"  Hidden dim: {hidden_dim}")

    # Pre-generate random subspaces for each k
    random_Vs = {}
    for k in K_VALUES:
        random_Vs[k] = [random_subspace(hidden_dim, k, seed=args.seed + s) for s in range(N_RANDOM_SEEDS)]

    layer_results = {}

    for l in LAYER_INDICES:
        print(f"\n  --- Layer {l} ---")
        H_base = hs_base[l]  # [n_samples, hidden_dim]
        H_rl = hs_rl[l]
        delta_H = H_rl - H_base  # [n_samples, hidden_dim]

        # Δh magnitude statistics
        delta_norms = np.linalg.norm(delta_H, axis=1)
        base_norms = np.linalg.norm(H_base, axis=1)
        relative_delta = delta_norms / (base_norms + 1e-12)
        print(f"  Δh norm: mean={delta_norms.mean():.2f}, relative={relative_delta.mean():.4f}")

        k_results = {}

        for k in K_VALUES:
            print(f"    k={k}:")

            # Compute base activation principal subspace
            V_base_k, base_var_explained, S_base = compute_activation_subspace(H_base, k)
            print(f"      Base subspace: top-{k} explains {base_var_explained:.1%} of variance")

            # Feature sets for probing:
            features = {
                # Upper bound: full RL hidden states
                "h_rl_full": H_rl,
                # Full activation delta
                "delta_h_full": delta_H,
                # Δh projected to base principal subspace (key test)
                f"delta_h_principal_k{k}": project_to_subspace(delta_H, V_base_k),
                # RL hidden states projected to base principal subspace
                f"h_rl_principal_k{k}": project_to_subspace(H_rl, V_base_k),
                # Base hidden states projected to own principal subspace (control)
                f"h_base_principal_k{k}": project_to_subspace(H_base, V_base_k),
            }

            # Random subspace baselines (Δh projected to random directions)
            for rs_idx, V_rand in enumerate(random_Vs[k]):
                features[f"delta_h_random_{rs_idx}"] = project_to_subspace(delta_H, V_rand)

            # Compute projection energy: how much of Δh is in the principal subspace?
            delta_h_in_principal = project_to_subspace(delta_H, V_base_k)  # [n, k]
            delta_h_reconstructed = delta_h_in_principal @ V_base_k.T  # [n, d]
            energy_in = np.sum(delta_h_reconstructed ** 2) / (np.sum(delta_H ** 2) + 1e-12)

            # Binary probes
            print(f"      Running binary probes...")
            binary_results = run_probes(features, binary_labels, target_type="binary",
                                        seed=args.seed)

            # Regression probes
            print(f"      Running regression probes...")
            reg_results = run_probes(features, mean_ems, target_type="regression",
                                     seed=args.seed)

            # Aggregate random results
            for probe_type, results_dict in [("binary", binary_results), ("regression", reg_results)]:
                random_metrics = []
                for rs_idx in range(N_RANDOM_SEEDS):
                    key = f"delta_h_random_{rs_idx}"
                    if key in results_dict and "error" not in results_dict[key]:
                        random_metrics.append(results_dict[key])
                if random_metrics:
                    avg_random = {}
                    for metric_key in random_metrics[0]:
                        if metric_key == "dim":
                            avg_random[metric_key] = random_metrics[0][metric_key]
                        elif random_metrics[0][metric_key] is not None:
                            vals = [m[metric_key] for m in random_metrics if m[metric_key] is not None]
                            avg_random[metric_key] = float(np.mean(vals))
                    results_dict["delta_h_random_avg"] = avg_random
                for rs_idx in range(N_RANDOM_SEEDS):
                    results_dict.pop(f"delta_h_random_{rs_idx}", None)

            k_results[str(k)] = {
                "base_var_explained": float(base_var_explained),
                "delta_h_energy_in_principal": float(energy_in),
                "binary_probes": binary_results,
                "regression_probes": reg_results,
            }

            # Print summary
            for ptype, res in [("binary", binary_results), ("regression", reg_results)]:
                print(f"      [{ptype}]")
                for name in ["h_rl_full", "delta_h_full",
                             f"delta_h_principal_k{k}", f"h_rl_principal_k{k}",
                             f"h_base_principal_k{k}", "delta_h_random_avg"]:
                    if name in res and "error" not in res[name]:
                        if ptype == "binary":
                            acc = res[name]['accuracy_mean']
                            auc_str = f" auc={res[name]['auc_mean']:.3f}" if res[name].get('auc_mean') else ""
                            print(f"        {name:>30s}: acc={acc:.3f}±{res[name]['accuracy_std']:.3f}{auc_str}")
                        else:
                            r2 = res[name].get('r2_mean')
                            corr = res[name].get('corr_mean')
                            if r2 is not None:
                                print(f"        {name:>30s}: R²={r2:.3f}" + (f" corr={corr:.3f}" if corr else ""))
                            else:
                                print(f"        {name:>30s}: N/A")

        layer_results[str(l)] = {
            "delta_h_norm_mean": float(delta_norms.mean()),
            "delta_h_relative_mean": float(relative_delta.mean()),
            "k_analysis": k_results,
        }

    # --- Compile output ---
    print("\n[6/6] Compiling results...")
    elapsed = time.time() - t0

    # Go/no-go: principal Δh probe > random Δh probe in majority of comparisons
    principal_wins = 0
    total_comparisons = 0

    for l_str, l_res in layer_results.items():
        for k_str, k_res in l_res["k_analysis"].items():
            k = int(k_str)
            for ptype in ["binary_probes", "regression_probes"]:
                probes = k_res[ptype]
                principal = probes.get(f"delta_h_principal_k{k}", {})
                rand_avg = probes.get("delta_h_random_avg", {})

                if ptype == "binary_probes":
                    p_val = principal.get("accuracy_mean")
                    r_val = rand_avg.get("accuracy_mean")
                else:
                    p_val = principal.get("r2_mean")
                    r_val = rand_avg.get("r2_mean")

                if p_val is not None and r_val is not None:
                    total_comparisons += 1
                    if p_val > r_val:
                        principal_wins += 1

    advantage_rate = principal_wins / total_comparisons if total_comparisons > 0 else 0

    output = {
        "config": {
            "rl_dir": args.rl_dir,
            "base_dir": args.base_dir,
            "k_values": K_VALUES,
            "n_values": N_VALUES,
            "samples_per_n": args.samples_per_n,
            "layer_indices": LAYER_INDICES,
            "n_total_samples": len(all_samples),
            "n_random_seeds": N_RANDOM_SEEDS,
            "elapsed_s": round(elapsed, 1),
            "dry_run": args.dry_run,
        },
        "label_stats": {
            "mean_em_overall": float(mean_ems.mean()),
            "binary_pos_rate": float(binary_labels.mean()),
            "per_n_mean_em": {str(n): float(mean_ems[n_values_arr == n].mean()) for n in N_VALUES},
        },
        "layer_results": layer_results,
        "go_no_go": {
            "principal_advantage": bool(advantage_rate > 0.5),
            "principal_advantage_rate": float(advantage_rate),
            "detail": f"Principal Δh probe outperforms random Δh probe in {principal_wins}/{total_comparisons} comparisons",
            "overall": "GO" if advantage_rate > 0.6 else "PARTIAL" if advantage_rate > 0.4 else "NO-GO",
        },
        "sample_labels": sample_labels,
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")
    print(f"Elapsed: {elapsed:.0f}s")

    # --- Print summary ---
    print("\n" + "=" * 80)
    print("SUMMARY — Phase 1c: Principal Direction Activation Shift Probing")
    print("=" * 80)
    print(f"\nLabel distribution: mean_em={mean_ems.mean():.3f}, binary_pos={binary_labels.mean():.1%}")
    for n in N_VALUES:
        mask = n_values_arr == n
        print(f"  N={n}: mean_em={mean_ems[mask].mean():.3f}, n={mask.sum()}")

    for k in K_VALUES:
        print(f"\n--- k={k} ---")
        print(f"{'Layer':>6} {'Δh_full':>10} {'Δh_princ':>10} {'Δh_random':>10} {'h_rl':>10} {'h_base':>10} {'Δh_energy':>10}")
        for l in LAYER_INDICES:
            l_res = layer_results[str(l)]
            k_res = l_res["k_analysis"][str(k)]
            bp = k_res["binary_probes"]

            vals = []
            for name in ["delta_h_full", f"delta_h_principal_k{k}", "delta_h_random_avg",
                         f"h_rl_principal_k{k}", f"h_base_principal_k{k}"]:
                v = bp.get(name, {}).get("accuracy_mean")
                vals.append(f"{v:.3f}" if v is not None else "N/A")
            vals.append(f"{k_res['delta_h_energy_in_principal']:.3f}")

            print(f"  L{l:>3}: {'  '.join(f'{v:>10}' for v in vals)}")

        print(f"\nRegression R²:")
        print(f"{'Layer':>6} {'Δh_full':>10} {'Δh_princ':>10} {'Δh_random':>10} {'h_rl':>10} {'h_base':>10}")
        for l in LAYER_INDICES:
            l_res = layer_results[str(l)]
            k_res = l_res["k_analysis"][str(k)]
            rp = k_res["regression_probes"]

            vals = []
            for name in ["delta_h_full", f"delta_h_principal_k{k}", "delta_h_random_avg",
                         f"h_rl_principal_k{k}", f"h_base_principal_k{k}"]:
                v = rp.get(name, {}).get("r2_mean")
                vals.append(f"{v:.3f}" if v is not None else "N/A")
            print(f"  L{l:>3}: {'  '.join(f'{v:>10}' for v in vals)}")

    print(f"\nGo/No-Go: {output['go_no_go']['overall']}")
    print(f"  {output['go_no_go']['detail']}")


if __name__ == "__main__":
    main()
