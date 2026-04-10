#!/usr/bin/env python3
"""
exp_007: MLP Probe for Phase 1c Redo

Tests whether RL's multi-objective capability is encoded on a nonlinear manifold
in hidden states. Compares Linear vs MLP vs Random-feature probes on identical
data and CV splits.

Two stages:
  Stage 1 (GPU): Extract hidden states from RL + base models, save to disk.
  Stage 2 (CPU): Train/evaluate linear, MLP, and random-feature probes.

Usage:
    # Full pipeline (feature extraction + probing):
    python scripts/exp_007_mlp_probe.py --stage all

    # Only feature extraction (GPU):
    python scripts/exp_007_mlp_probe.py --stage extract

    # Only probing (CPU, features must exist):
    python scripts/exp_007_mlp_probe.py --stage probe

    # Dry run:
    python scripts/exp_007_mlp_probe.py --stage all --dry_run
"""

import argparse
import gc
import json
import os
import random
import re
import string
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from scipy import stats as scipy_stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_INPUT_TOKENS = 4096
N_VALUES = [1, 2, 4, 8, 12, 16]
LAYER_INDICES = [0, 3, 7, 11, 15, 19, 23, 27]
K_VALUES = [64, 256]
N_RANDOM_SEEDS = 5
N_PROBE_SEEDS = 5
N_FOLDS = 5

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


# ---------------------------------------------------------------------------
# Data loading (shared with Phase 1a/1c)
# ---------------------------------------------------------------------------

def normalize_answer(s):
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
# Hidden state collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_hidden_states(model, tokenizer, prompts, layer_indices, device="cuda"):
    hidden_states = {l: [] for l in layer_indices}
    model.eval()

    for i in tqdm(range(len(prompts)), desc="Collecting hidden states"):
        inputs = tokenizer(prompts[i], return_tensors="pt", truncation=True,
                           max_length=MAX_INPUT_TOKENS).to(device)
        outputs = model(**inputs, output_hidden_states=True)

        for l in layer_indices:
            hs = outputs.hidden_states[l + 1]
            last_pos = inputs["attention_mask"][0].sum().item() - 1
            hidden_states[l].append(hs[0, last_pos].cpu().float().numpy())

        del outputs, inputs
        torch.cuda.empty_cache()

    return {l: np.stack(vecs) for l, vecs in hidden_states.items()}


# ---------------------------------------------------------------------------
# Subspace operations
# ---------------------------------------------------------------------------

def compute_activation_subspace(H, k):
    H_centered = H - H.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(H_centered, full_matrices=False)
    V_k = Vt[:k].T
    explained_var = (S[:k] ** 2).sum() / (S ** 2).sum()
    return V_k, float(explained_var)


def project_to_subspace(H, V_k):
    return H @ V_k


def random_subspace(hidden_dim, k, seed=42):
    rng = np.random.RandomState(seed)
    A = rng.randn(hidden_dim, k)
    Q, _ = np.linalg.qr(A)
    return Q


# ---------------------------------------------------------------------------
# Stage 1: Feature Extraction (GPU)
# ---------------------------------------------------------------------------

def stage1_extract(args):
    print("=" * 60)
    print("STAGE 1: Feature Extraction (GPU)")
    print("=" * 60)

    feature_dir = Path(args.feature_dir)
    feature_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    n_values = N_VALUES
    layer_indices = LAYER_INDICES
    if args.dry_run:
        args.samples_per_n = 10
        n_values = [1, 4, 8]
        layer_indices = [0, 15, 27]
        print("*** DRY RUN ***")

    # Load data
    print("[1/5] Loading QA data...")
    pool = load_qa_pool(args.cache_dir, args.seed)
    all_samples = []
    for n in n_values:
        samples = build_samples(pool, n, args.samples_per_n, args.seed)
        for sample in samples:
            all_samples.append((n, sample))

    # Build prompts
    print("[2/5] Building prompts...")
    tokenizer = AutoTokenizer.from_pretrained(args.rl_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = [build_prompt(sample, tokenizer) for _, sample in all_samples]
    print(f"  {len(prompts)} prompts built")

    # Monitor VRAM before loading
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated() / 1e9
        vram_total = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"  VRAM before model load: {vram_used:.1f}/{vram_total:.1f} GB")

    # Collect base hidden states
    print("[3/5] Collecting base model hidden states...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_dir, torch_dtype=torch.float16, device_map=args.device,
        trust_remote_code=True)
    hs_base = collect_hidden_states(base_model, tokenizer, prompts, layer_indices,
                                    device=args.device)
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    # Collect RL hidden states + EM labels
    print("[4/5] Collecting RL model hidden states + inference...")
    rl_model = AutoModelForCausalLM.from_pretrained(
        args.rl_dir, torch_dtype=torch.float16, device_map=args.device,
        trust_remote_code=True)
    hs_rl = collect_hidden_states(rl_model, tokenizer, prompts, layer_indices,
                                  device=args.device)

    print("  Running RL inference for EM labels...")
    sample_labels = []
    for idx, (n, sample) in enumerate(tqdm(all_samples, desc="RL inference")):
        inputs = tokenizer(prompts[idx], return_tensors="pt").to(args.device)
        with torch.no_grad():
            output_ids = rl_model.generate(
                **inputs, max_new_tokens=512, do_sample=False, temperature=1.0)
        generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)

        raw_answer = extract_answer(generated)
        per_obj_em = []
        if raw_answer is not None:
            answer_parts = [a.strip() for a in raw_answer.split(";")]
            for q_idx, item in enumerate(sample):
                em = 1 if q_idx < len(answer_parts) and em_check(answer_parts[q_idx], item["answers"]) else 0
                per_obj_em.append(em)
        else:
            per_obj_em = [0] * n
        mean_em = sum(per_obj_em) / len(per_obj_em) if per_obj_em else 0.0
        sample_labels.append({"n": n, "mean_em": float(mean_em), "per_obj_em": per_obj_em})

    del rl_model
    gc.collect()
    torch.cuda.empty_cache()

    # Save features
    print("[5/5] Saving features to disk...")
    for l in layer_indices:
        np.save(feature_dir / f"hs_base_L{l}.npy", hs_base[l])
        np.save(feature_dir / f"hs_rl_L{l}.npy", hs_rl[l])

    with open(feature_dir / "labels.json", "w") as f:
        json.dump(sample_labels, f, indent=2)

    meta = {
        "n_values": n_values,
        "layer_indices": layer_indices,
        "samples_per_n": args.samples_per_n,
        "n_samples": len(all_samples),
        "hidden_dim": int(hs_base[layer_indices[0]].shape[1]),
        "seed": args.seed,
        "extract_time_s": round(time.time() - t0, 1),
    }
    with open(feature_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Features saved to {feature_dir}")
    print(f"  Extraction time: {time.time() - t0:.0f}s")
    print(f"  GPU released.")
    return meta


# ---------------------------------------------------------------------------
# MLP Probe
# ---------------------------------------------------------------------------

class MLPProbe(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256, 128), dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp_probe(X_train, y_train, X_val, y_val, input_dim, seed,
                     hidden_dims=(256, 128), dropout=0.3,
                     lr=1e-3, weight_decay=1e-4, epochs=100, patience=15):
    torch.manual_seed(seed)
    model = MLPProbe(input_dim, hidden_dims, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    X_v = torch.tensor(X_val, dtype=torch.float32)
    y_v = torch.tensor(y_val, dtype=torch.float32)

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_tr)
        loss = criterion(pred, y_tr)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_v)
            val_loss = criterion(val_pred, y_v).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate_probe_predictions(y_true, y_pred):
    """Compute Spearman rho and R2."""
    rho, p_val = scipy_stats.spearmanr(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"spearman_rho": float(rho), "spearman_p": float(p_val), "r2": float(r2)}


# ---------------------------------------------------------------------------
# Stage 2: Probing (CPU)
# ---------------------------------------------------------------------------

def stage2_probe(args):
    print("=" * 60)
    print("STAGE 2: Probing (CPU)")
    print("=" * 60)

    feature_dir = Path(args.feature_dir)
    t0 = time.time()

    # Load metadata and labels
    with open(feature_dir / "meta.json") as f:
        meta = json.load(f)
    with open(feature_dir / "labels.json") as f:
        sample_labels = json.load(f)

    layer_indices = meta["layer_indices"]
    hidden_dim = meta["hidden_dim"]
    n_samples = meta["n_samples"]
    mean_ems = np.array([s["mean_em"] for s in sample_labels])
    binary_labels = (mean_ems > 0.5).astype(int)

    print(f"  Samples: {n_samples}, hidden_dim: {hidden_dim}")
    print(f"  mean_EM: {mean_ems.mean():.3f}, binary_pos_rate: {binary_labels.mean():.3f}")

    # Pre-generate random subspaces
    random_Vs = {}
    for k in K_VALUES:
        random_Vs[k] = [random_subspace(hidden_dim, k, seed=args.seed + s)
                        for s in range(N_RANDOM_SEEDS)]

    all_results = {}

    for l in layer_indices:
        print(f"\n--- Layer {l} ---")
        H_base = np.load(feature_dir / f"hs_base_L{l}.npy")
        H_rl = np.load(feature_dir / f"hs_rl_L{l}.npy")
        delta_H = H_rl - H_base

        layer_res = {}

        for k in K_VALUES:
            print(f"  k={k}")
            V_base_k, var_explained = compute_activation_subspace(H_base, k)

            # Build feature dict
            features = {
                "h_rl_full": H_rl,
                "delta_h_full": delta_H,
                f"delta_h_principal_k{k}": project_to_subspace(delta_H, V_base_k),
                f"h_rl_principal_k{k}": project_to_subspace(H_rl, V_base_k),
                f"h_base_principal_k{k}": project_to_subspace(H_base, V_base_k),
            }
            for rs_idx, V_rand in enumerate(random_Vs[k]):
                features[f"delta_h_random_{rs_idx}"] = project_to_subspace(delta_H, V_rand)

            # --- Run probes ---
            probe_results = {}
            for feat_name, X in features.items():
                print(f"    {feat_name} (dim={X.shape[1]})...", end=" ")
                feat_res = run_all_probes(X, mean_ems, binary_labels, args.seed)
                probe_results[feat_name] = feat_res
                mlp_rho = feat_res["mlp_regression"]["spearman_rho_mean"]
                lin_rho = feat_res["linear_regression"]["spearman_rho_mean"]
                print(f"MLP ρ={mlp_rho:.3f}, Linear ρ={lin_rho:.3f}")

            # Aggregate random baselines
            for probe_type in ["linear_regression", "linear_binary", "mlp_regression", "mlp_binary"]:
                random_metrics = []
                for rs_idx in range(N_RANDOM_SEEDS):
                    key = f"delta_h_random_{rs_idx}"
                    if key in probe_results:
                        random_metrics.append(probe_results[key][probe_type])
                if random_metrics:
                    avg = {}
                    for mk in random_metrics[0]:
                        vals = [m[mk] for m in random_metrics if m[mk] is not None]
                        avg[mk] = float(np.mean(vals)) if vals else None
                    probe_results[f"delta_h_random_avg"] = probe_results.get(f"delta_h_random_avg", {})
                    probe_results[f"delta_h_random_avg"][probe_type] = avg
                for rs_idx in range(N_RANDOM_SEEDS):
                    probe_results.pop(f"delta_h_random_{rs_idx}", None)

            # Random feature MLP baseline (shuffle features, same MLP)
            print(f"    random_feature_mlp (control)...", end=" ")
            X_principal = features[f"delta_h_principal_k{k}"]
            rand_feat_res = run_random_feature_mlp(X_principal, mean_ems, binary_labels, args.seed)
            probe_results["random_feature_mlp"] = rand_feat_res
            print(f"ρ={rand_feat_res['mlp_regression']['spearman_rho_mean']:.3f}")

            layer_res[str(k)] = {
                "var_explained": var_explained,
                "probes": probe_results,
            }

        all_results[str(l)] = layer_res

    # Compile output
    output = compile_results(all_results, meta, sample_labels, mean_ems, binary_labels, t0)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print_summary(output)
    print(f"\nResults saved to {output_path}")
    print(f"Total probe time: {time.time() - t0:.0f}s")
    return output


def run_all_probes(X, mean_ems, binary_labels, base_seed):
    """Run linear + MLP probes with 5-fold CV × 5 seeds."""
    result = {
        "linear_regression": run_linear_cv(X, mean_ems, "regression", base_seed),
        "linear_binary": run_linear_cv(X, binary_labels, "binary", base_seed),
        "mlp_regression": run_mlp_cv(X, mean_ems, base_seed),
        "mlp_binary": run_mlp_cv(X, binary_labels, base_seed, task="binary"),
    }
    return result


def run_linear_cv(X, y, task, seed):
    """Linear probe with 5-fold CV."""
    if task == "binary":
        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        y_int = y.astype(int)
        if len(np.unique(y_int)) < 2:
            return {"accuracy_mean": None, "spearman_rho_mean": None}
        accs = []
        for train_idx, test_idx in kf.split(X, y_int):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_idx])
            X_te = scaler.transform(X[test_idx])
            clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
            clf.fit(X_tr, y_int[train_idx])
            accs.append(accuracy_score(y_int[test_idx], clf.predict(X_te)))
        return {
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "spearman_rho_mean": None,
        }
    else:  # regression
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        rhos, r2s, train_rhos = [], [], []
        for train_idx, test_idx in kf.split(X):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_idx])
            X_te = scaler.transform(X[test_idx])
            reg = Ridge(alpha=1.0)
            reg.fit(X_tr, y[train_idx])
            y_pred_test = reg.predict(X_te)
            y_pred_train = reg.predict(X_tr)

            rho_test, _ = scipy_stats.spearmanr(y[test_idx], y_pred_test)
            rho_train, _ = scipy_stats.spearmanr(y[train_idx], y_pred_train)
            rhos.append(float(rho_test))
            train_rhos.append(float(rho_train))
            r2s.append(r2_score(y[test_idx], y_pred_test))

        return {
            "spearman_rho_mean": float(np.mean(rhos)),
            "spearman_rho_std": float(np.std(rhos)),
            "r2_mean": float(np.mean(r2s)),
            "train_spearman_mean": float(np.mean(train_rhos)),
            "train_test_gap": float(np.mean(train_rhos) - np.mean(rhos)),
        }


def run_mlp_cv(X, y, base_seed, task="regression"):
    """MLP probe with 5-fold CV × 5 seeds."""
    all_test_rhos, all_train_rhos, all_test_accs, all_r2s = [], [], [], []

    for seed_offset in range(N_PROBE_SEEDS):
        seed = base_seed + seed_offset * 100
        if task == "binary":
            kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
            split_iter = kf.split(X, y.astype(int))
        else:
            kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
            split_iter = kf.split(X)

        for train_idx, test_idx in split_iter:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_idx])
            X_te = scaler.transform(X[test_idx])
            y_tr, y_te = y[train_idx].astype(float), y[test_idx].astype(float)

            # Use 20% of train as validation for early stopping
            n_val = max(1, int(len(train_idx) * 0.2))
            X_val, y_val = X_tr[-n_val:], y_tr[-n_val:]
            X_train, y_train = X_tr[:-n_val], y_tr[:-n_val]

            model = train_mlp_probe(
                X_train, y_train, X_val, y_val,
                input_dim=X.shape[1], seed=seed)

            model.eval()
            with torch.no_grad():
                y_pred_test = model(torch.tensor(X_te, dtype=torch.float32)).numpy()
                y_pred_train = model(torch.tensor(X_tr, dtype=torch.float32)).numpy()

            if task == "regression":
                rho_test, _ = scipy_stats.spearmanr(y_te, y_pred_test)
                rho_train, _ = scipy_stats.spearmanr(y_tr, y_pred_train)
                all_test_rhos.append(float(rho_test))
                all_train_rhos.append(float(rho_train))
                all_r2s.append(r2_score(y_te, y_pred_test))
            else:
                pred_binary = (y_pred_test > 0.5).astype(int)
                all_test_accs.append(accuracy_score(y_te.astype(int), pred_binary))

    if task == "regression":
        return {
            "spearman_rho_mean": float(np.mean(all_test_rhos)),
            "spearman_rho_std": float(np.std(all_test_rhos)),
            "r2_mean": float(np.mean(all_r2s)),
            "train_spearman_mean": float(np.mean(all_train_rhos)),
            "train_test_gap": float(np.mean(all_train_rhos) - np.mean(all_test_rhos)),
            "n_runs": len(all_test_rhos),
        }
    else:
        return {
            "accuracy_mean": float(np.mean(all_test_accs)),
            "accuracy_std": float(np.std(all_test_accs)),
            "spearman_rho_mean": None,
            "n_runs": len(all_test_accs),
        }


def run_random_feature_mlp(X, mean_ems, binary_labels, base_seed):
    """Shuffle feature dimensions (break structure) and run MLP — overfitting control."""
    rng = np.random.RandomState(base_seed + 999)
    X_shuffled = X.copy()
    for col in range(X_shuffled.shape[1]):
        rng.shuffle(X_shuffled[:, col])

    return {
        "mlp_regression": run_mlp_cv(X_shuffled, mean_ems, base_seed),
        "mlp_binary": run_mlp_cv(X_shuffled, binary_labels, base_seed, task="binary"),
    }


# ---------------------------------------------------------------------------
# Results compilation
# ---------------------------------------------------------------------------

def compile_results(all_results, meta, sample_labels, mean_ems, binary_labels, t0):
    # Statistical test: MLP vs Linear across all (layer, k) combinations
    mlp_rhos, linear_rhos = [], []
    for l_str, l_res in all_results.items():
        for k_str, k_res in l_res.items():
            for feat_name, feat_res in k_res["probes"].items():
                if feat_name.startswith("delta_h_principal_k"):
                    mlp_r = feat_res.get("mlp_regression", {}).get("spearman_rho_mean")
                    lin_r = feat_res.get("linear_regression", {}).get("spearman_rho_mean")
                    if mlp_r is not None and lin_r is not None:
                        mlp_rhos.append(mlp_r)
                        linear_rhos.append(lin_r)

    # Paired t-test
    if len(mlp_rhos) >= 2:
        t_stat, p_val = scipy_stats.ttest_rel(mlp_rhos, linear_rhos)
    else:
        t_stat, p_val = 0, 1.0

    # Best MLP result
    best_mlp_rho = max(mlp_rhos) if mlp_rhos else 0
    best_linear_rho = max(linear_rhos) if linear_rhos else 0

    go_no_go = "GO" if (best_mlp_rho > 0.3 and p_val < 0.05) else \
               "PARTIAL" if best_mlp_rho > 0.2 else "NO-GO"

    return {
        "config": meta,
        "label_stats": {
            "mean_em": float(mean_ems.mean()),
            "binary_pos_rate": float(binary_labels.mean()),
            "n_samples": len(mean_ems),
        },
        "layer_results": all_results,
        "summary": {
            "best_mlp_spearman": best_mlp_rho,
            "best_linear_spearman": best_linear_rho,
            "mlp_vs_linear_ttest_p": float(p_val),
            "mlp_vs_linear_ttest_t": float(t_stat),
            "go_no_go": go_no_go,
        },
        "probe_time_s": round(time.time() - t0, 1),
    }


def print_summary(output):
    print("\n" + "=" * 70)
    print("SUMMARY — exp_007: MLP Probe Phase 1c Redo")
    print("=" * 70)
    s = output["summary"]
    print(f"  Best MLP Spearman ρ:    {s['best_mlp_spearman']:.3f}")
    print(f"  Best Linear Spearman ρ: {s['best_linear_spearman']:.3f}")
    print(f"  MLP vs Linear t-test:   t={s['mlp_vs_linear_ttest_t']:.2f}, p={s['mlp_vs_linear_ttest_p']:.4f}")
    print(f"  Go/No-Go:               {s['go_no_go']}")

    print(f"\n  Per-layer principal Δh (regression Spearman ρ):")
    print(f"  {'Layer':>6} {'k':>4} {'Linear':>10} {'MLP':>10} {'Random':>10} {'Train-Test Gap':>15}")
    for l_str, l_res in output["layer_results"].items():
        for k_str, k_res in l_res.items():
            probes = k_res["probes"]
            pk = f"delta_h_principal_k{k_str}"
            lin = probes.get(pk, {}).get("linear_regression", {}).get("spearman_rho_mean", 0)
            mlp = probes.get(pk, {}).get("mlp_regression", {}).get("spearman_rho_mean", 0)
            gap = probes.get(pk, {}).get("mlp_regression", {}).get("train_test_gap", 0)
            rand_res = probes.get("random_feature_mlp", {}).get("mlp_regression", {})
            rand = rand_res.get("spearman_rho_mean", 0) if rand_res else 0
            print(f"  L{l_str:>4} k={k_str:>3}  {lin:>10.3f} {mlp:>10.3f} {rand:>10.3f} {gap:>15.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="exp_007: MLP Probe Phase 1c Redo")
    parser.add_argument("--stage", choices=["all", "extract", "probe"], default="all")
    parser.add_argument("--rl_dir", default="/root/autodl-tmp/models/MEM1")
    parser.add_argument("--base_dir", default="/root/autodl-tmp/models/Qwen2.5-7B")
    parser.add_argument("--feature_dir",
                        default="/root/autodl-tmp/learned_traj_compress/artifacts/exp_007_features")
    parser.add_argument("--output_path",
                        default="/root/autodl-tmp/learned_traj_compress/artifacts/exp_007_mlp_probe_results.json")
    parser.add_argument("--cache_dir", default="/root/autodl-tmp/.hf_cache")
    parser.add_argument("--samples_per_n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.stage in ("all", "extract"):
        stage1_extract(args)

    if args.stage in ("all", "probe"):
        stage2_probe(args)


if __name__ == "__main__":
    main()
