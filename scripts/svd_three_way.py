#!/usr/bin/env python3
"""
Three-way SVD Comparison: RL vs Base, SFT vs Base, RL vs SFT

Extends exp_003b random baseline analysis to include SFT model.
Core question: Does SFT produce the same alignment pattern as RL?

Analyzes:
1. obs/random ratios for each pair (RL-base, SFT-base, RL-SFT)
2. One-sided projection asymmetry (U-side vs V-side) — key novelty signal
3. k-sensitivity (k ∈ {64, 128, 256, 512})
4. Layer-wise patterns

Note on k=1024: k/v_proj have shape [512, 3584], so k=1024 is clamped to 512.
We use k_max=512 and annotate this in outputs.

Usage:
    python scripts/svd_three_way.py \
        --rl_dir /root/autodl-tmp/models/MEM1 \
        --base_dir /root/autodl-tmp/models/Qwen2.5-7B \
        --sft_dir /root/autodl-tmp/models/Qwen2.5-7B-SFT-N2 \
        --output_path /root/autodl-tmp/learned_traj_compress/artifacts/exp_004_svd_three_way.json \
        [--dry_run]
"""

import argparse
import gc
import json
import math
import os
import time
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

# ---------- constants ----------
WEIGHT_PATTERNS = (
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
)

ATTN_PATTERNS = (".q_proj.", ".k_proj.", ".v_proj.", ".o_proj.")
MLP_PATTERNS = (".gate_proj.", ".up_proj.", ".down_proj.")

K_VALUES = [64, 128, 256, 512]  # Dropped 1024 (k/v_proj clamped to 512)
N_RANDOM = 100


def param_wanted(name):
    return any(name.endswith(p) for p in WEIGHT_PATTERNS)

def get_layer_idx(name):
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            return int(parts[i + 1])
    return -1

def get_component(name):
    if any(p in name for p in ATTN_PATTERNS):
        return "attn"
    elif any(p in name for p in MLP_PATTERNS):
        return "mlp"
    return "other"

def get_proj_type(name):
    for pattern in WEIGHT_PATTERNS:
        suffix = pattern.lstrip(".")
        if name.endswith(suffix):
            return suffix.replace(".weight", "")
    return "unknown"


# ---------- core analysis ----------
@torch.no_grad()
def analyze_delta(dW, U_ref_k, V_ref_k, n_random, device):
    """Analyze a single ΔW relative to a reference model's SVD subspace.

    Returns dict with bilateral in-ratio, one-sided ratios, random baseline, etc.
    """
    m, n = dW.shape
    k = U_ref_k.shape[1]
    dW_fro2 = torch.norm(dW, p="fro") ** 2

    if dW_fro2 < 1e-12:
        return {"skipped": True}

    dW_fro = math.sqrt(dW_fro2.item())

    # Bilateral projection
    proj = U_ref_k.T @ dW @ V_ref_k
    bilateral_in = (torch.norm(proj, p="fro") ** 2 / dW_fro2).item()

    # One-sided
    u_proj = U_ref_k.T @ dW
    u_side = (torch.norm(u_proj, p="fro") ** 2 / dW_fro2).item()

    v_proj = dW @ V_ref_k
    v_side = (torch.norm(v_proj, p="fro") ** 2 / dW_fro2).item()

    # Random baseline
    random_ratios = []
    for _ in range(n_random):
        R = torch.randn(m, n, device=device, dtype=torch.float32)
        R = R * (dW_fro / torch.norm(R, p="fro").item())
        rp = U_ref_k.T @ R @ V_ref_k
        random_ratios.append((torch.norm(rp, p="fro") ** 2 / torch.norm(R, p="fro") ** 2).item())

    random_mean = float(np.mean(random_ratios))
    theoretical = k ** 2 / (m * n)
    obs_over_random = bilateral_in / random_mean if random_mean > 1e-12 else float("inf")

    return {
        "bilateral_in_ratio": bilateral_in,
        "u_side_ratio": u_side,
        "v_side_ratio": v_side,
        "u_v_asymmetry": u_side - v_side,
        "random_in_mean": random_mean,
        "random_in_std": float(np.std(random_ratios)),
        "theoretical_expected": theoretical,
        "obs_over_random": obs_over_random,
        "dw_fro_norm": dW_fro,
    }


@torch.no_grad()
def analyze_matrix_three_way(W_rl, W_base, W_sft, k_values, n_random, device):
    """Three-way analysis for one weight matrix."""
    W_rl_f = W_rl.float().to(device)
    W_base_f = W_base.float().to(device)
    W_sft_f = W_sft.float().to(device)
    m, n = W_rl_f.shape

    # SVD of base (reference subspace)
    U_base, S_base, Vt_base = torch.linalg.svd(W_base_f, full_matrices=False)
    V_base = Vt_base.T

    # Deltas
    dW_rl = W_rl_f - W_base_f
    dW_sft = W_sft_f - W_base_f
    dW_rl_sft = W_rl_f - W_sft_f

    k_results = {}
    for k in k_values:
        k_actual = min(k, S_base.numel())
        U_k = U_base[:, :k_actual]
        V_k = V_base[:, :k_actual]

        # Three pairs, all projected onto base's subspace
        rl_base = analyze_delta(dW_rl, U_k, V_k, n_random, device)
        sft_base = analyze_delta(dW_sft, U_k, V_k, n_random, device)
        rl_sft = analyze_delta(dW_rl_sft, U_k, V_k, n_random, device)

        k_results[str(k_actual)] = {
            "k": k_actual,
            "rl_vs_base": rl_base,
            "sft_vs_base": sft_base,
            "rl_vs_sft": rl_sft,
        }

    return {"shape": [m, n], "k_analysis": k_results}


# ---------- visualization ----------
def generate_plots(results, plot_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plots")
        return

    os.makedirs(plot_dir, exist_ok=True)
    summary = results["summary"]

    # --- Plot 1: obs/random comparison across k values ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Three-way SVD: obs/random Comparison\n(RL-Base vs SFT-Base vs RL-SFT)", fontsize=13)

    for ax_idx, (pair_name, pair_label) in enumerate([
        ("rl_vs_base", "RL vs Base"),
        ("sft_vs_base", "SFT vs Base"),
        ("rl_vs_sft", "RL vs SFT"),
    ]):
        ax = axes[ax_idx]
        for comp, marker, color in [("attn", "o", "C0"), ("mlp", "s", "C1")]:
            ks = []
            vals = []
            for k_str, kd in sorted(summary["by_component_k"][comp].items(), key=lambda x: int(x[0])):
                if pair_name in kd:
                    ks.append(int(k_str))
                    vals.append(kd[pair_name]["obs_over_random_mean"])
            ax.plot(ks, vals, f"{marker}-", label=comp.upper(), color=color)

        ax.axhline(y=1.0, color="red", linestyle="--", linewidth=0.8, label="Random (1.0)")
        ax.set_xlabel("k")
        ax.set_ylabel("obs/random")
        ax.set_title(pair_label)
        ax.set_xscale("log", base=2)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "svd_three_way_obs_random.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- Plot 2: U-side vs V-side asymmetry ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("U-side vs V-side Asymmetry at k=256\n(Positive = U-side dominant, Negative = V-side dominant)", fontsize=13)

    per_matrix = results["per_matrix"]
    for ax_idx, (pair_name, pair_label) in enumerate([
        ("rl_vs_base", "RL vs Base"),
        ("sft_vs_base", "SFT vs Base"),
        ("rl_vs_sft", "RL vs SFT"),
    ]):
        ax = axes[ax_idx]
        attn_asym = []
        mlp_asym = []
        attn_layers = []
        mlp_layers = []

        for name, data in per_matrix.items():
            k_data = data.get("k_analysis", {}).get("256", {})
            pair_data = k_data.get(pair_name, {})
            if pair_data.get("skipped"):
                continue
            asym = pair_data.get("u_v_asymmetry", 0)
            layer = data["layer"]
            if data["component"] == "attn":
                attn_asym.append(asym)
                attn_layers.append(layer)
            else:
                mlp_asym.append(asym)
                mlp_layers.append(layer)

        ax.scatter(attn_layers, attn_asym, alpha=0.6, label="Attention", s=20)
        ax.scatter(mlp_layers, mlp_asym, alpha=0.6, label="MLP", marker="s", s=20)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("U-side - V-side ratio")
        ax.set_title(pair_label)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "svd_three_way_asymmetry.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plots to {plot_dir}/")


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(description="Three-way SVD Comparison")
    parser.add_argument("--rl_dir", required=True)
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--sft_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--n_random", type=int, default=N_RANDOM)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    k_values = K_VALUES
    n_random = args.n_random

    if args.dry_run:
        k_values = [64, 256]
        n_random = 10
        print("DRY RUN: layers [0, 27], k=[64, 256], n_random=10")

    t0 = time.time()

    # Load all three models to CPU
    print("Loading base model...")
    base_sd = AutoModelForCausalLM.from_pretrained(
        args.base_dir, torch_dtype=torch.float16, device_map="cpu").state_dict()
    print("Loading RL model...")
    rl_sd = AutoModelForCausalLM.from_pretrained(
        args.rl_dir, torch_dtype=torch.float16, device_map="cpu").state_dict()
    print("Loading SFT model...")
    sft_sd = AutoModelForCausalLM.from_pretrained(
        args.sft_dir, torch_dtype=torch.float16, device_map="cpu").state_dict()
    gc.collect()

    # Collect weight pairs
    pairs = []
    for name in sorted(base_sd.keys()):
        if param_wanted(name) and name in rl_sd and name in sft_sd:
            layer = get_layer_idx(name)
            if args.dry_run and layer not in [0, 27]:
                continue
            pairs.append(name)

    print(f"\nAnalyzing {len(pairs)} matrices × {len(k_values)} k × 3 pairs × {n_random} random")

    per_matrix = {}
    for name in tqdm(pairs, desc="Analyzing"):
        result = analyze_matrix_three_way(
            rl_sd[name], base_sd[name], sft_sd[name], k_values, n_random, device)
        result["layer"] = get_layer_idx(name)
        result["component"] = get_component(name)
        result["proj_type"] = get_proj_type(name)
        per_matrix[name] = result
        torch.cuda.empty_cache()

    # ---------- aggregate ----------
    summary = {"by_k": {}, "by_component_k": {"attn": {}, "mlp": {}}}

    for k in k_values:
        k_str = str(min(k, 3584))
        comp_data = {"attn": defaultdict(lambda: defaultdict(list)),
                     "mlp": defaultdict(lambda: defaultdict(list))}
        all_data = defaultdict(lambda: defaultdict(list))

        for name, data in per_matrix.items():
            if k_str not in data["k_analysis"]:
                continue
            k_res = data["k_analysis"][k_str]
            comp = data["component"]

            for pair_name in ["rl_vs_base", "sft_vs_base", "rl_vs_sft"]:
                pd = k_res[pair_name]
                if pd.get("skipped"):
                    continue
                for metric in ["obs_over_random", "bilateral_in_ratio", "u_side_ratio",
                               "v_side_ratio", "u_v_asymmetry", "dw_fro_norm"]:
                    if metric in pd:
                        all_data[pair_name][metric].append(pd[metric])
                        comp_data[comp][pair_name][metric].append(pd[metric])

        by_k_entry = {}
        for pair_name in ["rl_vs_base", "sft_vs_base", "rl_vs_sft"]:
            if all_data[pair_name]:
                by_k_entry[pair_name] = {
                    f"{m}_mean": float(np.mean(v)) for m, v in all_data[pair_name].items()
                }

        summary["by_k"][k_str] = by_k_entry

        for comp in ["attn", "mlp"]:
            comp_entry = {}
            for pair_name in ["rl_vs_base", "sft_vs_base", "rl_vs_sft"]:
                if comp_data[comp][pair_name]:
                    comp_entry[pair_name] = {
                        f"{m}_mean": float(np.mean(v)) for m, v in comp_data[comp][pair_name].items()
                    }
            if comp_entry:
                summary["by_component_k"][comp][k_str] = comp_entry

    # Key comparison: is SFT alignment pattern different from RL?
    k256 = summary["by_k"].get("256", {})
    rl_obs = k256.get("rl_vs_base", {}).get("obs_over_random_mean", 0)
    sft_obs = k256.get("sft_vs_base", {}).get("obs_over_random_mean", 0)
    rl_asym = k256.get("rl_vs_base", {}).get("u_v_asymmetry_mean", 0)
    sft_asym = k256.get("sft_vs_base", {}).get("u_v_asymmetry_mean", 0)

    key_comparison = {
        "at_k256": {
            "rl_obs_over_random": rl_obs,
            "sft_obs_over_random": sft_obs,
            "rl_u_v_asymmetry": rl_asym,
            "sft_u_v_asymmetry": sft_asym,
            "alignment_ratio_rl_over_sft": rl_obs / sft_obs if sft_obs > 1e-6 else float("inf"),
            "asymmetry_diff": rl_asym - sft_asym,
        },
        "interpretation": (
            f"RL alignment {rl_obs:.2f}x vs SFT alignment {sft_obs:.2f}x (both relative to random). "
            f"RL U-V asymmetry: {rl_asym:.4f}, SFT U-V asymmetry: {sft_asym:.4f}. "
            + ("SFT and RL show DIFFERENT alignment patterns — supports RL-specific mechanism."
               if abs(rl_obs - sft_obs) > 0.5 or abs(rl_asym - sft_asym) > 0.01
               else "SFT and RL show SIMILAR alignment patterns — fine-tuning generic effect.")
        ),
    }

    elapsed = time.time() - t0

    output = {
        "config": {
            "rl_dir": args.rl_dir,
            "base_dir": args.base_dir,
            "sft_dir": args.sft_dir,
            "k_values": k_values,
            "n_random": n_random,
            "n_matrices": len(per_matrix),
            "dry_run": args.dry_run,
            "note": "k/v_proj [512,3584]: k>512 clamped. Dropped k=1024 from analysis.",
        },
        "summary": summary,
        "key_comparison": key_comparison,
        "per_matrix": per_matrix,
        "elapsed_seconds": elapsed,
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output_path}")

    # Print key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS — Three-way SVD Comparison")
    print("=" * 60)
    for k_str in sorted(summary["by_k"].keys(), key=int):
        print(f"\nk={k_str}:")
        for pair in ["rl_vs_base", "sft_vs_base", "rl_vs_sft"]:
            pd = summary["by_k"][k_str].get(pair, {})
            obs = pd.get("obs_over_random_mean", 0)
            u = pd.get("u_side_ratio_mean", 0)
            v = pd.get("v_side_ratio_mean", 0)
            asym = pd.get("u_v_asymmetry_mean", 0)
            print(f"  {pair:>15s}: obs/rand={obs:.2f}x  U={u:.4f} V={v:.4f} asym={asym:+.4f}")

    print(f"\n{key_comparison['interpretation']}")

    # Generate plots
    plot_dir = os.path.join(os.path.dirname(args.output_path), "plots")
    generate_plots(output, plot_dir)


if __name__ == "__main__":
    main()
