#!/usr/bin/env python3
"""
Phase 1b补充: SVD Random Baseline Analysis

修正 exp_003 方法论缺陷：97.6% out-of-subspace energy 是高维几何的数学必然。
本脚本添加随机基线校准，给出正确的统计解读。

分析内容：
1. 随机基线：100个同范数随机矩阵的 in-subspace ratio → observed/random 比值
2. k 灵敏度：k ∈ {64, 128, 256, 512, 1024}，in-ratio 和 observed/random 如何变化
3. 单侧投影：分别报告 U侧 和 V侧 投影能量
4. 维度归一化对比：in_ratio / (k²/(m×n))，attention vs MLP

Usage:
    python scripts/svd_random_baseline.py \
        --rl_dir /root/autodl-tmp/models/MEM1 \
        --base_dir /root/autodl-tmp/models/Qwen2.5-7B \
        --output_path /root/autodl-tmp/learned_traj_compress/artifacts/exp_003b_random_baseline.json \
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

K_VALUES = [64, 128, 256, 512, 1024]
N_RANDOM = 100


def param_wanted(name: str) -> bool:
    return any(name.endswith(p) for p in WEIGHT_PATTERNS)


def get_layer_idx(name: str) -> int:
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            return int(parts[i + 1])
    return -1


def get_component(name: str) -> str:
    if any(p in name for p in ATTN_PATTERNS):
        return "attn"
    elif any(p in name for p in MLP_PATTERNS):
        return "mlp"
    return "other"


def get_proj_type(name: str) -> str:
    for pattern in WEIGHT_PATTERNS:
        suffix = pattern.lstrip(".")
        if name.endswith(suffix):
            return suffix.replace(".weight", "")
    return "unknown"


# ---------- core analysis ----------
@torch.no_grad()
def compute_random_baseline_in_ratio(m: int, n: int, U_base_k: torch.Tensor,
                                      V_base_k: torch.Tensor, dW_fro: float,
                                      n_random: int = 100, device: str = "cuda") -> dict:
    """
    Generate n_random random matrices of shape [m, n] with Frobenius norm = dW_fro.
    Compute in-subspace ratio for each under bilateral projection U_base_k.T @ R @ V_base_k.

    Returns: mean, std, min, max of in-ratio for random matrices.
    """
    k = U_base_k.shape[1]
    ratios = []

    for _ in range(n_random):
        R = torch.randn(m, n, device=device, dtype=torch.float32)
        R_fro = torch.norm(R, p="fro")
        if R_fro > 1e-12:
            R = R * (dW_fro / R_fro.item())

        proj = U_base_k.T @ R @ V_base_k  # [k, k]
        proj_fro2 = torch.norm(proj, p="fro") ** 2
        R_fro2 = torch.norm(R, p="fro") ** 2

        if R_fro2 > 1e-12:
            ratios.append((proj_fro2 / R_fro2).item())

    ratios_arr = np.array(ratios)
    # Theoretical expected value: k^2 / (m * n)
    theoretical = k ** 2 / (m * n)

    return {
        "mean": float(ratios_arr.mean()),
        "std": float(ratios_arr.std()),
        "min": float(ratios_arr.min()),
        "max": float(ratios_arr.max()),
        "theoretical_expected": float(theoretical),
        "k": k,
        "m": m,
        "n": n,
    }


@torch.no_grad()
def compute_one_sided_projections(dW: torch.Tensor, U_base_k: torch.Tensor,
                                   V_base_k: torch.Tensor) -> dict:
    """
    Compute one-sided projection energies:
    - U-side: ||U_base_k.T @ ΔW||² / ||ΔW||²
    - V-side: ||ΔW @ V_base_k||² / ||ΔW||²
    """
    dW_fro2 = torch.norm(dW, p="fro") ** 2
    if dW_fro2 < 1e-12:
        return {"u_side_ratio": 0.0, "v_side_ratio": 0.0}

    # U-side: project rows of ΔW onto U_base_k
    u_proj = U_base_k.T @ dW  # [k, n]
    u_proj_fro2 = torch.norm(u_proj, p="fro") ** 2

    # V-side: project columns of ΔW onto V_base_k
    v_proj = dW @ V_base_k  # [m, k]
    v_proj_fro2 = torch.norm(v_proj, p="fro") ** 2

    return {
        "u_side_ratio": float((u_proj_fro2 / dW_fro2).item()),
        "v_side_ratio": float((v_proj_fro2 / dW_fro2).item()),
    }


@torch.no_grad()
def analyze_matrix_full(W_rl: torch.Tensor, W_base: torch.Tensor,
                        k_values: list[int], n_random: int,
                        device: str = "cuda") -> dict:
    """
    Full analysis for one weight matrix pair across multiple k values.
    """
    W_rl_f = W_rl.float().to(device)
    W_base_f = W_base.float().to(device)
    m, n = W_rl_f.shape

    dW = W_rl_f - W_base_f
    dW_fro = torch.norm(dW, p="fro").item()
    dW_fro2 = dW_fro ** 2

    if dW_fro < 1e-12:
        return {"skipped": True, "reason": "zero_delta"}

    # Full SVD of base (compute once, reuse for all k)
    U_base, S_base, Vt_base = torch.linalg.svd(W_base_f, full_matrices=False)
    V_base = Vt_base.T

    k_results = {}
    for k in k_values:
        k_actual = min(k, S_base.numel())
        U_k = U_base[:, :k_actual]
        V_k = V_base[:, :k_actual]

        # Bilateral in-subspace ratio (original metric)
        proj_bilateral = U_k.T @ dW @ V_k  # [k, k]
        bilateral_in_ratio = (torch.norm(proj_bilateral, p="fro") ** 2 / dW_fro2)
        bilateral_in_ratio = float(bilateral_in_ratio.item())

        # One-sided projections
        one_sided = compute_one_sided_projections(dW, U_k, V_k)

        # Random baseline
        random_stats = compute_random_baseline_in_ratio(
            m, n, U_k, V_k, dW_fro, n_random=n_random, device=device
        )

        # Observed / random ratio
        obs_over_random = bilateral_in_ratio / random_stats["mean"] if random_stats["mean"] > 1e-12 else float("inf")
        obs_over_theoretical = bilateral_in_ratio / random_stats["theoretical_expected"] if random_stats["theoretical_expected"] > 1e-12 else float("inf")

        # Dimension-normalized: in_ratio / (k²/(m*n))
        dim_normalized = bilateral_in_ratio / (k_actual ** 2 / (m * n))

        k_results[str(k_actual)] = {
            "k": k_actual,
            "bilateral_in_ratio": bilateral_in_ratio,
            "bilateral_out_ratio": 1.0 - bilateral_in_ratio,
            "u_side_in_ratio": one_sided["u_side_ratio"],
            "v_side_in_ratio": one_sided["v_side_ratio"],
            "random_in_ratio_mean": random_stats["mean"],
            "random_in_ratio_std": random_stats["std"],
            "random_in_ratio_theoretical": random_stats["theoretical_expected"],
            "obs_over_random": float(obs_over_random),
            "obs_over_theoretical": float(obs_over_theoretical),
            "dim_normalized": float(dim_normalized),
        }

    return {
        "shape": [m, n],
        "dw_fro_norm": dW_fro,
        "k_analysis": k_results,
    }


# ---------- visualization ----------
def generate_plots(results: dict, plot_dir: str):
    """Generate visualization plots for random baseline analysis."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plots")
        return

    os.makedirs(plot_dir, exist_ok=True)

    per_matrix = results["per_matrix"]
    k_values = results["config"]["k_values"]

    # --- Plot 1: k-sensitivity of obs/random ratio (attn vs MLP) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Phase 1b补充: Random Baseline Calibration\n(MEM1 RL vs Qwen2.5-7B Base)", fontsize=13)

    # Collect per-component per-k stats
    attn_obs_ratios = defaultdict(list)
    mlp_obs_ratios = defaultdict(list)
    attn_in_ratios = defaultdict(list)
    mlp_in_ratios = defaultdict(list)

    for name, data in per_matrix.items():
        if data.get("skipped"):
            continue
        component = data["component"]
        for k_str, kd in data["k_analysis"].items():
            k = kd["k"]
            if component == "attn":
                attn_obs_ratios[k].append(kd["obs_over_random"])
                attn_in_ratios[k].append(kd["bilateral_in_ratio"])
            else:
                mlp_obs_ratios[k].append(kd["obs_over_random"])
                mlp_in_ratios[k].append(kd["bilateral_in_ratio"])

    # 1a: obs/random ratio vs k
    ax = axes[0]
    ks_sorted = sorted(attn_obs_ratios.keys())
    attn_means = [np.mean(attn_obs_ratios[k]) for k in ks_sorted]
    mlp_means = [np.mean(mlp_obs_ratios[k]) for k in ks_sorted]
    attn_stds = [np.std(attn_obs_ratios[k]) for k in ks_sorted]
    mlp_stds = [np.std(mlp_obs_ratios[k]) for k in ks_sorted]

    ax.errorbar(ks_sorted, attn_means, yerr=attn_stds, fmt="o-", label="Attention", capsize=3)
    ax.errorbar(ks_sorted, mlp_means, yerr=mlp_stds, fmt="s-", label="MLP", capsize=3)
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=0.8, label="Random baseline (1.0)")
    ax.set_xlabel("k (subspace dimension)")
    ax.set_ylabel("Observed / Random in-ratio")
    ax.set_title("Calibrated Alignment: obs/random")
    ax.set_xscale("log", base=2)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 1b: raw in-ratio vs k, with theoretical line
    ax = axes[1]
    attn_in_means = [np.mean(attn_in_ratios[k]) for k in ks_sorted]
    mlp_in_means = [np.mean(mlp_in_ratios[k]) for k in ks_sorted]
    # Theoretical for each shape
    # attn shapes: q/o = [3584, 3584], k/v = [512, 3584]
    # mlp shapes: gate/up = [18944, 3584], down = [3584, 18944]
    # Use approximate theoretical for the dominant shape
    theoretical_attn = [k**2 / (3584 * 3584) for k in ks_sorted]
    theoretical_mlp = [k**2 / (18944 * 3584) for k in ks_sorted]

    ax.plot(ks_sorted, attn_in_means, "o-", label="Attention (observed)")
    ax.plot(ks_sorted, mlp_in_means, "s-", label="MLP (observed)")
    ax.plot(ks_sorted, theoretical_attn, "o--", alpha=0.5, label="Attention (random expected)")
    ax.plot(ks_sorted, theoretical_mlp, "s--", alpha=0.5, label="MLP (random expected)")
    ax.set_xlabel("k (subspace dimension)")
    ax.set_ylabel("Bilateral in-subspace ratio")
    ax.set_title("Raw in-ratio vs k")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 1c: U-side vs V-side ratio at k=256
    ax = axes[2]
    u_sides_attn = []
    v_sides_attn = []
    u_sides_mlp = []
    v_sides_mlp = []

    for name, data in per_matrix.items():
        if data.get("skipped"):
            continue
        # Find k=256 entry
        for k_str, kd in data["k_analysis"].items():
            if kd["k"] == 256:
                if data["component"] == "attn":
                    u_sides_attn.append(kd["u_side_in_ratio"])
                    v_sides_attn.append(kd["v_side_in_ratio"])
                else:
                    u_sides_mlp.append(kd["u_side_in_ratio"])
                    v_sides_mlp.append(kd["v_side_in_ratio"])

    ax.scatter(u_sides_attn, v_sides_attn, alpha=0.6, label=f"Attention (n={len(u_sides_attn)})", s=20)
    ax.scatter(u_sides_mlp, v_sides_mlp, alpha=0.6, label=f"MLP (n={len(u_sides_mlp)})", marker="s", s=20)
    # Add y=x line
    lim = max(max(u_sides_attn + u_sides_mlp, default=0.1), max(v_sides_attn + v_sides_mlp, default=0.1)) * 1.1
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, label="y=x")
    ax.set_xlabel("U-side in-ratio (row space)")
    ax.set_ylabel("V-side in-ratio (column space)")
    ax.set_title("One-sided Projections (k=256)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "svd_random_baseline.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {plot_dir}/svd_random_baseline.png")

    # --- Plot 2: Layer-wise obs/random at k=256 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Layer-wise Calibrated Alignment (k=256)", fontsize=13)

    layer_attn_obs = defaultdict(list)
    layer_mlp_obs = defaultdict(list)
    layer_attn_dim_norm = defaultdict(list)
    layer_mlp_dim_norm = defaultdict(list)

    for name, data in per_matrix.items():
        if data.get("skipped"):
            continue
        layer = data["layer"]
        for k_str, kd in data["k_analysis"].items():
            if kd["k"] == 256:
                if data["component"] == "attn":
                    layer_attn_obs[layer].append(kd["obs_over_random"])
                    layer_attn_dim_norm[layer].append(kd["dim_normalized"])
                else:
                    layer_mlp_obs[layer].append(kd["obs_over_random"])
                    layer_mlp_dim_norm[layer].append(kd["dim_normalized"])

    layers = sorted(set(list(layer_attn_obs.keys()) + list(layer_mlp_obs.keys())))

    ax = axes[0]
    ax.bar(np.array(layers) - 0.2, [np.mean(layer_attn_obs.get(l, [0])) for l in layers],
           0.4, label="Attention", alpha=0.8)
    ax.bar(np.array(layers) + 0.2, [np.mean(layer_mlp_obs.get(l, [0])) for l in layers],
           0.4, label="MLP", alpha=0.8)
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=0.8, label="Random (1.0)")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Observed / Random")
    ax.set_title("obs/random by Layer (k=256)")
    ax.legend()

    ax = axes[1]
    ax.bar(np.array(layers) - 0.2, [np.mean(layer_attn_dim_norm.get(l, [0])) for l in layers],
           0.4, label="Attention", alpha=0.8)
    ax.bar(np.array(layers) + 0.2, [np.mean(layer_mlp_dim_norm.get(l, [0])) for l in layers],
           0.4, label="MLP", alpha=0.8)
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=0.8, label="Random (1.0)")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Dimension-normalized ratio")
    ax.set_title("Dimension-normalized Alignment by Layer (k=256)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "svd_layer_calibrated.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {plot_dir}/svd_layer_calibrated.png")


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(description="Phase 1b补充: SVD Random Baseline Analysis")
    parser.add_argument("--rl_dir", type=str, required=True)
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--n_random", type=int, default=N_RANDOM)
    parser.add_argument("--dry_run", action="store_true",
                        help="Only analyze layers 0 and 27, k=[64,256], n_random=10")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    k_values = K_VALUES
    n_random = args.n_random

    if args.dry_run:
        k_values = [64, 256]
        n_random = 10
        print("DRY RUN: layers [0, 27], k=[64, 256], n_random=10")

    t0 = time.time()

    # Load models (CPU to save GPU memory, move per-matrix)
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_dir, torch_dtype=torch.float16, device_map="cpu"
    )
    print("Loading RL model...")
    rl_model = AutoModelForCausalLM.from_pretrained(
        args.rl_dir, torch_dtype=torch.float16, device_map="cpu"
    )

    base_sd = base_model.state_dict()
    rl_sd = rl_model.state_dict()

    # Free model objects
    del base_model, rl_model
    gc.collect()

    # Collect weight pairs
    pairs = []
    for name in sorted(base_sd.keys()):
        if param_wanted(name) and name in rl_sd:
            layer = get_layer_idx(name)
            if args.dry_run and layer not in [0, 27]:
                continue
            pairs.append(name)

    print(f"\nAnalyzing {len(pairs)} weight matrices × {len(k_values)} k values × {n_random} random samples")

    per_matrix = {}

    for name in tqdm(pairs, desc="Analyzing"):
        W_rl = rl_sd[name]
        W_base = base_sd[name]

        result = analyze_matrix_full(W_rl, W_base, k_values, n_random, device=device)
        result["layer"] = get_layer_idx(name)
        result["component"] = get_component(name)
        result["proj_type"] = get_proj_type(name)

        per_matrix[name] = result

        # Free GPU memory
        torch.cuda.empty_cache()

    # ---------- aggregate ----------
    summary = {
        "by_k": {},
        "by_component_k": {"attn": {}, "mlp": {}},
    }

    for k in k_values:
        k_str = str(min(k, 3584))  # clamp to actual

        all_bilateral = []
        all_obs_random = []
        all_obs_theo = []
        all_dim_norm = []
        all_u_side = []
        all_v_side = []

        comp_stats = {"attn": defaultdict(list), "mlp": defaultdict(list)}

        for name, data in per_matrix.items():
            if data.get("skipped"):
                continue
            k_actual_str = str(min(k, 3584))
            if k_actual_str not in data["k_analysis"]:
                continue
            kd = data["k_analysis"][k_actual_str]
            comp = data["component"]

            all_bilateral.append(kd["bilateral_in_ratio"])
            all_obs_random.append(kd["obs_over_random"])
            all_obs_theo.append(kd["obs_over_theoretical"])
            all_dim_norm.append(kd["dim_normalized"])
            all_u_side.append(kd["u_side_in_ratio"])
            all_v_side.append(kd["v_side_in_ratio"])

            comp_stats[comp]["bilateral"].append(kd["bilateral_in_ratio"])
            comp_stats[comp]["obs_over_random"].append(kd["obs_over_random"])
            comp_stats[comp]["dim_normalized"].append(kd["dim_normalized"])
            comp_stats[comp]["u_side"].append(kd["u_side_in_ratio"])
            comp_stats[comp]["v_side"].append(kd["v_side_in_ratio"])

        if all_bilateral:
            summary["by_k"][k_str] = {
                "k": int(k_str),
                "bilateral_in_ratio_mean": float(np.mean(all_bilateral)),
                "bilateral_in_ratio_std": float(np.std(all_bilateral)),
                "obs_over_random_mean": float(np.mean(all_obs_random)),
                "obs_over_random_std": float(np.std(all_obs_random)),
                "obs_over_theoretical_mean": float(np.mean(all_obs_theo)),
                "dim_normalized_mean": float(np.mean(all_dim_norm)),
                "u_side_mean": float(np.mean(all_u_side)),
                "v_side_mean": float(np.mean(all_v_side)),
            }

            for comp in ["attn", "mlp"]:
                if comp_stats[comp]["bilateral"]:
                    summary["by_component_k"][comp][k_str] = {
                        "bilateral_in_ratio_mean": float(np.mean(comp_stats[comp]["bilateral"])),
                        "obs_over_random_mean": float(np.mean(comp_stats[comp]["obs_over_random"])),
                        "dim_normalized_mean": float(np.mean(comp_stats[comp]["dim_normalized"])),
                        "u_side_mean": float(np.mean(comp_stats[comp]["u_side"])),
                        "v_side_mean": float(np.mean(comp_stats[comp]["v_side"])),
                    }

    # ---------- corrected go/no-go ----------
    # Key question: is obs/random > 1 (RL changes align MORE than random with base subspace)?
    # Or < 1 (RL changes are MORE orthogonal than random)?
    k256_data = summary["by_k"].get("256", {})
    obs_random_256 = k256_data.get("obs_over_random_mean", 0)

    corrected_interpretation = {
        "obs_over_random_at_k256": obs_random_256,
        "alignment_direction": "supra-random" if obs_random_256 > 1.0 else "sub-random",
        "interpretation": (
            f"RL weight changes show {obs_random_256:.1f}x the base-subspace alignment "
            f"expected from random perturbations. "
            + ("This indicates RL changes are PREFERENTIALLY aligned with the base model's "
               "principal directions — opposite to the 'orthogonal new directions' narrative."
               if obs_random_256 > 1.0 else
               "This indicates RL changes actively AVOID the base model's principal directions.")
        ),
    }

    elapsed = time.time() - t0

    output = {
        "config": {
            "rl_dir": args.rl_dir,
            "base_dir": args.base_dir,
            "k_values": k_values,
            "n_random": n_random,
            "dry_run": args.dry_run,
            "n_matrices": len(per_matrix),
        },
        "summary": summary,
        "corrected_interpretation": corrected_interpretation,
        "per_matrix": per_matrix,
        "elapsed_seconds": elapsed,
    }

    # Save
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output_path}")
    print(f"Elapsed: {elapsed:.0f}s")

    # Print key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS — Random Baseline Calibration")
    print("=" * 60)

    for k_str, kd in sorted(summary["by_k"].items(), key=lambda x: int(x[0])):
        print(f"\nk={kd['k']}:")
        print(f"  Bilateral in-ratio: {kd['bilateral_in_ratio_mean']:.4f} ± {kd['bilateral_in_ratio_std']:.4f}")
        print(f"  Random expected:    {kd['bilateral_in_ratio_mean']/kd['obs_over_random_mean']:.4f}")
        print(f"  Observed/Random:    {kd['obs_over_random_mean']:.2f}x")
        print(f"  U-side ratio:       {kd['u_side_mean']:.4f}")
        print(f"  V-side ratio:       {kd['v_side_mean']:.4f}")
        print(f"  Dim-normalized:     {kd['dim_normalized_mean']:.2f}")

    print(f"\nAttn vs MLP at k=256:")
    for comp in ["attn", "mlp"]:
        cd = summary["by_component_k"][comp].get("256", {})
        if cd:
            print(f"  {comp}: obs/random={cd['obs_over_random_mean']:.2f}x, "
                  f"dim_norm={cd['dim_normalized_mean']:.2f}, "
                  f"U={cd['u_side_mean']:.4f}, V={cd['v_side_mean']:.4f}")

    print(f"\n{corrected_interpretation['interpretation']}")

    # Generate plots
    plot_dir = os.path.join(os.path.dirname(args.output_path), "plots")
    generate_plots(output, plot_dir)


if __name__ == "__main__":
    main()
