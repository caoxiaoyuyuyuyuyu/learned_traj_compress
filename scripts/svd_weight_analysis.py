#!/usr/bin/env python3
"""
Phase 1b: SVD Weight Analysis — MEM1 RL vs Qwen2.5-7B Base

Compare weight-space structure between RL-trained and base models:
1. Effective rank (Shannon entropy of normalized singular values)
2. Singular value spectrum comparison
3. Principal angle rotation between top-k subspaces
4. ΔW decomposition: how much change lives in/outside base subspace

Also computes Holm-Bonferroni correction for Phase 1a statistical tests.

Usage:
    python scripts/svd_weight_analysis.py \
        --rl_dir /root/autodl-tmp/models/MEM1 \
        --base_dir /root/autodl-tmp/models/Qwen2.5-7B \
        --output_path /root/autodl-tmp/learned_traj_compress/artifacts/exp_003_svd_analysis.json \
        --top_k 256
"""

import argparse
import gc
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

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


def param_wanted(name: str) -> bool:
    return any(name.endswith(p) for p in WEIGHT_PATTERNS)


def get_layer_idx(name: str) -> int:
    """Extract layer index from parameter name like 'model.layers.5.self_attn.q_proj.weight'."""
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            return int(parts[i + 1])
    return -1


def get_component(name: str) -> str:
    """Return 'attn' or 'mlp' based on parameter name."""
    if any(p in name for p in ATTN_PATTERNS):
        return "attn"
    elif any(p in name for p in MLP_PATTERNS):
        return "mlp"
    return "other"


def get_proj_type(name: str) -> str:
    """Extract projection type like 'q_proj' from parameter name."""
    for pattern in WEIGHT_PATTERNS:
        suffix = pattern.lstrip(".")
        if name.endswith(suffix):
            return suffix.replace(".weight", "")
    return "unknown"


# ---------- SVD metrics ----------
@torch.no_grad()
def compute_effective_rank(S: torch.Tensor) -> float:
    """Effective rank = exp(Shannon entropy of normalized singular values)."""
    s = S.float()
    s = s[s > 1e-10]  # filter near-zero
    if len(s) == 0:
        return 0.0
    p = s / s.sum()
    entropy = -(p * torch.log(p)).sum().item()
    return math.exp(entropy)


@torch.no_grad()
def compute_energy_ratio(S: torch.Tensor, k: int) -> float:
    """Fraction of total energy in top-k singular values."""
    s2 = S.float() ** 2
    total = s2.sum().item()
    if total < 1e-12:
        return 0.0
    top_k = s2[:k].sum().item()
    return top_k / total


@torch.no_grad()
def compute_principal_angles(U_a: torch.Tensor, U_b: torch.Tensor) -> torch.Tensor:
    """Principal angles between two subspaces (columns of U_a and U_b)."""
    C = U_a.T @ U_b
    s = torch.linalg.svdvals(C)
    s = s.clamp(-1.0, 1.0)
    return torch.arccos(s)


@torch.no_grad()
def compute_subspace_alignment(U_a: torch.Tensor, U_b: torch.Tensor) -> float:
    """Mean max alignment between basis vectors: mean of max |u_a^T @ U_b| per u_a."""
    alignment = torch.abs(U_a.T @ U_b).max(dim=1).values
    return alignment.mean().item()


@torch.no_grad()
def compute_projection_distance(U_a: torch.Tensor, U_b: torch.Tensor) -> float:
    """Frobenius distance between projection matrices: ‖U_a U_a^T - U_b U_b^T‖_F."""
    # Efficient: sqrt(2k - 2‖U_a^T U_b‖_F^2)
    C = U_a.T @ U_b
    k = U_a.shape[1]
    val = 2 * k - 2 * torch.norm(C, p="fro") ** 2
    return math.sqrt(max(0, val.item()))


@torch.no_grad()
def analyze_delta_w(W_rl: torch.Tensor, W_base: torch.Tensor, U_base: torch.Tensor,
                    V_base: torch.Tensor, top_k: int) -> dict:
    """Analyze ΔW = W_rl - W_base relative to base model's SVD subspace."""
    dW = (W_rl - W_base).float()
    dW_fro2 = torch.norm(dW, p="fro") ** 2

    if dW_fro2 < 1e-12:
        return {"dw_fro_norm": 0.0, "in_subspace_energy": 0.0, "out_subspace_energy": 0.0,
                "diag_energy": 0.0, "rot_energy": 0.0}

    # Project ΔW into base subspace: U_b^T @ ΔW @ V_b
    proj = U_base.T @ dW @ V_base  # [k, k]
    proj_fro2 = torch.norm(proj, p="fro") ** 2
    diag_fro2 = torch.diag(proj).float().pow(2).sum()

    in_ratio = (proj_fro2 / dW_fro2).item()
    out_ratio = 1.0 - in_ratio
    diag_ratio = (diag_fro2 / dW_fro2).item()
    rot_ratio = ((proj_fro2 - diag_fro2) / dW_fro2).item()

    return {
        "dw_fro_norm": math.sqrt(dW_fro2.item()),
        "in_subspace_energy": float(in_ratio),
        "out_subspace_energy": float(out_ratio),
        "diag_energy": float(diag_ratio),
        "rot_energy": float(rot_ratio),
    }


@torch.no_grad()
def analyze_weight_pair(W_rl: torch.Tensor, W_base: torch.Tensor,
                        top_k: int, device: str = "cuda") -> dict:
    """Full SVD analysis for a single weight matrix pair."""
    W_rl_f = W_rl.float().to(device)
    W_base_f = W_base.float().to(device)

    # SVD
    U_rl, S_rl, Vt_rl = torch.linalg.svd(W_rl_f, full_matrices=False)
    U_base, S_base, Vt_base = torch.linalg.svd(W_base_f, full_matrices=False)
    V_rl = Vt_rl.T
    V_base = Vt_base.T

    k = min(top_k, S_rl.numel(), S_base.numel())

    # 1. Effective rank
    eff_rank_rl = compute_effective_rank(S_rl)
    eff_rank_base = compute_effective_rank(S_base)

    # 2. Energy concentration
    energy_top64_rl = compute_energy_ratio(S_rl, 64)
    energy_top64_base = compute_energy_ratio(S_base, 64)
    energy_topk_rl = compute_energy_ratio(S_rl, k)
    energy_topk_base = compute_energy_ratio(S_base, k)

    # 3. Singular value stats
    rel_change = ((S_rl[:k] - S_base[:k]) / (S_base[:k] + 1e-12)).cpu()
    s_rl_top10 = S_rl[:10].cpu().tolist()
    s_base_top10 = S_base[:10].cpu().tolist()
    s_rl_tail10 = S_rl[-10:].cpu().tolist()
    s_base_tail10 = S_base[-10:].cpu().tolist()

    # 4. Principal angles (top-k subspaces)
    U_angles = compute_principal_angles(U_base[:, :k], U_rl[:, :k])
    V_angles = compute_principal_angles(V_base[:, :k], V_rl[:, :k])

    # 5. Subspace alignment
    align_U = compute_subspace_alignment(U_base[:, :k], U_rl[:, :k])
    align_V = compute_subspace_alignment(V_base[:, :k], V_rl[:, :k])

    # 6. Projection distance
    proj_dist_U = compute_projection_distance(U_base[:, :k], U_rl[:, :k])
    proj_dist_V = compute_projection_distance(V_base[:, :k], V_rl[:, :k])

    # 7. ΔW analysis
    delta_stats = analyze_delta_w(W_rl_f, W_base_f, U_base[:, :k], V_base[:, :k], k)

    result = {
        "shape": list(W_rl.shape),
        "top_k": k,
        # Effective rank
        "eff_rank_rl": float(eff_rank_rl),
        "eff_rank_base": float(eff_rank_base),
        "eff_rank_delta": float(eff_rank_rl - eff_rank_base),
        "eff_rank_ratio": float(eff_rank_rl / (eff_rank_base + 1e-12)),
        # Energy concentration
        "energy_top64_rl": float(energy_top64_rl),
        "energy_top64_base": float(energy_top64_base),
        "energy_topk_rl": float(energy_topk_rl),
        "energy_topk_base": float(energy_topk_base),
        # Singular values
        "s_rl_top10": [float(x) for x in s_rl_top10],
        "s_base_top10": [float(x) for x in s_base_top10],
        "s_rl_tail10": [float(x) for x in s_rl_tail10],
        "s_base_tail10": [float(x) for x in s_base_tail10],
        "rel_s_change_mean": float(rel_change.mean()),
        "rel_s_change_std": float(rel_change.std()),
        "rel_s_change_max": float(rel_change.max()),
        # Principal angles (in degrees)
        "U_angle_mean_deg": float(U_angles.mean() * 180 / math.pi),
        "U_angle_max_deg": float(U_angles.max() * 180 / math.pi),
        "V_angle_mean_deg": float(V_angles.mean() * 180 / math.pi),
        "V_angle_max_deg": float(V_angles.max() * 180 / math.pi),
        # Subspace alignment
        "align_U": float(align_U),
        "align_V": float(align_V),
        # Projection distance
        "proj_dist_U": float(proj_dist_U),
        "proj_dist_V": float(proj_dist_V),
        # ΔW decomposition
        **delta_stats,
    }

    return result


# ---------- Phase 1a Holm-Bonferroni correction ----------
def holm_bonferroni_correction(p_values: list[tuple[str, float]]) -> list[dict]:
    """Apply Holm-Bonferroni correction to a list of (label, p_value) pairs."""
    n = len(p_values)
    sorted_pv = sorted(p_values, key=lambda x: x[1])
    results = []
    for i, (label, p) in enumerate(sorted_pv):
        adjusted_p = min(1.0, p * (n - i))
        results.append({
            "test": label,
            "raw_p": float(p),
            "adjusted_p": float(adjusted_p),
            "significant_005": bool(adjusted_p < 0.05),
            "rank": i + 1,
        })
    return results


def compute_cohens_d(group1: list, group2: list) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_std = math.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    if pooled_std < 1e-12:
        return 0.0
    return float((m1 - m2) / pooled_std)


def phase1a_supplementary(phase1a_path: str) -> dict:
    """Compute Holm-Bonferroni correction and Cohen's d for Phase 1a results."""
    if not os.path.exists(phase1a_path):
        return {"error": f"Phase 1a results not found: {phase1a_path}"}

    with open(phase1a_path) as f:
        data = json.load(f)

    stat_tests = data.get("statistical_tests", {})
    p_values = []
    for test_name, test_data in stat_tests.items():
        if "p_value" in test_data:
            p_values.append((test_name, test_data["p_value"]))

    holm_results = holm_bonferroni_correction(p_values)

    # Cohen's d from per-sample data
    cohens_d = {}
    per_sample = data.get("per_sample_details", {}).get("mem1", {})
    if per_sample:
        for n_high in ["8", "12", "16"]:
            for n_low in ["1", "2"]:
                if n_high in per_sample and n_low in per_sample:
                    em_high = [s["em_mean"] for s in per_sample[n_high]]
                    em_low = [s["em_mean"] for s in per_sample[n_low]]
                    d = compute_cohens_d(em_low, em_high)
                    cohens_d[f"mem1_n{n_high}_vs_n{n_low}"] = {
                        "cohens_d": d,
                        "interpretation": "large" if abs(d) >= 0.8 else "medium" if abs(d) >= 0.5 else "small",
                    }

    return {
        "holm_bonferroni": holm_results,
        "cohens_d": cohens_d,
    }


# ---------- visualization ----------
def generate_plots(layer_depth_trend: dict, per_matrix_results: dict, plot_dir: str):
    """Generate visualization plots for SVD analysis results."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plots")
        return

    os.makedirs(plot_dir, exist_ok=True)

    layers = sorted(int(k) for k in layer_depth_trend.keys())
    trend = {int(k): v for k, v in layer_depth_trend.items()}

    # --- Plot 1: Layer depth vs eff_rank_delta (attn vs mlp) ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Phase 1b: SVD Analysis — Layer Depth Trends\n(MEM1 RL vs Qwen2.5-7B Base)", fontsize=13)

    # 1a: Effective rank delta
    ax = axes[0, 0]
    attn_vals = [trend[l].get("attn_eff_rank_delta_mean", 0) for l in layers]
    mlp_vals = [trend[l].get("mlp_eff_rank_delta_mean", 0) for l in layers]
    ax.bar(np.array(layers) - 0.2, attn_vals, 0.4, label="Attention", alpha=0.8)
    ax.bar(np.array(layers) + 0.2, mlp_vals, 0.4, label="MLP", alpha=0.8)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Effective Rank Delta (RL - Base)")
    ax.set_title("Mode Expansion by Layer")
    ax.legend()

    # 1b: U angle mean
    ax = axes[0, 1]
    attn_angles = [trend[l].get("attn_U_angle_mean", 0) for l in layers]
    mlp_angles = [trend[l].get("mlp_U_angle_mean", 0) for l in layers]
    ax.plot(layers, attn_angles, "o-", label="Attention", markersize=4)
    ax.plot(layers, mlp_angles, "s-", label="MLP", markersize=4)
    ax.axhline(y=5.0, color="red", linestyle="--", linewidth=0.8, label="5° threshold")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Mean Principal Angle (degrees)")
    ax.set_title("Subspace Rotation by Layer")
    ax.legend()

    # 1c: Out-of-subspace energy
    ax = axes[1, 0]
    out_vals = [trend[l]["out_subspace_energy_mean"] for l in layers]
    ax.bar(layers, out_vals, color="coral", alpha=0.8)
    ax.axhline(y=0.1, color="red", linestyle="--", linewidth=0.8, label="10% threshold")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Out-of-Subspace Energy Ratio")
    ax.set_title("ΔW New Direction Energy by Layer")
    ax.legend()

    # 1d: ΔW Frobenius norm
    ax = axes[1, 1]
    dw_vals = [trend[l]["dw_fro_norm_mean"] for l in layers]
    ax.bar(layers, dw_vals, color="steelblue", alpha=0.8)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Mean ‖ΔW‖_F")
    ax.set_title("Weight Change Magnitude by Layer")

    plt.tight_layout()
    path1 = os.path.join(plot_dir, "svd_layer_depth_trend.png")
    plt.savefig(path1, dpi=150)
    plt.close()
    print(f"  Saved: {path1}")

    # --- Plot 2: Per-projection heatmap ---
    proj_types = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    metric = "eff_rank_delta"

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Phase 1b: Per-Projection SVD Metrics (Layer × Projection)", fontsize=13)

    for idx, (metric_key, title, cmap) in enumerate([
        ("eff_rank_delta", "Effective Rank Delta", "RdBu"),
        ("U_angle_mean", "U Angle (deg)", "YlOrRd"),
        ("out_energy", "Out-Subspace Energy", "YlOrRd"),
    ]):
        ax = axes[idx]
        data_matrix = np.zeros((len(layers), len(proj_types)))
        for i, l in enumerate(layers):
            for j, proj in enumerate(proj_types):
                key = f"{proj}_{metric_key}"
                data_matrix[i, j] = trend[l].get(key, 0)

        im = ax.imshow(data_matrix.T, aspect="auto", cmap=cmap,
                       interpolation="nearest")
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([str(l) for l in layers], fontsize=7, rotation=45)
        ax.set_yticks(range(len(proj_types)))
        ax.set_yticklabels(proj_types, fontsize=9)
        ax.set_xlabel("Layer")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    path2 = os.path.join(plot_dir, "svd_projection_heatmap.png")
    plt.savefig(path2, dpi=150)
    plt.close()
    print(f"  Saved: {path2}")

    # --- Plot 3: Concentration analysis ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    # Scatter: x = dw_fro_norm, y = out_subspace_energy, color = layer
    for name, r in per_matrix_results.items():
        layer_idx = get_layer_idx(name)
        component = get_component(name)
        marker = "o" if component == "attn" else "s"
        ax.scatter(r["dw_fro_norm"], r["out_subspace_energy"],
                   c=layer_idx, cmap="viridis", marker=marker,
                   vmin=0, vmax=27, s=30, alpha=0.7)
    ax.axhline(y=0.1, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("‖ΔW‖_F (weight change magnitude)")
    ax.set_ylabel("Out-of-Subspace Energy (new direction fraction)")
    ax.set_title("Weight Change: Magnitude vs New Directions\n(○=attn, □=mlp, color=layer depth)")
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 27))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Layer Index")
    plt.tight_layout()
    path3 = os.path.join(plot_dir, "svd_magnitude_vs_direction.png")
    plt.savefig(path3, dpi=150)
    plt.close()
    print(f"  Saved: {path3}")


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(description="Phase 1b: SVD Weight Analysis")
    parser.add_argument("--rl_dir", default="/root/autodl-tmp/models/MEM1")
    parser.add_argument("--base_dir", default="/root/autodl-tmp/models/Qwen2.5-7B")
    parser.add_argument("--output_path", default="/root/autodl-tmp/learned_traj_compress/artifacts/exp_003_svd_analysis.json")
    parser.add_argument("--top_k", type=int, default=256,
                        help="Number of top singular vectors for subspace analysis")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--phase1a_results", default="/root/autodl-tmp/learned_traj_compress/artifacts/exp_002_collapse_curve_results.json",
                        help="Path to Phase 1a results for supplementary stats")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only analyze layers 0 and 27 (for quick verification)")
    parser.add_argument("--plot_dir", default="/root/autodl-tmp/learned_traj_compress/artifacts/plots",
                        help="Directory to save visualization plots")
    args = parser.parse_args()

    print(f"=== Phase 1b: SVD Weight Analysis ===")
    print(f"RL model: {args.rl_dir}")
    print(f"Base model: {args.base_dir}")
    print(f"top_k: {args.top_k}")
    if args.dry_run:
        print("*** DRY RUN: only layers 0 and 27 ***")
    print()

    t0 = time.time()

    dry_run_layers = {0, 27} if args.dry_run else None

    def should_include(name):
        if not param_wanted(name) or not name.endswith(".weight"):
            return False
        if dry_run_layers is not None:
            layer_idx = get_layer_idx(name)
            if layer_idx >= 0 and layer_idx not in dry_run_layers:
                return False
        return True

    # --- Load base model weights ---
    print("[1/5] Loading base model weights...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_dir, torch_dtype=torch.float16, low_cpu_mem_usage=True,
        trust_remote_code=True)
    base_weights = {}
    for n, p in base_model.named_parameters():
        if should_include(n) and p.ndim == 2:
            base_weights[n] = p.data.cpu().clone()
    n_matrices = len(base_weights)
    print(f"  Collected {n_matrices} weight matrices")
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    # --- Load RL model weights ---
    print("[2/5] Loading RL model weights...")
    rl_model = AutoModelForCausalLM.from_pretrained(
        args.rl_dir, torch_dtype=torch.float16, low_cpu_mem_usage=True,
        trust_remote_code=True)
    rl_weights = {}
    for n, p in rl_model.named_parameters():
        if should_include(n) and p.ndim == 2:
            rl_weights[n] = p.data.cpu().clone()
    print(f"  Collected {len(rl_weights)} weight matrices")
    del rl_model
    gc.collect()
    torch.cuda.empty_cache()

    # --- Analyze each weight matrix ---
    print(f"[3/5] Analyzing {n_matrices} weight matrices (top_k={args.top_k})...")
    per_layer_results = {}
    layer_summaries = defaultdict(lambda: defaultdict(list))

    for name in tqdm(sorted(base_weights.keys()), desc="SVD analysis"):
        if name not in rl_weights:
            print(f"  WARNING: {name} not in RL model, skipping")
            continue

        result = analyze_weight_pair(
            rl_weights[name], base_weights[name],
            top_k=args.top_k, device=args.device
        )

        per_layer_results[name] = result

        # Accumulate for summaries
        layer_idx = get_layer_idx(name)
        component = get_component(name)
        proj_type = get_proj_type(name)

        layer_summaries[layer_idx][f"{component}_{proj_type}_eff_rank_delta"] = result["eff_rank_delta"]
        layer_summaries[layer_idx][f"{component}_{proj_type}_U_angle_mean"] = result["U_angle_mean_deg"]

        # Print progress
        tqdm.write(
            f"  {name}: eff_rank Δ={result['eff_rank_delta']:+.1f} "
            f"({result['eff_rank_base']:.0f}→{result['eff_rank_rl']:.0f}), "
            f"U_angle={result['U_angle_mean_deg']:.1f}°, "
            f"ΔW_out={result['out_subspace_energy']:.1%}"
        )

        # Free GPU memory periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del base_weights, rl_weights
    gc.collect()

    # --- Aggregate summaries ---
    print("[4/5] Computing summaries...")

    # Global aggregates
    all_eff_rank_delta = [r["eff_rank_delta"] for r in per_layer_results.values()]
    all_eff_rank_ratio = [r["eff_rank_ratio"] for r in per_layer_results.values()]
    all_U_angle_mean = [r["U_angle_mean_deg"] for r in per_layer_results.values()]
    all_V_angle_mean = [r["V_angle_mean_deg"] for r in per_layer_results.values()]
    all_out_energy = [r["out_subspace_energy"] for r in per_layer_results.values()]
    all_in_energy = [r["in_subspace_energy"] for r in per_layer_results.values()]
    all_dw_fro = [r["dw_fro_norm"] for r in per_layer_results.values()]

    # Per-component aggregates
    attn_results = {n: r for n, r in per_layer_results.items() if get_component(n) == "attn"}
    mlp_results = {n: r for n, r in per_layer_results.items() if get_component(n) == "mlp"}

    def agg(results, key):
        vals = [r[key] for r in results.values()]
        if not vals:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    # Per-layer aggregates (for plotting layer depth trends)
    layer_depth_trend = {}
    for layer_idx in sorted(set(get_layer_idx(n) for n in per_layer_results)):
        layer_params = {n: r for n, r in per_layer_results.items() if get_layer_idx(n) == layer_idx}
        layer_attn = {n: r for n, r in layer_params.items() if get_component(n) == "attn"}
        layer_mlp = {n: r for n, r in layer_params.items() if get_component(n) == "mlp"}
        entry = {
            "eff_rank_delta_mean": float(np.mean([r["eff_rank_delta"] for r in layer_params.values()])),
            "U_angle_mean_deg": float(np.mean([r["U_angle_mean_deg"] for r in layer_params.values()])),
            "out_subspace_energy_mean": float(np.mean([r["out_subspace_energy"] for r in layer_params.values()])),
            "dw_fro_norm_mean": float(np.mean([r["dw_fro_norm"] for r in layer_params.values()])),
        }
        # Per-projection breakdown
        for n, r in layer_params.items():
            proj = get_proj_type(n)
            entry[f"{proj}_eff_rank_delta"] = r["eff_rank_delta"]
            entry[f"{proj}_U_angle_mean"] = r["U_angle_mean_deg"]
            entry[f"{proj}_out_energy"] = r["out_subspace_energy"]
            entry[f"{proj}_dw_fro"] = r["dw_fro_norm"]
        if layer_attn:
            entry["attn_eff_rank_delta_mean"] = float(np.mean([r["eff_rank_delta"] for r in layer_attn.values()]))
            entry["attn_U_angle_mean"] = float(np.mean([r["U_angle_mean_deg"] for r in layer_attn.values()]))
        if layer_mlp:
            entry["mlp_eff_rank_delta_mean"] = float(np.mean([r["eff_rank_delta"] for r in layer_mlp.values()]))
            entry["mlp_U_angle_mean"] = float(np.mean([r["U_angle_mean_deg"] for r in layer_mlp.values()]))
        layer_depth_trend[str(layer_idx)] = entry

    # --- Phase 1a supplementary statistics ---
    print("Computing Phase 1a supplementary statistics...")
    phase1a_supp = phase1a_supplementary(args.phase1a_results)

    # --- Compile output ---
    elapsed = time.time() - t0
    output = {
        "config": {
            "rl_dir": args.rl_dir,
            "base_dir": args.base_dir,
            "top_k": args.top_k,
            "n_matrices": n_matrices,
            "elapsed_s": round(elapsed, 1),
        },
        "summary": {
            "global": {
                "eff_rank_delta": agg(per_layer_results, "eff_rank_delta"),
                "eff_rank_ratio": agg(per_layer_results, "eff_rank_ratio"),
                "U_angle_mean_deg": agg(per_layer_results, "U_angle_mean_deg"),
                "V_angle_mean_deg": agg(per_layer_results, "V_angle_mean_deg"),
                "out_subspace_energy": agg(per_layer_results, "out_subspace_energy"),
                "in_subspace_energy": agg(per_layer_results, "in_subspace_energy"),
                "dw_fro_norm": agg(per_layer_results, "dw_fro_norm"),
                "align_U": agg(per_layer_results, "align_U"),
                "align_V": agg(per_layer_results, "align_V"),
            },
            "attention": {
                "eff_rank_delta": agg(attn_results, "eff_rank_delta"),
                "U_angle_mean_deg": agg(attn_results, "U_angle_mean_deg"),
                "out_subspace_energy": agg(attn_results, "out_subspace_energy"),
            },
            "mlp": {
                "eff_rank_delta": agg(mlp_results, "eff_rank_delta"),
                "U_angle_mean_deg": agg(mlp_results, "U_angle_mean_deg"),
                "out_subspace_energy": agg(mlp_results, "out_subspace_energy"),
            },
        },
        "layer_depth_trend": layer_depth_trend,
        "per_matrix": per_layer_results,
        "phase1a_supplementary": phase1a_supp,
    }

    # --- Go/no-go assessment ---
    mean_eff_rank_delta = float(np.mean(all_eff_rank_delta))
    mean_U_angle = float(np.mean(all_U_angle_mean))
    mean_out_energy = float(np.mean(all_out_energy))

    # Criteria:
    # (A) Mode expansion: mean eff_rank_delta > 0 (RL expands rank)
    # (B) Direction rotation: mean U_angle > 5° (non-trivial rotation)
    # (C) New directions: mean out_subspace_energy > 0.1 (>10% of ΔW outside base subspace)
    mode_expansion = mean_eff_rank_delta > 0
    direction_rotation = mean_U_angle > 5.0
    new_directions = mean_out_energy > 0.1

    output["go_no_go"] = {
        "mode_expansion": bool(mode_expansion),
        "mode_expansion_detail": f"mean eff_rank_delta = {mean_eff_rank_delta:+.1f}",
        "direction_rotation": bool(direction_rotation),
        "direction_rotation_detail": f"mean U_angle = {mean_U_angle:.1f}°",
        "new_directions": bool(new_directions),
        "new_directions_detail": f"mean out_subspace_energy = {mean_out_energy:.1%}",
        "overall": "GO" if (mode_expansion and direction_rotation) else "PARTIAL" if any([mode_expansion, direction_rotation, new_directions]) else "NO-GO",
    }

    # --- Save ---
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")
    print(f"Elapsed: {elapsed:.0f}s")

    # --- [5/5] Generate visualizations ---
    print("[5/5] Generating visualizations...")
    generate_plots(layer_depth_trend, per_layer_results, args.plot_dir)

    # --- Print summary ---
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nEffective Rank Delta (RL - base):")
    print(f"  Global: {mean_eff_rank_delta:+.1f} (std={float(np.std(all_eff_rank_delta)):.1f})")
    print(f"  Attention: {agg(attn_results, 'eff_rank_delta')['mean']:+.1f}")
    print(f"  MLP: {agg(mlp_results, 'eff_rank_delta')['mean']:+.1f}")
    print(f"\nSubspace Rotation (U angles, degrees):")
    print(f"  Global: {mean_U_angle:.1f}° (std={float(np.std(all_U_angle_mean)):.1f})")
    print(f"\nΔW Decomposition:")
    print(f"  In base subspace: {float(np.mean(all_in_energy)):.1%}")
    print(f"  Outside base subspace: {mean_out_energy:.1%}")
    print(f"\nGo/No-Go:")
    print(f"  mode_expansion: {mode_expansion}")
    print(f"  direction_rotation: {direction_rotation}")
    print(f"  new_directions: {new_directions}")
    print(f"  overall: {output['go_no_go']['overall']}")

    if phase1a_supp.get("holm_bonferroni"):
        print(f"\nPhase 1a Holm-Bonferroni correction:")
        for item in phase1a_supp["holm_bonferroni"]:
            sig = "✅" if item["significant_005"] else "❌"
            print(f"  {sig} {item['test']}: raw_p={item['raw_p']:.4f} → adj_p={item['adjusted_p']:.4f}")

    if phase1a_supp.get("cohens_d"):
        print(f"\nPhase 1a Cohen's d effect sizes:")
        for test, stats in phase1a_supp["cohens_d"].items():
            print(f"  {test}: d={stats['cohens_d']:.3f} ({stats['interpretation']})")


if __name__ == "__main__":
    main()
