#!/usr/bin/env python3
"""
SVD Alignment Ratio Statistical Analysis for C2 Revision.

Reads exp_004_svd_three_way.json and produces:
  (a) Descriptive statistics (mean ± std) by attention/MLP
  (b) Paired Wilcoxon signed-rank test + paired bootstrap
  (c) Outlier detection (layers where r_SFT > r_RL)
  (d) Per-layer boxplot (PDF)
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

# ── Config ──────────────────────────────────────────────────────────────
DATA_PATH = "artifacts/exp_004_svd_three_way.json"
FIG_DIR = "artifacts/_project/paper/figures"
FIG_PATH = os.path.join(FIG_DIR, "svd_alignment_boxplot.pdf")
K_VALUES = [64, 128, 256, 512]


def load_data(path):
    with open(path) as f:
        data = json.load(f)
    per_matrix = data["per_matrix"]

    records = []
    for name, info in per_matrix.items():
        layer = info["layer"]
        component = info["component"]  # "attn" or "mlp"
        proj_type = info["proj_type"]

        for k_str, k_data in info["k_analysis"].items():
            k = int(k_str)
            r_rl = k_data["rl_vs_base"]["obs_over_random"]
            r_sft = k_data["sft_vs_base"]["obs_over_random"]
            r_rl_sft = k_data["rl_vs_sft"]["obs_over_random"]
            records.append({
                "name": name,
                "layer": layer,
                "component": component,
                "proj_type": proj_type,
                "k": k,
                "r_rl": r_rl,
                "r_sft": r_sft,
                "r_rl_sft": r_rl_sft,
            })
    return records


def descriptive_stats(records):
    """Mean ± std of r_RL and r_SFT, by k and component."""
    print("=" * 80)
    print("(a) Descriptive Statistics: obs_over_random (alignment ratio)")
    print("=" * 80)

    for k in K_VALUES:
        recs_k = [r for r in records if r["k"] == k]
        for comp in ["attn", "mlp", "all"]:
            if comp == "all":
                subset = recs_k
            else:
                subset = [r for r in recs_k if r["component"] == comp]
            if not subset:
                continue

            rl_vals = np.array([r["r_rl"] for r in subset])
            sft_vals = np.array([r["r_sft"] for r in subset])

            print(f"\n  k={k:>4d}  |  {comp:>4s}  (n={len(subset)})")
            print(f"    r_RL  = {rl_vals.mean():.4f} ± {rl_vals.std():.4f}  "
                  f"[median={np.median(rl_vals):.4f}]")
            print(f"    r_SFT = {sft_vals.mean():.4f} ± {sft_vals.std():.4f}  "
                  f"[median={np.median(sft_vals):.4f}]")
            print(f"    ratio r_RL/r_SFT = {rl_vals.mean() / sft_vals.mean():.2f}x")


def paired_tests(records):
    """Wilcoxon signed-rank + paired bootstrap for r_RL vs r_SFT."""
    print("\n" + "=" * 80)
    print("(b) Paired Statistical Tests: r_RL vs r_SFT")
    print("=" * 80)

    for k in K_VALUES:
        recs_k = [r for r in records if r["k"] == k]
        print(f"\n── k = {k} ──")

        for comp in ["attn", "mlp", "all"]:
            if comp == "all":
                subset = recs_k
            else:
                subset = [r for r in recs_k if r["component"] == comp]
            if not subset:
                continue

            rl_vals = np.array([r["r_rl"] for r in subset])
            sft_vals = np.array([r["r_sft"] for r in subset])
            diffs = rl_vals - sft_vals

            # Wilcoxon signed-rank (two-sided)
            stat_w, p_wilcoxon = stats.wilcoxon(rl_vals, sft_vals, alternative="two-sided")

            # Paired bootstrap (10000 resamples)
            rng = np.random.default_rng(42)
            n = len(diffs)
            boot_means = np.array([
                rng.choice(diffs, size=n, replace=True).mean()
                for _ in range(10000)
            ])
            ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
            p_boot = np.mean(boot_means <= 0) * 2  # two-sided
            p_boot = min(p_boot, 2 - p_boot)

            frac_rl_gt = np.mean(diffs > 0)

            print(f"  {comp:>4s} (n={len(subset):>3d}): "
                  f"Wilcoxon p={p_wilcoxon:.2e}, "
                  f"Bootstrap 95% CI=[{ci_lo:.4f}, {ci_hi:.4f}], p={p_boot:.4f}, "
                  f"r_RL > r_SFT in {frac_rl_gt*100:.1f}% of layers")


def outlier_detection(records):
    """Find layers where r_SFT > r_RL."""
    print("\n" + "=" * 80)
    print("(c) Outliers: layers where r_SFT > r_RL")
    print("=" * 80)

    for k in K_VALUES:
        recs_k = [r for r in records if r["k"] == k]
        outliers = [r for r in recs_k if r["r_sft"] > r["r_rl"]]
        print(f"\n  k={k}: {len(outliers)}/{len(recs_k)} layers have r_SFT > r_RL")
        if outliers:
            # Sort by difference
            outliers.sort(key=lambda r: r["r_sft"] - r["r_rl"], reverse=True)
            for r in outliers[:10]:  # show top 10
                print(f"    L{r['layer']:02d}.{r['proj_type']:>20s}  "
                      f"r_RL={r['r_rl']:.4f}  r_SFT={r['r_sft']:.4f}  "
                      f"Δ={r['r_sft']-r['r_rl']:.4f}")


def make_boxplot(records):
    """Per-layer boxplot of alignment ratio, RL vs SFT, split by attn/MLP."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)

    # Use k=64 as the representative (strongest signal)
    k_plot = 64
    recs_k = [r for r in records if r["k"] == k_plot]

    # Get all layers sorted
    layers = sorted(set(r["layer"] for r in recs_k))
    n_layers = len(layers)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for ax, comp, title in zip(axes, ["attn", "mlp"],
                                ["Attention Layers", "MLP Layers"]):
        # Collect per-layer data
        rl_by_layer = defaultdict(list)
        sft_by_layer = defaultdict(list)
        for r in recs_k:
            if r["component"] == comp:
                rl_by_layer[r["layer"]].append(r["r_rl"])
                sft_by_layer[r["layer"]].append(r["r_sft"])

        positions_rl = []
        positions_sft = []
        data_rl = []
        data_sft = []

        for i, layer in enumerate(layers):
            if layer in rl_by_layer:
                data_rl.append(rl_by_layer[layer])
                positions_rl.append(i - 0.15)
                data_sft.append(sft_by_layer[layer])
                positions_sft.append(i + 0.15)

        if not data_rl:
            continue

        bp_rl = ax.boxplot(data_rl, positions=positions_rl, widths=0.25,
                           patch_artist=True, showfliers=True,
                           flierprops=dict(markersize=3))
        bp_sft = ax.boxplot(data_sft, positions=positions_sft, widths=0.25,
                            patch_artist=True, showfliers=True,
                            flierprops=dict(markersize=3))

        for patch in bp_rl["boxes"]:
            patch.set_facecolor("#4C72B0")
            patch.set_alpha(0.7)
        for patch in bp_sft["boxes"]:
            patch.set_facecolor("#DD8452")
            patch.set_alpha(0.7)

        ax.set_title(f"{title} — Alignment Ratio r(k={k_plot})", fontsize=12)
        ax.set_ylabel("obs / random", fontsize=10)
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, label="random baseline")
        ax.legend([bp_rl["boxes"][0], bp_sft["boxes"][0]], ["RL", "SFT"],
                  loc="upper right", fontsize=9)
        ax.set_xticks(range(n_layers))
        ax.set_xticklabels([str(l) for l in layers], fontsize=7)

    axes[-1].set_xlabel("Layer Index", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    print(f"\n  Boxplot saved to {FIG_PATH}")
    plt.close()


def main():
    records = load_data(DATA_PATH)
    print(f"Loaded {len(records)} records "
          f"({len(records)//len(K_VALUES)} matrices × {len(K_VALUES)} k-values)\n")

    descriptive_stats(records)
    paired_tests(records)
    outlier_detection(records)

    print("\n" + "=" * 80)
    print("(d) Generating boxplot...")
    print("=" * 80)
    make_boxplot(records)


if __name__ == "__main__":
    main()
