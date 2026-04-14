#!/usr/bin/env python3
"""Parse exp_007 MLP probe log → JSON artifact.

Log format (per feature line):
    <feature_name> (dim=<d>)... MLP ρ=<x>, Linear ρ=<y>
or (control-only):
    random_feature_mlp (control)... ρ=<x>

Layer sections delimited by '--- Layer <L> ---', k sections by '  k=<K>'.
"""
import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean, stdev

try:
    from scipy import stats as sstats
except ImportError:
    sstats = None

LAYER_RE = re.compile(r"^---\s*Layer\s+(\d+)\s*---")
K_RE = re.compile(r"^\s*k=(\d+)\s*$")
FEAT_RE = re.compile(
    r"^\s*(\S+)\s*\(dim=(\d+)\)\.\.\.\s*MLP\s*ρ=(-?\d+\.\d+),\s*Linear\s*ρ=(-?\d+\.\d+)"
)
CTRL_RE = re.compile(r"^\s*random_feature_mlp\s*\(control\)\.\.\.\s*ρ=(-?\d+\.\d+)")


def parse_log(log_path: Path):
    layer = None
    k = None
    results = []
    with log_path.open("r", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            m = LAYER_RE.search(line)
            if m:
                layer = int(m.group(1))
                continue
            m = K_RE.match(line)
            if m:
                k = int(m.group(1))
                continue
            if layer is None or k is None:
                continue
            m = FEAT_RE.match(line)
            if m:
                feat, dim, mlp_rho, lin_rho = m.groups()
                results.append(
                    {
                        "layer": layer,
                        "k": k,
                        "feature": feat,
                        "dim": int(dim),
                        "mlp_rho": float(mlp_rho),
                        "linear_rho": float(lin_rho),
                    }
                )
                continue
            m = CTRL_RE.match(line)
            if m:
                results.append(
                    {
                        "layer": layer,
                        "k": k,
                        "feature": "random_feature_mlp_control",
                        "dim": None,
                        "mlp_rho": float(m.group(1)),
                        "linear_rho": None,
                    }
                )
    return results


def summarize(results):
    real_features = {
        "h_rl_full",
        "delta_h_full",
        "delta_h_principal_k64",
        "delta_h_principal_k256",
        "h_rl_principal_k64",
        "h_rl_principal_k256",
        "h_base_principal_k64",
        "h_base_principal_k256",
    }
    real_pairs = [
        r for r in results if r["feature"] in real_features and r["linear_rho"] is not None
    ]
    random_pairs = [
        r for r in results if r["feature"].startswith("delta_h_random_")
    ]
    ctrl = [r for r in results if r["feature"] == "random_feature_mlp_control"]

    mlp_vals = [r["mlp_rho"] for r in real_pairs]
    lin_vals = [r["linear_rho"] for r in real_pairs]
    n_pairs = len(real_pairs)

    summary = {
        "n_real_pairs": n_pairs,
        "n_random_pairs": len(random_pairs),
        "n_control": len(ctrl),
    }

    if sstats and n_pairs >= 2:
        t, p = sstats.ttest_rel(mlp_vals, lin_vals)
        diffs = [a - b for a, b in zip(mlp_vals, lin_vals)]
        d_std = stdev(diffs) if len(diffs) > 1 else 0.0
        cohens_d = mean(diffs) / d_std if d_std > 0 else 0.0
        summary["mlp_vs_linear_paired_t"] = {
            "t": float(t),
            "p": float(p),
            "n_pairs": n_pairs,
            "mlp_mean_rho": mean(mlp_vals),
            "linear_mean_rho": mean(lin_vals),
            "mean_diff_mlp_minus_linear": mean(diffs),
            "cohens_d": cohens_d,
            "mlp_better_count": sum(1 for a, b in zip(mlp_vals, lin_vals) if a > b),
            "linear_better_count": sum(1 for a, b in zip(mlp_vals, lin_vals) if b > a),
        }
        summary["mlp_lost_to_linear_pct"] = (
            summary["mlp_vs_linear_paired_t"]["linear_better_count"] / n_pairs
        )

    if random_pairs:
        rp_lin = [r["linear_rho"] for r in random_pairs if r["linear_rho"] is not None]
        rp_mlp = [r["mlp_rho"] for r in random_pairs]
        summary["random_baseline_linear_rho"] = {
            "min": min(rp_lin),
            "max": max(rp_lin),
            "abs_max": max(abs(v) for v in rp_lin),
            "mean": mean(rp_lin),
            "std": stdev(rp_lin) if len(rp_lin) > 1 else 0.0,
        }
        summary["random_baseline_mlp_rho"] = {
            "min": min(rp_mlp),
            "max": max(rp_mlp),
            "abs_max": max(abs(v) for v in rp_mlp),
            "mean": mean(rp_mlp),
            "std": stdev(rp_mlp) if len(rp_mlp) > 1 else 0.0,
        }
        # noise band = max |ρ| seen on random features, Linear side
        noise_band = summary["random_baseline_linear_rho"]["abs_max"]
        summary["noise_band_linear_abs"] = noise_band
        above = sum(
            1 for r in real_pairs if abs(r["linear_rho"]) > noise_band
        )
        summary["real_linear_above_noise_band_count"] = above
        summary["real_linear_above_noise_band_pct"] = (
            above / n_pairs if n_pairs else 0.0
        )

    layers_seen = sorted({r["layer"] for r in results})
    ks_seen = sorted({r["k"] for r in results})
    summary["layers_covered"] = layers_seen
    summary["ks_covered"] = ks_seen
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--n_samples", type=int, default=300)
    ap.add_argument("--hidden_dim", type=int, default=3584)
    args = ap.parse_args()

    results = parse_log(args.log)
    if not results:
        print("ERROR: no results parsed from log", file=sys.stderr)
        sys.exit(1)
    summary = summarize(results)

    out = {
        "exp_id": "exp_007",
        "source_log": str(args.log),
        "model": {
            "hidden_dim": args.hidden_dim,
            "layers_sampled_planned": [0, 3, 7, 11, 15, 19, 23, 27],
            "layers_covered": summary["layers_covered"],
            "ks": summary["ks_covered"],
        },
        "n_samples": args.n_samples,
        "note": (
            "Log emits single ρ per feature (mean over 5 seeds × 5-fold already "
            "collapsed inside script); std not in log so only point estimates here. "
            "Process was SIGTERM'd mid Layer 19 k=64 final control line; partial "
            "rows preserved."
        ),
        "results": results,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.output} ({len(results)} rows, {len(summary)} summary keys)")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
