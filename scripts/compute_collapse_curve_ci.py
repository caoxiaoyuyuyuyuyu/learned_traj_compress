"""Compute 95% bootstrap CI for Figure 1 (collapse curve) data points."""
import json
import numpy as np
import os

SEED = 42
N_BOOTSTRAP = 10000
CI_LEVEL = 0.95

BASE_DIR = "/root/autodl-tmp/learned_traj_compress"
EXP002 = f"{BASE_DIR}/artifacts/exp_002_collapse_curve_results.json"
SFT_EVAL = f"{BASE_DIR}/artifacts/exp_006_eval/eval_sft_shared.json"
DPO_N8_EVAL = f"{BASE_DIR}/artifacts/exp_006_eval/eval_merged_N8.json"
OUT_PATH = f"{BASE_DIR}/artifacts/_project/paper/figures/collapse_curve_ci.json"


def bootstrap_ci(values, n_bootstrap=N_BOOTSTRAP, ci_level=CI_LEVEL, seed=SEED):
    """Compute bootstrap mean and CI."""
    rng = np.random.RandomState(seed)
    arr = np.array(values, dtype=float)
    n = len(arr)
    boot_means = np.array([
        rng.choice(arr, size=n, replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    alpha = (1 - ci_level) / 2
    ci_lower = float(np.percentile(boot_means, 100 * alpha))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha)))
    return {
        "mean": float(arr.mean()),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_samples": n,
    }


def load_exp002():
    """Load per-sample em_mean from exp_002 (RL teacher + base)."""
    with open(EXP002) as f:
        d = json.load(f)
    psd = d["per_sample_details"]
    results = {}
    for model in ["mem1", "base"]:
        results[model] = {}
        for n_str, samples in psd[model].items():
            values = [s["em_mean"] for s in samples]
            results[model][int(n_str)] = bootstrap_ci(values)
    return results


def load_eval_file(path, model_label):
    """Load per-prompt em_partial from eval JSON files."""
    with open(path) as f:
        d = json.load(f)
    results = {}
    for n_key, prompts in d["details"].items():
        n = int(n_key.replace("N", ""))
        values = [p["em_partial"] for p in prompts]
        results[n] = bootstrap_ci(values)
    return results


def main():
    exp002 = load_exp002()

    sft = load_eval_file(SFT_EVAL, "sft_student")
    dpo_n8 = load_eval_file(DPO_N8_EVAL, "dpo_n8_student")

    output = {
        "description": "95% bootstrap CI (10000 resamples) for Figure 1 collapse curve",
        "metric": "em_partial (per-prompt mean EM over objectives)",
        "data": {
            "rl_teacher_mem1": {
                str(n): v for n, v in sorted(exp002["mem1"].items())
            },
            "base_model": {
                str(n): v for n, v in sorted(exp002["base"].items())
            },
            "sft_student": {
                str(n): v for n, v in sorted(sft.items())
            },
            "dpo_n8_student": {
                str(n): v for n, v in sorted(dpo_n8.items())
            },
        },
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
