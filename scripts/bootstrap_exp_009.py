"""Paired bootstrap significance test: SFT+API (exp_009) vs SFT-only (exp_006)."""
import json
import numpy as np
import sys
from pathlib import Path

BASE_DIR = Path("/root/autodl-tmp/learned_traj_compress")
API_FILE = BASE_DIR / "artifacts/exp_009_eval/eval_sft_shared_api_oracle.json"
BASELINE_FILE = BASE_DIR / "artifacts/exp_006_eval/eval_sft_shared.json"
OUT_FILE = BASE_DIR / "artifacts/exp_009_eval/bootstrap_results.json"

N_BOOTSTRAP = 10000
SEED = 42


def load_per_prompt_em(data, n_key):
    """Extract per-prompt em_partial list and gold_answers fingerprint."""
    details = data["details"][n_key]
    em_scores = [d["em_partial"] for d in details]
    fingerprints = [str(d["gold_answers"]) for d in details]
    return em_scores, fingerprints


def paired_bootstrap(scores_a, scores_b, n_boot=N_BOOTSTRAP, seed=SEED):
    """Paired bootstrap test: H0: mean(a) - mean(b) = 0."""
    rng = np.random.RandomState(seed)
    a = np.array(scores_a)
    b = np.array(scores_b)
    n = len(a)
    assert len(b) == n

    observed_delta = a.mean() - b.mean()

    # Bootstrap distribution of delta
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        deltas[i] = a[idx].mean() - b[idx].mean()

    # Two-sided p-value: fraction of bootstrap deltas <= 0 (since we expect delta > 0)
    p_value = (deltas <= 0).mean()

    ci_lo = np.percentile(deltas, 2.5)
    ci_hi = np.percentile(deltas, 97.5)

    return {
        "observed_delta": round(float(observed_delta), 4),
        "p_value": float(p_value),
        "ci_95_lo": round(float(ci_lo), 4),
        "ci_95_hi": round(float(ci_hi), 4),
        "n_prompts": n,
        "mean_api": round(float(a.mean()), 4),
        "mean_baseline": round(float(b.mean()), 4),
        "n_bootstrap": n_boot,
    }


def main():
    with open(API_FILE) as f:
        api_data = json.load(f)
    with open(BASELINE_FILE) as f:
        baseline_data = json.load(f)

    results = {}
    for n_key in ["N2", "N4", "N8"]:
        api_em, api_fp = load_per_prompt_em(api_data, n_key)
        base_em, base_fp = load_per_prompt_em(baseline_data, n_key)

        # Verify alignment via gold_answers
        n_match = sum(a == b for a, b in zip(api_fp, base_fp))
        n_total = min(len(api_fp), len(base_fp))
        if n_match < n_total:
            print(f"WARNING {n_key}: only {n_match}/{n_total} gold_answers match. Using intersection by gold_answers.")
            # Fall back to matching by gold_answers
            base_map = {}
            for i, fp in enumerate(base_fp):
                base_map.setdefault(fp, i)
            paired_api, paired_base = [], []
            for i, fp in enumerate(api_fp):
                if fp in base_map:
                    paired_api.append(api_em[i])
                    paired_base.append(base_em[base_map[fp]])
            api_em, base_em = paired_api, paired_base
            print(f"  Paired {len(api_em)} prompts")
        else:
            print(f"{n_key}: all {n_total} prompts aligned ✓")

        res = paired_bootstrap(api_em, base_em)
        results[n_key] = res
        print(f"{n_key}: delta={res['observed_delta']:.4f}  "
              f"p={res['p_value']:.4f}  "
              f"95% CI=[{res['ci_95_lo']:.4f}, {res['ci_95_hi']:.4f}]  "
              f"(API={res['mean_api']:.4f} vs base={res['mean_baseline']:.4f}, n={res['n_prompts']})")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump({"test": "paired_bootstrap", "comparison": "sft_api_oracle vs sft_baseline",
                    "n_bootstrap": N_BOOTSTRAP, "seed": SEED, "results": results}, f, indent=2)
    print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    main()
