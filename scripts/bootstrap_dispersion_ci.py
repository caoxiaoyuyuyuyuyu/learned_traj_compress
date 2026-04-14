#!/usr/bin/env python3
"""Bootstrap 95% CIs for per-sample surface & task metrics, cross-backbone.

Inputs:
  - artifacts/w1_surface_metrics/results.json        (Qwen-3B per-sample action_seq, rouge_l, N=8 no-API)
  - artifacts/exp_006_eval/eval_sft_shared.json      (Qwen-3B per-sample em_full/em_partial, N={2,4,8})
  - artifacts/exp_013/eval_no_api/eval_exp_013_sft_llama1b.json
  - artifacts/exp_013/eval_oracle/eval_sft_shared_api_oracle.json
  - artifacts/exp_013/eval_bm25/eval_sft_shared_api_bm25.json
  - artifacts/phase1d_v2_data/raw_trajectories_N8.json  (teacher for Llama-1B surface recompute)

Outputs:
  - artifacts/_project/paper/bootstrap_ci_table.md
  - artifacts/_project/paper/bootstrap_ci_table.tex
  - artifacts/_project/paper/bootstrap_ci_log.txt
"""
import json, os, re, sys
import numpy as np

N_BOOT = 1000
RNG = np.random.default_rng(42)

ACTION_TAG_RE = re.compile(r"<(think|search|information|answer)\b")
INFO_BLOCK_RE = re.compile(r"\n*<information>.*?</information>\n*", re.DOTALL)


def strip_info(text):
    return INFO_BLOCK_RE.sub("\n", text).strip()


def extract_actions(text):
    return ACTION_TAG_RE.findall(text)


def action_seq_acc(s, t):
    if not t:
        return 1.0 if not s else 0.0
    m = min(len(s), len(t))
    if m == 0:
        return 0.0
    return sum(a == b for a, b in zip(s[:m], t[:m])) / len(t)


def boot_ci(arr, n_boot=N_BOOT):
    arr = np.asarray(arr, dtype=float)
    if len(arr) == 0:
        return (float("nan"),) * 3
    idx = RNG.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compute_llama_per_sample_surface():
    """Recompute per-sample action_seq_acc and ROUGE-L for Llama-1B at N=8 no-API."""
    from rouge_score import rouge_scorer
    eval_path = "artifacts/exp_013/eval_no_api/eval_exp_013_sft_llama1b.json"
    teacher_path = "artifacts/phase1d_v2_data/raw_trajectories_N8.json"
    d = json.load(open(eval_path))
    td = json.load(open(teacher_path))
    teacher_lookup = {}
    for r in td["results"]:
        best = max(r["responses"], key=lambda x: x["total_em"])
        teacher_lookup[r["prompt_idx"]] = best["full_assistant"]
    items = d["details"]["N8"]
    action_accs = []
    rouge_ls = []
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    for it in items:
        pidx = it["prompt_idx"]
        if pidx not in teacher_lookup:
            continue
        stu = it["generated_text"]
        tea = teacher_lookup[pidx]
        s_seq = extract_actions(stu)
        t_seq = extract_actions(tea)
        action_accs.append(action_seq_acc(s_seq, t_seq))
        stu_m = strip_info(stu)
        tea_m = strip_info(tea)
        r = scorer.score(tea_m, stu_m)["rougeL"].fmeasure
        rouge_ls.append(r)
    return action_accs, rouge_ls


def per_sample_from_eval(path, metric, n_key):
    d = json.load(open(path))
    items = d["details"][n_key]
    return [float(it[metric]) for it in items]


def main():
    rows = []  # (backbone, mode, metric, N_bucket, n, mean, lo, hi)

    # --- Qwen-3B N=8 surface (from existing w1_surface_metrics) ---
    w1 = json.load(open("artifacts/w1_surface_metrics/results.json"))
    qwen_action = w1["diagnostics"]["action_seq_accs"]
    qwen_rouge = w1["diagnostics"]["rouge_l_scores"]
    m, lo, hi = boot_ci(qwen_action)
    rows.append(("Qwen-3B", "no-API", "action_seq_acc", "N=8", len(qwen_action), m, lo, hi))
    m, lo, hi = boot_ci(qwen_rouge)
    rows.append(("Qwen-3B", "no-API", "ROUGE-L", "N=8", len(qwen_rouge), m, lo, hi))

    # --- Llama-1B N=8 surface (recomputed) ---
    llama_action, llama_rouge = compute_llama_per_sample_surface()
    m, lo, hi = boot_ci(llama_action)
    rows.append(("Llama-1B", "no-API", "action_seq_acc", "N=8", len(llama_action), m, lo, hi))
    m, lo, hi = boot_ci(llama_rouge)
    rows.append(("Llama-1B", "no-API", "ROUGE-L", "N=8", len(llama_rouge), m, lo, hi))

    # --- em_partial per-sample from eval JSONs (cross-backbone × mode × N) ---
    qwen_no_api = "artifacts/exp_006_eval/eval_sft_shared.json"
    llama_paths = {
        "no-API": "artifacts/exp_013/eval_no_api/eval_exp_013_sft_llama1b.json",
        "oracle": "artifacts/exp_013/eval_oracle/eval_sft_shared_api_oracle.json",
        "BM25": "artifacts/exp_013/eval_bm25/eval_sft_shared_api_bm25.json",
    }
    for n_key in ("N2", "N4", "N8"):
        vals = per_sample_from_eval(qwen_no_api, "em_partial", n_key)
        m, lo, hi = boot_ci(vals)
        rows.append(("Qwen-3B", "no-API", "em_partial", n_key, len(vals), m, lo, hi))
    for mode, path in llama_paths.items():
        for n_key in ("N2", "N4", "N8"):
            vals = per_sample_from_eval(path, "em_partial", n_key)
            m, lo, hi = boot_ci(vals)
            rows.append(("Llama-1B", mode, "em_partial", n_key, len(vals), m, lo, hi))

    # markdown
    md = [
        "| Backbone | Retrieval | Metric | Split | N | Mean | 95% CI |",
        "|----------|-----------|--------|-------|---|------|--------|",
    ]
    for bb, mo, me, nk, n, m, lo, hi in rows:
        md.append(f"| {bb} | {mo} | {me} | {nk} | {n} | {m:.4f} | [{lo:.4f}, {hi:.4f}] |")
    os.makedirs("artifacts/_project/paper", exist_ok=True)
    open("artifacts/_project/paper/bootstrap_ci_table.md", "w").write("\n".join(md) + "\n")

    # LaTeX (booktabs)
    tex = [
        r"\begin{tabular}{llllrrr}",
        r"\toprule",
        r"Backbone & Retrieval & Metric & Split & $N$ & Mean & 95\% CI \\",
        r"\midrule",
    ]
    for bb, mo, me, nk, n, m, lo, hi in rows:
        me_tex = me.replace("_", r"\_")
        tex.append(f"{bb} & {mo} & {me_tex} & {nk} & {n} & {m:.3f} & [{lo:.3f}, {hi:.3f}] \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    open("artifacts/_project/paper/bootstrap_ci_table.tex", "w").write("\n".join(tex) + "\n")

    # log
    log = [f"{r}" for r in rows]
    open("artifacts/_project/paper/bootstrap_ci_log.txt", "w").write("\n".join(log) + "\n")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
