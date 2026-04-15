import json, random, re, os
WD = "/root/autodl-tmp/learned_traj_compress/artifacts/exp_016_workdir"

def load(name):
    with open(os.path.join(WD, name)) as f:
        return json.load(f)

def per_sample(d):
    return [x["em_partial"] for x in d["details"]]

of = load("oracle_full_eval.json")
orf = load("oracle_redacted_full_eval.json")
orh = load("oracle_redacted_half_eval.json")
na = load("no_api_eval.json")

a = per_sample(of); b = per_sample(orf)
assert len(a) == 125 and len(b) == 125
diffs = [a[i]-b[i] for i in range(125)]
mean = sum(diffs)/125

rng = random.Random(42)
N = 10000
resamples = []
gt0 = 0
for _ in range(N):
    idx = [rng.randrange(125) for _ in range(125)]
    m = sum(diffs[i] for i in idx)/125
    resamples.append(m)
    if m > 0: gt0 += 1
resamples.sort()
lo = resamples[int(0.025*N)]
hi = resamples[int(0.975*N)-1]
p_boot_gt0 = gt0/N

# Update bootstrap JSON — append only
bj = load("bootstrap_redaction_ci.json")
bj["oracle_full_vs_oracle_redacted_full_delta"] = {
    "delta_em_partial": round(mean, 4),
    "ci95_lo": round(lo, 4),
    "ci95_hi": round(hi, 4),
    "p_boot_gt0": p_boot_gt0,
    "n_resamples": N,
    "seed": 42,
    "interpretation": "leakage: oracle_full − oracle_redacted_full (paired)"
}
with open(os.path.join(WD, "bootstrap_redaction_ci.json"), "w") as f:
    json.dump(bj, f, indent=2)

# (b) half-subset redaction stats
RED = "[REDACTED]"
total_blocks = 0
total_subs = 0
redacted_pass_idx = []
for i, det in enumerate(orh["details"]):
    # concatenate all stringish fields in detail
    s = json.dumps(det)
    c = s.count(RED)
    if c > 0:
        redacted_pass_idx.append(i)
        total_subs += c
        # Count blocks via prompt/context field if present. Fallback: 1 block/passage
        total_blocks += 1  # we'll refine below if possible

# Better: scan original redaction_stats.json
rs = load("redaction_stats.json")
stats_half = {
    "source": "oracle_redacted_half_eval.json details (post-hoc regex scan)",
    "n_prompts": len(orh["details"]),
    "n_passages_with_redactions": len(redacted_pass_idx),
    "first_N_redacted": max(redacted_pass_idx)+1 if redacted_pass_idx else 0,
    "total_redacted_substitutions": total_subs,
    "mean_substitutions_per_redacted_passage": round(total_subs/max(1,len(redacted_pass_idx)), 3),
    "mean_substitutions_per_all_passages": round(total_subs/len(orh["details"]), 3),
    "redacted_passage_indices_sample_first20": redacted_pass_idx[:20],
    "redacted_passage_indices_sample_last20": redacted_pass_idx[-20:] if len(redacted_pass_idx)>20 else [],
    "methodology_note": "half-subset redaction applied to first N of 125 passages; exact stats re-computed post-hoc from eval details via regex scan of [REDACTED] tokens. Original redaction_stats.json covers full-redacted condition only.",
    "cross_ref_original_redaction_stats": {k: rs[k] for k in rs if k not in ("samples",)}
}
with open(os.path.join(WD, "redaction_stats_half.json"), "w") as f:
    json.dump(stats_half, f, indent=2)

# (d) no_api latency probe — avg generated tokens
def avg_tokens(d, field_candidates=("generated_text","response","pred_text","predicted_answer","raw_output")):
    if not d.get("details"): return None, None
    sample = d["details"][0]
    field = None
    for k in field_candidates:
        if k in sample:
            field = k; break
    if field is None:
        # dump keys for inspect
        return None, list(sample.keys())
    lens = []
    for det in d["details"]:
        t = det.get(field, "")
        if isinstance(t, str):
            lens.append(len(t.split()))
    return (sum(lens)/len(lens) if lens else None), field

na_avg, na_field = avg_tokens(na)
of_avg, of_field = avg_tokens(of)
probe = {
    "no_api_field": na_field, "no_api_avg_word_tokens": na_avg,
    "oracle_full_field": of_field, "oracle_full_avg_word_tokens": of_avg,
    "no_api_elapsed_per_item": na.get("elapsed_seconds",0)/max(1,na.get("n_prompts",1)),
    "oracle_full_elapsed_per_item": of.get("elapsed_seconds",0)/max(1,of.get("n_prompts",1)),
    "avg_searches_no_api": na.get("avg_searches"),
    "truncated_rate_no_api": na.get("truncated_rate"),
    "no_api_samples_first3": [na["details"][i].get(na_field,"")[:400] if na_field else str(list(na["details"][i].keys())) for i in range(min(3, len(na.get("details",[]))))]
}
with open(os.path.join(WD, "no_api_latency_probe.json"), "w") as f:
    json.dump(probe, f, indent=2)

out = {
    "delta_leakage": bj["oracle_full_vs_oracle_redacted_full_delta"],
    "half_stats": stats_half,
    "latency_probe_summary": {k: v for k, v in probe.items() if k != "no_api_samples_first3"}
}
print(json.dumps(out, indent=2, ensure_ascii=False))
