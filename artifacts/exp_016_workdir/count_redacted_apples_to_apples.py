import json, os
WD = "/root/autodl-tmp/learned_traj_compress/artifacts/exp_016_workdir"
RED = "[REDACTED]"

def load(n):
    with open(os.path.join(WD, n)) as f:
        return json.load(f)

def count_eval(name):
    d = load(name)
    details = d["details"]
    total = 0
    n_items_with = 0
    per_item = []
    for det in details:
        c = json.dumps(det).count(RED)
        per_item.append(c)
        total += c
        if c > 0:
            n_items_with += 1
    return {
        "total_redacted": total,
        "n_items_with_redacted": n_items_with,
        "n_items_total": len(details),
        "per_item_counts": per_item,
    }

full = count_eval("oracle_redacted_full_eval.json")
half = count_eval("oracle_redacted_half_eval.json")

D_BASELINE = 325
sanity_pass = abs(half["total_redacted"] - D_BASELINE) <= 5

ratio = (half["total_redacted"] / full["total_redacted"]) if full["total_redacted"] else None

# interpretation
def classify(r):
    if r is None: return "undefined"
    if abs(r - 0.5) <= 0.08: return "first-N_subset"
    if abs(r - 0.2) <= 0.08: return "density-half"
    return "other"

out = {
    "counting_method": "json.dumps(details[i]).count('[REDACTED]') — identical to D worker's postprocess.py",
    "field_path_used": "json.dumps of entire details[i] dict (same as D)",
    "full_eval_total_redacted": full["total_redacted"],
    "full_eval_n_items_with_redacted": full["n_items_with_redacted"],
    "full_eval_n_items_total": full["n_items_total"],
    "half_eval_total_redacted": half["total_redacted"],
    "half_eval_n_items_with_redacted": half["n_items_with_redacted"],
    "half_eval_n_items_total": half["n_items_total"],
    "half_eval_sanity_check_matches_D": sanity_pass,
    "D_baseline_for_half": D_BASELINE,
    "ratio_half_over_full": round(ratio, 4) if ratio is not None else None,
    "phase0_dry_run_full_count_for_reference": 1663,
    "phase0_dry_run_half_not_available": True,
    "interpretation": {
        "ratio_0.5": "half is first-N subset (50% passages full-redact)",
        "ratio_0.2": "half is density-half (~20% redaction density per passage)",
        "observed_ratio": round(ratio, 4) if ratio is not None else None,
        "closest_to": classify(ratio),
    },
    "full_per_item_counts_first20": full["per_item_counts"][:20],
    "full_per_item_counts_last20": full["per_item_counts"][-20:],
    "half_per_item_counts_first20": half["per_item_counts"][:20],
    "half_per_item_counts_last20": half["per_item_counts"][-20:],
}

with open(os.path.join(WD, "redaction_stats_apples_to_apples.json"), "w") as f:
    json.dump(out, f, indent=2)

print("SANITY_PASS" if sanity_pass else "SANITY_FAIL")
print(f"half_total={half['total_redacted']} (D baseline=325)")
print(f"full_total={full['total_redacted']}")
print(f"ratio={ratio}")
print(f"closest_to={classify(ratio)}")
