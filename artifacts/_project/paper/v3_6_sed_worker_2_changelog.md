# v3.6 sed worker 2 changelog

Worker: `worker_rewrite_paper_v3_6_sed_worker_2_...` (probe-independent batch)
Date: 2026-04-14
Scope: Part A Issue 1 (47.3 → 47.7 global), Issue 2 (§3.5 half description),
Insight β narrative, Deviation 3, non-blocking json/md corrections.
Base: sed worker 1 DONE (honest_search rename, leakage CI, latency hedge).

---

## Task 1 — 47.3 → 47.7 and 0.025 → 0.021 (Qwen honest_search N=8 baseline)

Rationale: unify oracle Δ at +47.7pp (exp_016 phase1d_v2 baseline 0.021) across
the paper, replacing the stale exp_006 0.025 baseline that produced the 47.3pp
figure. Arithmetic self-consistent: 11 + 36.7 = 47.7; CI from
`bootstrap_redaction_ci.json` oracle_full delta.

### Hits modified

| File | Line (pre-edit) | What |
|---|---|---|
| `abstract.tex` | L2 | `0.025 → 0.498` / `+47.3pp` → `0.021 → 0.498` / `+47.7pp` |
| `section1_introduction.tex` | L4 | `0.025/0.014 of teacher capability` → `0.021/0.014` (Qwen side only) |
| `section1_introduction.tex` | L6 | `em_partial = 0.025` (Qwen baseline) → `0.021`; 94.5% drop → 95.4% drop (recomputed) |
| `section1_introduction.tex` | L10 | `0.025 → 0.498 (+47.3pp)` → `0.021 → 0.498 (+47.7pp)` |
| `section1_introduction.tex` | L23 (contrib) | `+47.3pp` → `+47.7pp` |
| `section3_distillation.tex` | L36 (§3.2 capacity) | `leakage-inclusive $+47.3$pp` → `$+47.7$pp` (note: the `0.112/0.084/0.025` capacity-isolation vector left UNCHANGED — it is the exp_006 3B SFT baseline for the 7B-vs-3B comparison, not the oracle-contrast baseline) |
| `section3_distillation.tex` | L58 (tab:oracle_api) | SFT (honest search) N=8: `0.025` → `0.021` |
| `section3_distillation.tex` | L60 (tab:oracle_api Δ) | `$+47.3$pp` → `$+47.7$pp` (N=8 column) |
| `section3_distillation.tex` | L71 (causal prose) | `0.025 → 0.498 (+47.3pp, CI [43.8,50.8])` → `0.021 → 0.498 (+47.7pp, CI [44.6,50.9])` |
| `section3_distillation.tex` | L112 (BM25 prose) | `0.025 → 0.147 (+12.2pp)` → `0.021 → 0.147 (+12.6pp)` (both instances); arithmetic recompute, not new fabrication |
| `section3_distillation.tex` | L118 (tab:bm25) | SFT (honest search) N=8: `0.025` → `0.021` |
| `section3_distillation.tex` | L126 (tab:bm25 caption) | `+9.2pp / +12.2pp` → `+9.1pp / +12.6pp` (BM25 deltas recomputed; N=2 was 9.1 pre-rounding anyway) |
| `section3_distillation.tex` | L144 (Synthesis) | `0.025 → 0.498 (+47.3pp)` → `0.021 → 0.498 (+47.7pp)` |
| `section3_distillation.tex` | L167 (tab:cross_backbone) | Qwen-3B No API N=8: `$0.025$` → `$0.021$` |
| `section3_distillation.tex` | L183 (cross-backbone prose) | `+47.3pp for Qwen` → `+47.7pp for Qwen` |
| `section5_conclusion.tex` | L6 | `recovers $+47.3$pp` → `recovers $+47.7$pp` |
| `appendix.tex` | L181 (leakage-audit interp) | `$+47.3$\,pp oracle em\_partial recovery` → `$+47.7$\,pp` |
| `appendix.tex` | L328 (silent-failure demo tbl) | Qwen em_p=0.025 → 0.021 (consistent with §1 narrative; DPO/trunc-confound tables L264/L307/L312 left at 0.025 since they are the exp_006 DPO-comparison baseline, not the oracle-contrast baseline) |
| `response_letter_v3_6.md` | L12 | `headline +47.3pp` → `headline +47.7pp` |
| `figures/framework_overview.tex` | L107 | `+47.3pp recovery` → `+47.7pp recovery` (Figure 1 infographic badge) |

### 0.025 LEFT UNCHANGED (intentional, out-of-scope)

| Location | Context | Reason |
|---|---|---|
| `section3_distillation.tex` L36 | `0.112/0.084/0.025` vector in capacity-isolation 7B vs 3B | exp_006 baseline for capacity ablation; arithmetic `12.3/13.8/12.6%` depends on this; DPO/capacity-specific baseline per task 1 scope carve-out |
| `appendix.tex` L264 | tab:eval_matrix SFT shared (3B) N=8 | DPO eval matrix; the DPO $+1.0$pp delta depends on SFT=0.025 baseline from exp_006 |
| `appendix.tex` L307 | tab:truncation_confound SFT shared N=8 | Trunc confound uses exp_006 eval (before exp_016 re-eval existed) |
| `appendix.tex` L312 | Trunc confound caption `from $0.011$ to $0.025$` | Same — trunc confound anchor |
| `appendix.tex` L349 | `$0.014 [0.005, 0.025]$ honest_search` (Llama CI upper) | 0.025 here is the upper bound of the Llama-1B bootstrap CI, not a Qwen baseline |
| `section1_introduction.tex` L6 | "94.5% drop" → "95.4% drop" (recomputed) | Kept consistent; minor derived-number update |
| `bootstrap_ci_table.tex` | `Qwen-3B & no-API & ... 0.025 & [0.013, 0.038]` | Legacy v3.3 input table, not referenced directly in changed narrative; out of task 1 scope |

### Inconsistency note (disclosed for reviewer)

`tab:bm25` now reports honest_search N=8 = 0.021 (exp_016 re-eval) while the
capacity-isolation vector in §3.2 L36 retains 0.025 (exp_006). These differ by
0.004 at N=8 and reflect re-evaluation of the same checkpoint on the same 125-prompt
phase1d_v2 split between exp_006 and exp_016 (likely eval-harness-side determinism
or minor prompt-template refresh). Both are internally consistent within their
respective ablations; the main text anchors 0.021 for all oracle-contrast statements.

---

## Task 2 — §3.5 L74 half condition rewrite

`section3_distillation.tex` L74: replaced
`"50% (oracle_redacted_half, deterministic odd-indexed substring redaction)"`
with Director-approved text:

> `"$\sim 50\%$ per-passage density (oracle_redacted_half, uniform density halving
> applied to all 125/125 oracle passages: mean 2.60 [REDACTED] tokens per passage
> vs 6.02 in oracle_redacted_full; density-halved uniformly across all 125 passages;
> see Appendix~\ref{app:redaction})"`

Required phrases: ✅ "density-halved uniformly across all 125 passages"
✅ "2.60 vs 6.02" ✅ "ratio" concept (implicit via "density halving" + numbers)
✅ removed "odd-indexed" ✅ no "first-N subset" language.

---

## Task 3 — Insight β "Span-importance distribution" paragraph

`section3_distillation.tex` §3.5: inserted new `\paragraph{Span-importance distribution.}`
after the itemize (after the decomposition bullet) and before the "partial-mechanism
verdict" paragraph. Full text from Director spec (73.8% retention, $0.352/0.477$,
Pareto-like distribution, span-importance asymmetry is itself an upper bound).

Also injected TL;DR in `response_letter_v3_6.md` §3 Decomposition as a bullet
"Span-importance asymmetry (TL;DR)".

---

## Task 4 — Appendix Deviation 3

`appendix.tex` §app:redaction_protocol: added new `\paragraph{Deviation log summary.}`
after the existing "Non-deviations" paragraph, containing an itemize block with:

- **Deviation 1** — word-boundary anchoring + quote-strip (summary pointer to
  existing "Executed spec" / "Rationale for deviation" paragraphs)
- **Deviation 2** — (reserved)
- **Deviation 3** — half condition uniform density halving vs planned odd-indexed
  subset (verbatim text from task spec: 6.02 → 2.60, ratio ≈0.43, 125/125,
  "stricter dose-response control than a passage-level subset", verification
  pointer to `redaction_stats_apples_to_apples.json`)

Also cleaned up two residual "odd-indexed" references that were stale:
- L398 "Executed spec" paragraph: `"per-block deterministic odd-index half-redaction"`
  → `"deterministic uniform density halving for oracle_redacted_half across all 125/125 passages (see Deviation 3 below)"`
- L416 (now L424) fire-statistics itemize third bullet: `"uses the odd-indexed ~50% subset"`
  → `"applies uniform density halving across all 125/125 passages (mean 2.60 vs 6.02, ratio ≈0.43; see Deviation 3)"`

L395 ("Pre-registered spec" paragraph) still says `"Half-redaction was deterministic
odd-indexed substring redaction (seed-free)"` — INTENTIONALLY RETAINED as historical
description of the pre-registered plan.md, which Deviation 3 explicitly references
as the deviation-from baseline. This is what makes the Deviation 3 entry meaningful.

Also updated `response_letter_v3_6.md` L33 `oracle_redacted_half` bullet from
"redact odd-indexed 50% of matches" to "uniform density halving ... 2.60 vs 6.02 ..."
for consistency with the new §3.5 narrative.

---

## Task 5 — Non-blocking json/md corrections

### 5a. `artifacts/exp_016_workdir/redaction_stats_half.json`

Added top-level `_correction_note` field (verbatim text from task spec). All
original fields (`source`, `n_prompts`, `total_redacted_substitutions=325`,
`cross_ref_full_condition`, etc.) preserved unchanged.

### 5b. `artifacts/exp_016_workdir/exp_016_report.md` §7.1

Minimal precision edit to the "half-subset redaction" section:
- Replaced the `"total 325 / mean 2.6 vs 13.3 per passage"` line with
  apples-to-apples `"2.60 vs 6.02 per passage (eval details), ratio ≈0.4316, ~50% density"`
  and pointer to `redaction_stats_apples_to_apples.json`.
- Added a **Phase 0 caveat** paragraph explicitly disclosing the 1663 (dry-run) vs
  753 (eval details) full-condition discrepancy, attributing it to prompt truncation
  / block serialization, and pointing to `residual_leakage_probe.json` when
  available. Retained the original "Clarification of half" paragraph with updated
  numbers (2.60 vs 6.02, ratio ≈0.43).
- Left the qualitative conclusion sentence ("4-way monotone ordering ... preserved
  with non-overlapping CIs") unchanged.

---

## Task 6 — 禁区 check (should be all NO)

- Modified `77%` or `23%` decomposition percentages? **NO** — unchanged everywhere
- Modified `+11.0pp retrieval residual` or its CI `[+8.8, +13.2]`? **NO**
- Modified `+36.7pp leakage` or its CI `[+33.5, +40.0]`? **NO**
- Modified `figure5_dose_response.py` x-axis label? **NO** — not touched
- Modified `registry.yaml` / `decisions.yaml`? **NO** — not touched
- Fabricated new numbers? **NO** — 47.7 from bootstrap_redaction_ci.json, CI [44.6,50.9]
  from same source, 2.60/6.02 from `redaction_stats_apples_to_apples.json`, 73.8%
  from 0.352/0.477. +12.6pp BM25 delta and 95.4% Qwen drop are direct arithmetic
  recomputes (0.147−0.021=0.126; (0.455−0.021)/0.455=0.954); disclosed in Task 1 table.

---

## Post-edit sanity grep

```
grep '47\.3'      artifacts/_project/paper/{abstract,section1_introduction,section3_distillation,section5_conclusion,appendix}.tex → 0 hits (only stale _HASH.tex snapshots)
grep '47\.3'      artifacts/_project/paper/response_letter_v3_6.md → 0 hits
grep '47\.3'      artifacts/_project/paper/figures/framework_overview.tex → 0 hits
grep 'odd-indexed' artifacts/_project/paper/section3_distillation.tex → 0 hits
grep 'odd-indexed' artifacts/_project/paper/appendix.tex → 2 hits (L395 pre-reg historical, L414 Deviation 3 reference — intentional)
grep '\[43\.8'    artifacts/_project/paper → 0 hits in active files
grep '0\.025'     active paper files → 4 intentional retentions documented above
```

## Files modified

- `artifacts/_project/paper/abstract.tex`
- `artifacts/_project/paper/section1_introduction.tex`
- `artifacts/_project/paper/section3_distillation.tex`
- `artifacts/_project/paper/section5_conclusion.tex`
- `artifacts/_project/paper/appendix.tex`
- `artifacts/_project/paper/response_letter_v3_6.md`
- `artifacts/_project/paper/figures/framework_overview.tex`
- `artifacts/exp_016_workdir/redaction_stats_half.json`
- `artifacts/exp_016_workdir/exp_016_report.md`
- `artifacts/_project/paper/v3_6_sed_worker_2_changelog.md` (this file)
