# v3.6 sed worker 3 changelog

Worker: `worker_rewrite_paper_v3_6_sed_worker_3_...`
Date: 2026-04-14
Scope: Director v6 Issue 3 (Table 3 n=125 disclosure + app:exp006_full_test stub)
+ Director v7 authoritative ratio 0.52/867/1663 override of eval-details 0.43/2.60/6.02
+ residual leakage probe paragraph
+ final grep sanity.
Base: sed worker 2 DONE (47.7 global rewrite, §3.5 L74 half description,
Insight β paragraph, Deviation 3, non-blocking json/md corrections).

---

## Task A — v6 Issue 3: Table 3 n=125 disclosure + app:exp006_full_test stub

### A.1 `section3_distillation.tex` L67 (tab:oracle_api caption)

Appended to Table 3 caption (after `\textsuperscript{$\dagger$}...not valid.`):

> Numbers in this table are computed on the phase1d\_v2 $N{=}8$ test subset
> ($n=125$); the exp\_006 canonical $N{=}8$ full test set yields
> $0.025 \to 0.498 = +47.3$pp (reported in Appendix~\ref{app:exp006_full_test}).

### A.2 `appendix.tex` L468 new `\subsection` stub

Inserted new subsection `\label{app:exp006_full_test}` immediately after
`app:redaction_limitations` body (before any later appendix material).
Content verbatim from task spec: explains the 0.021 (n=125) vs 0.025 (full)
baseline difference as subset sampling noise, anchors the bootstrap CIs to
the n=125 subset, and cross-refs `\ref{sec:distillation}`.

The `\ref{app:exp006_full_test}` in A.1 now resolves to this label.

---

## Task B — Authoritative ratio 0.52/867/1663 override

### B.1 `section3_distillation.tex` §3.5 half description

Covered the sed 2 `2.60` / `6.02` wording with Director v7 verbatim text
(tracks `\texttt{redact\_info\_block\_tag}` → `867` vs `1663` substitutions,
ratio `0.52`).

### B.2 `section3_distillation.tex` Insight β paragraph (`\paragraph{Span-importance distribution.}`)

Updated the opening sentence from
`"Halving the per-passage substitution density (from 6.02 to 2.60 average [REDACTED] tokens per passage ...)"`
to
`"Halving the per-passage substitution density (from 1663 to 867 total [REDACTED] substitutions across all 125/125 passages, ratio 0.52 via the preprocessing-level redact_info_block_tag tool)"`.

**73.8%** and the `0.352 / 0.477` derivation **UNCHANGED** (禁区).

### B.3 `appendix.tex` Deviation 3

Rewrote Deviation 3 body:
- Retains "half condition uniform density halving (vs planned odd-indexed subset)" title and the historical reference to plan.md's `"odd-indexed 62 of 125"` baseline (required for the deviation to make sense).
- Replaces the half-condition measurement with authoritative numbers: `867 vs 1663 substitutions via preprocessing-level redact_info_block_tag, ratio 0.52`.
- Discloses the eval-details JSON layer `325 vs 753` and attributes the discrepancy to `generated_text[:2000]` serialization truncation — NOT authoritative.
- Verification pointers updated to `redaction_stats.json` (full) and `redaction_stats_half.json` (half).

### B.4 `appendix.tex` fire-statistics bullet (L424→L427)

Updated the per-condition bullet under `app:redaction_stats`:
- Old: `"applies uniform density halving across all 125/125 passages (mean 2.60 vs 6.02 [REDACTED] tokens per passage, ratio ≈ 0.43; see Deviation 3)"`
- New: `"oracle_redacted_full uses all 1663 substitutions; oracle_redacted_half applies uniform density halving across all 125/125 passages, yielding 867 substitutions (ratio 0.52 via preprocessing-level redact_info_block_tag; see Deviation 3)"`.

### B.5 `response_letter_v3_6.md` consistency

Two hits in the response letter also referenced the now-superseded
`2.60/6.02/0.43`. Updated both:
- L33-35 `oracle_redacted_half` definition bullet: rewritten to `867 vs 1663 substitutions, ratio 0.52 via preprocessing-level redact_info_block_tag`.
- L61-64 `Span-importance asymmetry (TL;DR)` bullet: rewritten to `867 vs 1663 total substitutions, ratio 0.52`.

---

## Task C — Residual leakage probe paragraph

`appendix.tex` `app:redaction_stats` subsection (after the fire-statistics
itemize) — added new `\paragraph{Serialization vs model-visible context.}`:

> Phase-0 \texttt{redaction\_stats.json} reports 1663 \texttt{[REDACTED]}
> substitutions pre-eval; the eval-output JSON \texttt{details} dict reports
> 753 because \texttt{eval\_sft\_with\_api.py:397} stores \texttt{generated\_text[:2000]}
> as a disk-space measure. A residual leakage probe
> (\texttt{residual\_leakage\_probe.json}) reconstructs the full
> model-visible context from \texttt{oracle\_map} and confirms $0/125$
> blocks retain any unredacted gold substring; the retrieval-delta
> decomposition is therefore robust to the serialization artefact.

Placed in Appendix per task spec recommendation (not inlined in §3.5) to
avoid main-body clutter.

---

## Task D — v5 sed residual sweep

Specified hotspots re-checked via `grep -n '47\.3'`, `grep -n '\bno_api\b'`,
`grep -n 'odd-indexed'`:

| File | Line (spec) | 47.3 residual? | Notes |
|---|---|---|---|
| `section3_distillation.tex` | L60 | **NO** | `+41.8/+43.7/+47.7` (sed 2 done) |
| `section3_distillation.tex` | L71 | **NO** | `+47.7pp, CI [44.6,50.9]` (sed 2 done) |
| `section3_distillation.tex` | L144 | **NO** | Synthesis prose `+47.7pp` (sed 2 done) |
| `section3_distillation.tex` | L183 | **NO** | Cross-backbone prose `+47.7pp` (sed 2 done) |
| `section3_distillation.tex` | L67 (new) | **INTENTIONAL** | Table 3 caption cross-ref to app:exp006_full_test stub (Task A.1) |
| `section5_conclusion.tex` | L6 | **NO** | `+47.7pp` (sed 2 done) |
| `appendix.tex` | L181 | **NO** | Leakage-audit interp `+47.7pp` (sed 2 done) |
| `appendix.tex` | L473 (new) | **INTENTIONAL** | `app:exp006_full_test` stub body `+47.3pp` (Task A.2) |
| `response_letter_v3_6.md` | L12 | **NO** | `+47.7pp` (sed 2 done) |
| `figures/framework_overview.tex` | L107 | **NO** | `+47.7pp recovery` (sed 2 done) |

`\bno_api\b` active-file hits: **0** (`no\_api\_latency\_probe.json`
filename is LaTeX-escaped as `no\_api\_...` and does NOT match `\bno_api\b`;
no plain `no_api` references remain).

`odd-indexed` active-file hits: appendix.tex L395 (pre-reg historical),
L414 (Deviation 3 baseline reference) — both intentional per sed 2
changelog §4. Zero in `section3_distillation.tex`.

---

## Task E — `exp_016_report.md` §7.1 three-level correction

Replaced the sed 2 "apples-to-apples eval-details only" narrative with an
explicit three-level hierarchy:

1. Eval details JSON regex (apples-to-oranges initial): 325 vs 1663 → 0.195 ❌
2. Eval details JSON apples-to-apples: 325 vs 753 → 0.432 (partially correct; affected by `generated_text[:2000]` truncation, mean 2.60 vs 6.02 per passage)
3. Preprocessing `redact_info_block_tag` (authoritative): 867 vs 1663 → 0.521 ✅

Explicit pointer: `"Paper §3.5 and Appendix app:redaction report the authoritative ratio 0.52."`

Also updated the `Phase 0 caveat` paragraph:
- Re-attributes the 1663 vs 753 gap to `generated_text[:2000]` serialization truncation (not the earlier "prompt truncation / block serialization" language).
- Points to `residual_leakage_probe.json` as the definitive check that 0/125 blocks retain unredacted gold substrings.
- Clarifies that the preprocessing-level count is authoritative because it is measured on full pre-serialization text.

And `Clarification of "half"` paragraph:
- Replaces the `2.60 vs 6.02` wording with `"preprocessing-authoritative 867 vs 1663 substitutions, ratio 0.52"`.
- Monotone-ordering line changed from `no_api < redacted_full < ...` to `honest_search < redacted_full < ...` to match sed 1's no_api→honest_search rename convention.

---

## Task F — Final grep sanity (executed in `artifacts/_project/paper/`)

All patterns executed against the paper directory (*.tex + response_letter_v3_6.md + figures/*.tex), excluding `_HASH.tex` snapshots.

| Pattern | Expected | Actual | Status |
|---|---|---|---|
| `47\.3` in abstract/intro/§3/§5 | 0 (except intentional) | 2 hits: `section3_distillation.tex:67` (Table 3 caption cross-ref) + `appendix.tex:473` (app:exp006_full_test stub body) | ✅ both intentional per A.1/A.2 |
| `47\.3` in response_letter_v3_6.md | 0 | 0 | ✅ |
| `47\.3` in figures/framework_overview.tex | 0 | 0 | ✅ |
| `\bno_api\b` in all active paper files | 0 (LaTeX `no\_api` filename exception) | 0 | ✅ |
| `odd-indexed` in section3_distillation.tex | 0 | 0 | ✅ |
| `odd-indexed` in appendix.tex | 2 (L395 pre-reg + L414 Deviation 3) | 2 | ✅ both intentional |
| `\b0\.43\b` in active *.tex | 0 | 0 | ✅ |
| `\b2\.60\b` in active *.tex | 0 | 0 | ✅ |
| `\b6\.02\b` in active *.tex | 0 | 0 | ✅ |
| `\[43\.8` in active *.tex | 0 | 0 | ✅ |
| `50\.8` in active *.tex | 0 | 0 | ✅ |

All `_HASH.tex` historical snapshots left untouched.

---

## Task G — 禁区 confirmed UNCHANGED

Grep counts across `section3_distillation.tex` + `appendix.tex`:

| Number | Count (s3 / app) | Status |
|---|---|---|
| `+11.0pp` retrieval residual | 3 / 1 | ✅ UNCHANGED |
| `+36.7pp` leakage delta | 1 / 2 | ✅ UNCHANGED |
| `23%` / `77%` decomposition | 5 / 1 | ✅ UNCHANGED |
| CI `[+8.8, +13.2]` | 2 / 1 | ✅ UNCHANGED |
| CI `[+33.5, +40.0]` | 1 / 2 | ✅ UNCHANGED |
| `73.8%` span-importance recovery | 1 / 0 | ✅ UNCHANGED |
| `figure5_dose_response.py` x-axis label | — | NOT TOUCHED |
| `registry.yaml` / `decisions.yaml` | — | NOT TOUCHED |

---

## Files modified

- `artifacts/_project/paper/section3_distillation.tex` (A.1, B.1, B.2)
- `artifacts/_project/paper/appendix.tex` (A.2, B.3, B.4, C)
- `artifacts/_project/paper/response_letter_v3_6.md` (B.5)
- `artifacts/exp_016_workdir/exp_016_report.md` (E)
- `artifacts/_project/paper/v3_6_sed_worker_3_changelog.md` (this file)
