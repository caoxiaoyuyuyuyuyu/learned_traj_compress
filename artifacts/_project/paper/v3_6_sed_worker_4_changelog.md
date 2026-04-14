# v3.6 sed worker 4 changelog

Worker: `worker_rewrite_paper_v3_6_sed_worker_4`
Date: 2026-04-14
Scope: Director v9 polish (4 minor corrections) — Deviation 3 authoritative
citation swap, Deviation 3 "sub-selection granularity" wording rewrite,
tab:bm25 caption split disclosure, and 禁区 UNCHANGED verification.
Base: sed worker 3 DONE (commit 7e823ad — Issue 3 table 3 caption +
app:exp006_full_test stub + authoritative 0.52/867/1663 + probe footnote).

---

## Task (a) — Deviation 3 citation rewrite

File: `appendix.tex` L414 (Deviation 3 body; single long `\item` line).

Before:
> Verification via preprocessing-level authoritative count:
> `\texttt{exp\_016\_workdir/redaction\_stats.json}` (full) and
> `\texttt{redaction\_stats\_half.json}` (half).

After (Director v9 verbatim):
> Verification via preprocessing-level authoritative count:
> `\texttt{exp\_016\_workdir/residual\_leakage\_probe.json}`
> (n\_redacted\_substitutions\_applied: 1663 full / 867 half, ratio `$0.52$`).

Reason: the previous citation pointed to `redaction_stats_half.json` whose
eval-details serialization still reports 325 (apples-to-oranges). A reviewer
following the pointer would read the wrong number. `residual_leakage_probe.json`
is the authoritative preprocessing-level count (1663 / 867 / 0.52).

---

## Task (b) — Deviation 3 "sub-selection granularity" rewrite

File: `appendix.tex` L414 (same long `\item` line, earlier in the sentence).

Before:
> ...and is not the authoritative count), implemented at sub-selection
> granularity. The deviation preserves the plan's scientific intent...

After (Director v9 verbatim sentence inserted):
> ...and is not the authoritative count). The half condition reduces
> per-substitution density uniformly across all 125 passages; the
> differential is implemented at substitution-sampling level (within
> `\texttt{redact\_info\_block\_tag}`), not at passage-level subsetting.
> The deviation preserves the plan's scientific intent...

The trailing ", implemented at sub-selection granularity" clause was removed
and replaced with the Director v9 verbatim sentence. The `867 vs 1663 / 0.52`
authoritative parenthetical and the `eval-details 325 vs 753` caveat are
UNCHANGED (as required).

---

## Task (c) — tab:bm25 caption split disclosure

File: `section3_distillation.tex` L133 (`tab:bm25` `\caption{...}`).

Appended to caption, verbatim:
> All rows on phase1d\_v2 $N{=}8$ test subset ($n=125$), matching
> Table~\ref{tab:oracle_api}.

New caption tail text (L133):
> ...The remaining gap to oracle reflects retrieval quality, not student
> strategy. All rows on phase1d\_v2 $N{=}8$ test subset ($n=125$), matching
> Table~\ref{tab:oracle_api}.

No double period; sentence flows as a trailing disclosure clause.

---

## Task (d) — 禁区 UNCHANGED verification

Grep counts in `section3_distillation.tex` + `appendix.tex`
(sed 4 edits touched only the Deviation 3 `\item` line and the tab:bm25
caption; none of these forbidden numbers appear in the affected regions):

| Pattern | s3 / app | Status |
|---|---|---|
| `11\.0` retrieval residual | 3 / 1 | ✅ UNCHANGED |
| `36\.7` leakage delta | 1 / 2 | ✅ UNCHANGED |
| `23\%` decomposition | 4 / 1 | ✅ UNCHANGED |
| `77\%` decomposition | 4 / 1 | ✅ UNCHANGED |
| `73\.8` span-importance | 1 / 0 | ✅ UNCHANGED |
| `8\.8` CI lower | 2 / 1 | ✅ UNCHANGED |
| `13\.2` CI upper | 2 / 1 | ✅ UNCHANGED |
| `33\.5` CI lower | 1 / 2 | ✅ UNCHANGED |
| `40\.0` CI upper | 1 / 2 | ✅ UNCHANGED |

`_HASH.tex` snapshots, `figure5_dose_response.py`, `registry.yaml`,
`decisions.yaml`: NOT TOUCHED.

---

## Task F — Grep sanity sweep

| Pattern | Expected | Actual | Status |
|---|---|---|---|
| `redaction_stats_half.json` in appendix.tex | 0 | 0 | ✅ |
| `residual_leakage_probe.json` in appendix.tex | ≥1 | 2 (L414 Deviation 3 + L429 probe ¶) | ✅ |
| `All rows on phase1d\_v2` in section3_distillation.tex | 1 (tab:bm25 caption) | 1 (L133) | ✅ |

---

## Files modified

- `artifacts/_project/paper/appendix.tex` (Tasks a + b — single Deviation 3 `\item` line)
- `artifacts/_project/paper/section3_distillation.tex` (Task c — tab:bm25 caption)
- `artifacts/_project/paper/v3_6_sed_worker_4_changelog.md` (this file)

Commit: paper v3.6 sed 4 polish: deviation 3 auth citation + tab:bm25 split caption
