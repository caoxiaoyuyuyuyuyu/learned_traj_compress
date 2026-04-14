# sed worker 5 changelog (v3.6)

**Scope:** factual correction for phase1d_v2 per-N canonical partition (3 sites).
**Parent:** be1e615 (sed 4 polish)

## Background

`worker_verify_table3_bm25_split_consistency` confirmed phase1d_v2 test is **canonical
per-N partitions** (n=251/131/125 for N=2/4/8), and exp_006 N=8 vs exp_016 N=8 run
on byte-identical 125 prompts. The 0.025 vs 0.021 gap is cross-run decoding noise,
not split drift. sed 3 (Table 3 caption + app:exp006_full_test stub) and sed 4
(tab:bm25 caption) used misleading "subset" / "full test set" / "all rows on n=125"
language that this sed corrects.

## Edits (3)

### (i) Table 3 `tab:oracle_api` caption — section3_distillation.tex L67

**Before** (sed 3 A.1 text):
> Numbers in this table are computed on the phase1d_v2 N=8 test subset (n=125);
> the exp_006 canonical N=8 full test set yields 0.025 → 0.498 = +47.3pp
> (reported in Appendix~\ref{app:exp006_full_test}).

**After** (Director v10 verbatim):
> Numbers in this table are computed on canonical phase1d_v2 per-N test partitions
> (n_{N=2}=251, n_{N=4}=131, n_{N=8}=125); the N=8 row reflects exp_016 re-eval on
> the identical 125 prompts used by exp_006 canonical (0.021 vs 0.025 cross-run
> decoding noise, within ±0.02 pre-registered tolerance; see Appendix
> ~\ref{app:exp006_full_test}).

### (ii) `app:exp006_full_test` stub — appendix.tex L470-473

**Subsection title:** `exp_006 Canonical Full Test Set Cross-Reference` →
`exp_006 Canonical Cross-Reference` (label unchanged: `app:exp006_full_test`).

**Body before** (sed 3 A.2):
> The paper body reports honest_search N=8 baseline on the phase1d_v2 n=125 subset
> ... The earlier exp_006 canonical run on the N=8 **full test set** yields
> em_partial 0.025 → 0.498 ... a +47.3pp improvement. The 0.4pp baseline difference
> ... reflect **subset sampling noise**; the substantive conclusion ... is stable
> across both denominators. The bootstrap confidence intervals ... are constructed
> on the n=125 subset ...

**Body after** (Director v10 verbatim):
> The paper body reports honest_search N=8 baseline on the phase1d_v2 n=125
> partition (em_partial = 0.021) for consistency with the paired-bootstrap
> decomposition in exp_016. The exp_006 canonical N=8 evaluation on the
> **same 125-prompt phase1d_v2 partition** yields em_partial 0.025 → 0.498 = +47.3pp.
> The 0.4pp baseline drift is cross-run decoding noise (phase1d_v2 N=8 partition is
> deterministic; differences arise from independent model-forward runs), not a
> subset sampling artefact. The bootstrap confidence intervals reported in
> Section~\ref{sec:distillation} are constructed on these 125 prompts to match the
> leakage decomposition samples.

### (iii) `tab:bm25` caption — section3_distillation.tex L133

**Before** (sed 4 (c)):
> All rows on phase1d_v2 N=8 test subset (n=125), matching Table~\ref{tab:oracle_api}.

**After** (Director v10 verbatim):
> Per-N canonical phase1d_v2 partitions (n=251/131/125 for N=2/4/8), matching
> Table~\ref{tab:oracle_api}. All rows re-evaluated on these fixed partitions.

## Forbidden zone grep counts (across .tex sources)

Unchanged from sed 4:

| Pattern | section3 | appendix | total |
|---------|---------:|---------:|------:|
| `11.0`  | 3 | 1 | 4 |
| `36.7`  | 1 | 2 | 3 |
| `23%`   | 4 | 1 | 5 |
| `77%`   | 4 | 1 | 5 |
| `73.8`  | 1 | 0 | 1 |
| `8.8`   | 2 | 1 | 3 |
| `13.2`  | 2 | 1 | 3 |
| `33.5`  | 1 | 2 | 3 |
| `40.0`  | 1 | 2 | 3 |

`figure5_dose_response.py`, `registry.yaml`, `decisions.yaml`, `_HASH.tex` snapshots
all untouched.

## Sanity grep (post-edit)

- `47.3` in section3_distillation.tex: 0 matches (caption no longer carries it; lives only in app stub)
- `47.3` in appendix.tex: 1 match (app stub L473) ✓
- `on phase1d.*N=8 test subset` in section3: 0 ✓
- `All rows on phase1d` in section3: 0 ✓
- `full test set` in appendix: 0 ✓ (subsection retitled)
- `subset sampling noise` in appendix: 0 ✓
- `per-$N$ test partitions` in section3: 1 (Table 3 caption) ✓
- `n=251/131/125` in section3: 1 (tab:bm25 caption) ✓

## Files touched
- `artifacts/_project/paper/section3_distillation.tex`
- `artifacts/_project/paper/appendix.tex`
- `artifacts/_project/paper/v3_6_sed_worker_5_changelog.md` (new)
