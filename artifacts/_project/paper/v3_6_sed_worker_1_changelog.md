# v3.6 sed worker 1 — changelog

Scope: 3 independent tasks (no_api→honest_search rename, leakage CI injection,
latency hedge). Half-naming dispute (task 4) strictly untouched — awaiting
apples-to-apples verification.

Source of CI values: `artifacts/exp_016_workdir/bootstrap_redaction_ci.json`
field `oracle_full_vs_oracle_redacted_full_delta` = {delta 0.367, ci95_lo 0.335,
ci95_hi 0.400, n=125, 10k paired bootstrap, seed 42, p_boot_gt0 = 1.0}.

---

## Task 1 — no_api → honest_search rename

### abstract.tex
- No `no_api` occurrences pre-existed (abstract used prose only). No rename needed.

### section1_introduction.tex
- No `no_api` occurrences pre-existed. No rename needed.

### section5_conclusion.tex
- No `no_api` occurrences pre-existed. No rename needed.

### section3_distillation.tex
- L26 prose: `In the no-API setting both students...` → `In the \texttt{honest\_search} setting both students...`
- L58 table row: `SFT (no API) & 0.112 & 0.084 & 0.025` → `SFT (honest search) & 0.112 & 0.084 & 0.025`
- L118 table row (cross-backbone): `SFT (no API)` → `SFT (honest search)`
- L76 itemize: `\texttt{no\_api}` → `\texttt{honest\_search}` (em_partial dose-response sequence)
- L77 itemize: `vs \texttt{no\_api}` → `vs \texttt{honest\_search}`
- L36 §3.2 capacity: `\text{em}_{\text{no\_api}}` → `\text{em}_{\text{honest\_search}}`
- L88 Table header `$\Delta$ vs no\_api` → `$\Delta$ vs honest\_search`
- L90 Table row `no\_api & $0.021$ & ---` → `honest\_search & $0.021$ & ---`
- L96 Caption `\texttt{oracle\_redacted\_full} $-$ \texttt{no\_api}` → `\texttt{oracle\_redacted\_full} $-$ \texttt{honest\_search}`
- L104 Figure5 caption: `\texttt{no\_api}` (2 occurrences in `vs \texttt{no\_api}` and `\texttt{no\_api} floor`) → `\texttt{honest\_search}`
- L179 cross-backbone caption: `near-floor no-API performance` → `near-floor \texttt{honest\_search} performance`

### appendix.tex
- L384: `structurally degenerate in the no-API setting` → `structurally degenerate in the \texttt{honest\_search} setting`
- L395 §protocol: `(\texttt{no\_api} / \texttt{oracle\_full} / ...)` → `(\texttt{honest\_search} / \texttt{oracle\_full} / ...)`
- L401 rationale: `collapse \texttt{oracle\_redacted\_full} toward the \texttt{no\_api} floor` → `... \texttt{honest\_search} floor`
- L439 Table header: `$\Delta$ vs \texttt{no\_api}` → `$\Delta$ vs \texttt{honest\_search}`
- L441 Table row: `\texttt{no\_api} & $0.021$` → `\texttt{honest\_search} & $0.021$`
- L447 Table caption: two occurrences of `\texttt{no\_api}` → `\texttt{honest\_search}`
- L349 dispersion narrative: `$0.014~[0.005,0.025]$ no-API)` → `$0.014~[0.005,0.025]$ \texttt{honest\_search})`

### response_letter_v3_6.md
- L29, L38, L40, L47, L52, L62, L63, L66, L94, L108: all `no_api` → `honest_search` (replace_all, 10 substitutions) covering condition label, table header, table row, prose references to the floor.

### figures/figure5_dose_response.py
- L11 comment: `vs no_api (pp)` → `vs honest_search (pp)`
- L16 comment: `no_api + delta_lo` → `honest_search + delta_lo`
- L22 constant: `NO_API = 0.021` → `HONEST_SEARCH = 0.021`
- L27–28 axhline label: `no_api floor` → `honest_search floor` (and variable rename)

### Preserved (intentionally NOT renamed)
- `artifacts/_project/paper/response_letter_v3_6.md:143` — filename reference
  `exp_016_workdir/no_api_latency_probe.json` (actual on-disk artifact path).
- `artifacts/_project/paper/section3_distillation.tex:67` — filename reference
  `\texttt{exp\_016\_workdir/no\_api\_latency\_probe.json}` inside Table 3 caption
  (pointer to on-disk artifact; renaming would break the auditable link).
- `artifacts/_project/paper/data_forensics/llama1b_n8_action_seq_accs.json` —
  `"source": "...exp_013/eval_no_api/..."` (literal disk path from upstream eval).
- `*.tex` hash-suffixed backup files (`section3_distillation_e1347d.tex`,
  `section1_introduction_5d6322.tex`, `section5_conclusion_9d10cb.tex`,
  `appendix_41192a.tex`) — these are pre-v3.6 snapshots and out of scope.

## Task 1 — honest_search methodology sentence injection

Location: `artifacts/_project/paper/section3_distillation.tex`, Table 3
(`tab:oracle_api`) caption (L67). Inserted inline in the caption between the
bootstrap note and the teacher footnote:

> "The `honest_search` condition is a real-FlashRAG retrieval baseline (not a
> *no-search* setting): the student's `<search>` queries are executed by the
> production FlashRAG retrieval backend, returning real passages to the student
> via `<information>` blocks. The `oracle_*` conditions bypass retrieval entirely
> and inject teacher-replayed passages. The 4.72× latency gap between
> `honest_search` and `oracle_full` (3666.7s vs 777.4s over 125 prompts) is fully
> accounted for by FlashRAG retrieval I/O overhead (~23s/prompt; see
> Appendix D and `exp_016_workdir/no_api_latency_probe.json`)."

Rationale: Table 3 caption is the first place the honest_search baseline is
named in §3, making it the load-bearing place for disambiguation.

---

## Task 2 — leakage CI `+36.7pp [+33.5, +40.0]` injection (5 locations)

1. **abstract.tex L2**: `${\sim}37$pp ($77\%$)` → `$+36.7$\,pp $[+33.5, +40.0]$ ($\approx 77\%$)`
2. **section1_introduction.tex L10**: `the remaining ${\sim}37$pp (${\sim}77\%$)` → `the remaining $+36.7$\,pp $[+33.5, +40.0]$ ($\approx 77\%$)`
3. **section3_distillation.tex L79** (§3.5 decomposition itemize): `contributes $\approx 36.7$\,pp ($\approx 77\%$)` → `contributes $\approx 36.7$\,pp $[+33.5, +40.0]$ ($\approx 77\%$; paired bootstrap 10k resamples, seed 42, on \texttt{oracle\_full} $-$ \texttt{oracle\_redacted\_full}, $p < 0.0001$)`
4. **response_letter_v3_6.md L54** (§3 Decomposition): `≈+36.7pp (≈77%). This is the gap...` → `+36.7pp [+33.5, +40.0] (≈77%; paired bootstrap 10k resamples, seed 42, \`oracle_full − oracle_redacted_full\`, p<0.0001; source ...bootstrap_redaction_ci.json field oracle_full_vs_oracle_redacted_full_delta)`
5. **appendix.tex L444–447** (`tab:redaction_bootstrap_ci`): added 4th logical delta row below `\bottomrule`-adjacent `oracle_full` row. Inserted a `\midrule` then a `\multicolumn{4}{l}{...}` heading "Leakage component = oracle_full − oracle_redacted_full:" and a new row `\quad leakage (paired) & --- & $+36.7$ & $[+33.5,\;+40.0]$`. Caption updated to add `$+36.7$\,pp $[+33.5, +40.0]$ (77%; paired-bootstrap 10k resamples, seed 42, $p < 0.0001$; source ...bootstrap_redaction_ci.json field oracle_full_vs_oracle_redacted_full_delta)`.

Format consistency: all 5 locations write `+36.7pp [+33.5, +40.0]` or its LaTeX
equivalent `$+36.7$\,pp $[+33.5, +40.0]$`. No remaining `≈37pp`/`~37pp` at the 5
injection sites.

Note: other mentions of `$\sim 37$pp (77%)` in non-injection locations (Figure 1
framework caption L15, §3.2 capacity prose L36, Figure 5/gap_decomposition
captions, Synthesis L144, conclusion L6) were **not** updated — the Director
spec listed exactly 5 injection sites. These surviving `~37pp` phrasings are
consistent with each other and with the injected CI (both refer to the same
value, 36.7 rounding to 37).

---

## Task 3 — latency hedge methodology note

Location: `artifacts/_project/paper/response_letter_v3_6.md`. No pre-existing
`3666`/`4.72`/`latency` content → **inserted new §8** after §7 (broad-scope
rationale). Verbiage matches the Director-specified block verbatim:

> ### 8. Methodology note — latency disparity between honest_search and oracle_*
>
> The `honest_search` condition runs ~4.72× slower than `oracle_full` (3666.7s
> vs 777.4s over 125 prompts, 2889.3s delta). This latency disparity is **not**
> attributable to output length: average generation length is 325.7 word-tokens
> for `honest_search` vs 315.4 for `oracle_full`, a <4% delta
> (exp_016_workdir/no_api_latency_probe.json). The latency is fully accounted
> for by FlashRAG retrieval I/O overhead (~23s/prompt), which the `oracle_*`
> conditions bypass entirely by injecting teacher-replayed passages directly.
> `avg_searches = 1` and `truncated_rate = 0` for both conditions confirm results
> are trustworthy — the `honest_search` samples are not degenerate or
> pathologically-truncated fallbacks. This disclosure does not make a root-cause
> claim about internal retrieval timing variance (TGI warm-cache differences,
> retry timeout distribution), only that output-length confounding is excluded.

---

## Task 4 — frozen (untouched)

Zero edits to: half-subset / half-density / 20% density / odd-indexed prose,
`oracle_redacted_half` symbol, Figure 5 50% x-axis label, Insight β narrative,
`app:redaction` half-condition description, decisions.yaml D061. Confirmed via
post-edit grep: half-related strings appear only in pre-existing (untouched)
locations (appendix.tex L395 L416, response_letter_v3_6.md L33,
section3_distillation.tex L74).

---

## Post-edit verification (final grep)

Command: `rg '(no_api|no\\_api|no API|no-API|NO_API)' on 6 target files`

Residuals (all intentionally preserved):
- `response_letter_v3_6.md:143` — filename `exp_016_workdir/no_api_latency_probe.json` inside §8 latency block.
- `section3_distillation.tex:67` — filename `\texttt{exp\_016\_workdir/no\_api\_latency\_probe.json}` inside Table 3 caption.

Command: grep `+36.7.{0,5}\[\+33.5` → 5 hits, one per injection site (abstract,
intro, section3, response_letter, appendix). ✓

Command: grep `half-subset|half-density|20% density|odd-indexed` → 4 hits, all
pre-existing, zero edits touched them. ✓

## Files modified

- artifacts/_project/paper/abstract.tex
- artifacts/_project/paper/section1_introduction.tex
- artifacts/_project/paper/section3_distillation.tex
- artifacts/_project/paper/appendix.tex
- artifacts/_project/paper/response_letter_v3_6.md
- artifacts/_project/paper/figures/figure5_dose_response.py

## Files NOT modified (intentionally out of scope)

- artifacts/_project/paper/section5_conclusion.tex — no `no_api` references to rename; `~37pp` mention at L6 is not one of the 5 listed CI injection sites (per Director spec).
- Any *.tex hash-suffixed backup file.
- CLI flag strings in exp_016 scripts.
- data_forensics/llama1b_n8_action_seq_accs.json source path.
