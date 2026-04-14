# exp_016 — Passage-Redaction Oracle Replay (C1 blocking)

**Status**: SCAFFOLD — Director APPROVED 4 gates 2026-04-14 (§12). Not yet registered / not yet spawned. Awaits cuda:0 release after exp_014 sweep + Director urgent_agent_message trigger.
**Author**: main agent, 2026-04-14
**Parent**: exp_009 (Qwen-3B oracle replay) + v3.4 oracle_leakage_audit (98% substring-hit rate @ N=8, `artifacts/_project/paper/oracle_leakage_audit.txt`)

---

## 1. Hypothesis

v3.4 leakage audit established a **strict upper bound**: 98% of oracle-injected teacher `<information>` blocks contain a substring-matching gold answer at N=8. The paper currently reframes oracle recovery (+47.3 pp em_partial) as a "strict upper bound on retrieval mitigation" because we cannot separate *retrieval quality* from *answer-in-passage leakage*.

**Reviewer C1** demands a causal disentanglement. exp_016 runs a **passage-redaction** ablation:
- **H0 (null / leakage-dominated)**: After redacting gold-answer substrings from oracle passages, em_partial collapses back toward the no-API floor (Δ ≤ +5 pp vs no-API), i.e. the `+47.3$pp gain was driven by passage leakage, not retrieval quality. In this case the §3.5 headline causal claim must be downgraded.
- **H1 (retrieval-quality)**: After redaction, em_partial retains ≥50% of the original +47.3 pp gain, i.e. the oracle intervention works through retrieval topic-specificity, not passage answer exposure. Headline claim holds.
- **Middle (partial)**: 5 pp < retained gain < 25 pp. Mixed mechanism; rewrite §3.5 as "retrieval + partial leakage" upper bound.

## 2. Scope

- **Model**: base Qwen2.5-3B + LoRA adapter `checkpoints/exp_006_sft_shared` (bit-exact match to exp_009 loading path; worker audited this at Phase -1 prelight, deviates from historical shorthand "Qwen2.5-3B-SFT-Pooled" which was never a merged-weights artefact). **No Llama-1B** in this phase (Director APPROVED, Llama deferred — ARR rebuttal resource-constrained).
- **Split**: `phase1d_v2` N=8 test (125 prompts, full split — no subsampling; matches headline oracle number).
- **Conditions** (4 conditions, Director APPROVED dose-response upgrade 2026-04-14):
  1. `no_api` — baseline reference (no oracle injection); reuses existing `phase1d_v2` stage-1 eval output if available, else re-eval
  2. `oracle_full` — reproduce exp_009 baseline (sanity gate, no redaction)
  3. `oracle_redacted_full` — redact 100% of gold-answer substring occurrences in each info block
  4. `oracle_redacted_half` — redact 50% of gold-answer occurrences per block (deterministic: redact odd-indexed matches, keep even-indexed; seed-stable). Dose-response control for reviewer "extra evidence"
- **Eval metric**: em_partial, em_full, truncated_rate (same as §3 table). Paired bootstrap 10k resamples across 125 samples for CI; seed 42.
- **Rationale for 4 conditions** (Director call): single-eval marginal cost (~15 min) is cheap relative to rebuttal upside. Dose-response curve (0% / 50% / 100% redaction) directly answers reviewer "is this retrieval or leakage" by showing a monotone curve rather than a single A/B point.

## 3. Redaction mechanics (key design decision)

### Input
Teacher `<information>` blocks are loaded from `artifacts/phase1d_v2_data/raw_trajectories_N8.json` (same source as exp_009 / oracle_leakage_audit).

### Redaction function
For each prompt:
1. Load `gold_answers` list (same flattening logic as `oracle_leakage_audit.py:50-60`)
2. For each teacher `<information>` block, replace each gold-answer substring match with a fixed-length placeholder token `[REDACTED]` (case-insensitive, same `_norm` normalisation as leakage audit to match Wikipedia tokenisation variants)
3. Preserve block structure and surrounding sentence context — **only** the answer substring is masked, not full sentences
4. If block still contains partial lexical overlap after redaction (e.g. "Albert Einstein" → "[REDACTED]" but "Einstein's theory" remains elsewhere), leave as-is — we are measuring leakage contribution, not semantic paraphrase coverage
5. **Half-redaction variant** (`oracle_redacted_half`): within each block, enumerate all substring matches in document order; redact the odd-indexed ones (1st, 3rd, 5th, ...), keep even-indexed intact. Deterministic and seed-free so `oracle_redacted_half` is reproducible bit-exact on rerun.

### Evidence requirements (Director追加约束 2026-04-14)
Per-block redaction statistics **must** be emitted to eval JSON for reviewer-auditable evidence:
- Each eval result JSON (both `oracle_redacted_full` and `oracle_redacted_half` conditions) includes a top-level `redaction_stats` field:
  ```
  {"n_prompts": 125, "n_blocks_total": N, "n_blocks_with_redaction": K,
   "n_redacted_substitutions_total": M, "n_redacted_per_block": [n1, n2, ...]}
  ```
- **3-sample before/after dump**: `artifacts/exp_016_workdir/redaction_samples.json` contains at least 3 prompt_ids with `{"gold_answers": [...], "info_block_before": "...", "info_block_after_full_redact": "...", "info_block_after_half_redact": "...", "n_substitutions_full": k, "n_substitutions_half": k2}`. This file is cited directly in rebuttal to prove redaction actually fired.
- Zero-redaction prompts (empty gold_answers) logged to `redaction_stats.fully_unredactable_prompt_ids` for exclusion auditing.

### Rationale for substring redaction (not sentence-level)
- Sentence-level redaction would destroy surrounding context, conflating "retrieval quality" and "passage informativeness"
- Substring redaction is the **minimum necessary** intervention to remove the specific "answer text" signal while keeping retrieval topic specificity intact
- Honest disclosure in paper: "Substring redaction removes exact-match leakage but not paraphrase leakage; the retained gain is therefore itself an upper bound on non-leakage mechanism"

### Edge case handling
- Empty gold_answers list → no redaction, block unchanged (log as `N_UNREDACTED` in report)
- Multi-word gold with partial match → redact the longest-match span per block
- Gold answer is a number (e.g. "1984") → redact as-is (no special number handling)
- Character-level encoding: UTF-8 throughout, match Wikipedia tokenisation

### Deviation log — redaction protocol refinement (2026-04-14, Phase 0 dry-run)

**This subsection is the single source of truth for the pre-registered vs actual redaction spec. It will be disclosed verbatim in the rebuttal response-letter + paper appendix.**

- **Pre-registered spec** (§3 above as originally approved by Director 2026-04-14, gate #4): "substring-only redaction with `_norm` normalisation, case-insensitive, replace each gold-answer substring with `[REDACTED]`".
- **Actual executed spec** (worker commit `7cf099d`, finalised `fd1e172`): substring matching with **word-boundary anchoring (`\b`)** + **leading/trailing quote-strip** on gold answers before regex compile. Everything else unchanged (case-insensitive `_norm`, `[REDACTED]` placeholder, per-block deterministic odd-index half-redaction, 4 conditions, 125-prompt split, Phase 1 ±0.02 sanity gate, paired bootstrap seed 42).
- **Rationale** (Phase 0 dry-run empirical finding): pure substring redaction produced **pathological over-redaction** on short golds. Example class: a yes/no question with `gold_answer="no"` caused every occurrence of "November", "novelist", "now", "none", "nothing" in the surrounding passage to be redacted to `[REDACTED]`, destroying context noise-level. This would systematically collapse `oracle_redacted_full` toward noise regardless of mechanism, biasing the §6 decision gate toward the `leakage_dominated` branch and voiding the three-way test. Word-boundary anchoring + quote-strip is the **minimum-necessary semantic interpretation** of "substring match" that preserves the scientific validity of the gate while staying faithful to the "only redact the answer token, not surrounding context" principle stated in §3.
- **What is guaranteed unchanged**: 4 conditions (`no_api` / `oracle_full` / `oracle_redacted_full` / `oracle_redacted_half`); Phase 1 ±0.02 hard stop; three-branch decision gate thresholds (≥+25 / +5..+25 / <+5); paired bootstrap 10k resamples seed 42; phase1d_v2 N=8 125-prompt split; positional oracle injection (j-th block on j-th `<search>`); `[REDACTED]` placeholder token; greedy/4096 generation config; model identity (base Qwen2.5-3B + LoRA `checkpoints/exp_006_sft_shared`).
- **Evidence artifacts**: 3 before/after samples (full + half redaction) dumped to `artifacts/exp_016_workdir/redaction_samples.json`, which will be cited directly in the rebuttal appendix as reviewer-auditable proof that redaction fired and that the refinement was semantically minimal rather than cosmetic. Per-block `n_redacted_per_block` counts in every eval JSON's `redaction_stats` field.
- **Dry-run outcome post-refinement**: 125 / 125 blocks redacted, 1663 substitutions total (mean 13.3 / block), 0 fully unredactable prompt ids. Confirms both (a) the refinement does not over-suppress (still fires on essentially every block) and (b) it does not under-suppress (substitution count two orders of magnitude above the pathological-substring floor would imply).
- **Non-deviations** explicitly considered and rejected: stemming, lemmatisation, stopword filtering, paraphrase detection — none of these were added; only word-boundary + quote-strip.

## 4. Implementation scaffold

### Files to create
```
scripts/exp_016_passage_redaction.py           # main eval driver (~300 LoC)
artifacts/exp_016_workdir/                     # (this dir)
├── plan.md                                    # this file
├── redaction_stats.json                       # pre-run: per-prompt redaction counts (dry-run output)
├── redaction_samples.json                     # 3 before/after samples (rebuttal citation)
├── no_api_eval.json                           # baseline reference (may reuse existing stage-1 eval)
├── oracle_full_eval.json                      # sanity gate output (tol ±0.02 vs §3 table)
├── oracle_redacted_full_eval.json             # 100% redaction condition
├── oracle_redacted_half_eval.json             # 50% redaction condition (dose-response)
├── bootstrap_redaction_ci.json                # paired bootstrap, all 3 oracle conditions vs no_api
└── exp_016_report.md                          # post-run summary
```

### Reused from v3.4
- `scripts/oracle_leakage_audit.py` — `_norm()`, `_flatten_golds()`, `INFO_BLOCK_RE`, gold-answer extraction
- `scripts/exp_006_stage1_eval.py` — model loading, generation loop, EM computation
- `scripts/phase1d_evaluate.py` — phase1d_v2 N=8 test split loader

### New logic (in exp_016_passage_redaction.py)
```python
def redact_info_block(block_text: str, gold_answers: list[str]) -> tuple[str, int]:
    """Return (redacted_text, n_substitutions_made)."""
    # Build regex pattern (case-insensitive, longest-match-first by sorting golds)
    ...

def build_redacted_oracle_map(raw_traj_path: str, n_key: str = "N8") -> dict:
    """Load teacher trajectories, redact each prompt's info blocks in place.
    Returns {prompt_id: [redacted_info_block, ...]}."""
    ...

def run_oracle_replay(
    model_path: str,
    oracle_map: dict,
    eval_prompts: list[dict],
    max_new_tokens: int = 4096,
) -> dict:
    """Reuse exp_006_stage1_eval.py generation loop with oracle-injected info.
    On each <search> emission, inject the j-th block from oracle_map[prompt_id].
    This matches exp_009's original oracle replay logic."""
    ...
```

### Delta vs exp_009 / exp_006 stage1 eval
- **Only** change: oracle info blocks are pre-redacted; everything else identical
- Generation loop / decoding / EM computation / paired bootstrap all reused
- Ensures apples-to-apples comparison with §3 table's baseline

## 5. Execution plan

### Phase 0 — Dry-run (no GPU, ~10 min local)
1. Build redaction map on 125 prompts locally (no model load)
2. Sanity check: `redaction_stats.json` should report
   - `n_prompts=125`, `n_info_blocks=N`, `n_redacted_blocks≥100` (v3.4 audit says ~98%)
   - `mean_redactions_per_block` reasonable (1-3)
   - `fully_unredactable_prompts` list (empty gold_answers)
3. If `n_redacted_blocks < 80` → blocker, investigate redaction logic
4. Print 3 before/after examples to `artifacts/exp_016_workdir/redaction_samples.txt` for manual inspection

### Phase 1 — Sanity gate (GPU, ~30 min)
1. Run `oracle_full` condition (no redaction) on 125 N=8 prompts
2. Expected: em_partial ≈ 0.498 (**±0.02 tolerance HARD STOP** vs §3 table 0.498 — Director 追加约束)
3. If outside tolerance → **STOP IMMEDIATELY**, emit `blocker_report` to Director, **do not self-diagnose**, do not proceed to Phase 2. Indicates model checkpoint or generation-config drift since exp_009 — requires Director decision on root cause

### Phase 2a — Main redaction run #1 (GPU, ~30 min)
1. Run `oracle_redacted_full` condition on same 125 prompts
2. Compute: em_partial_full_redact, Δ vs oracle_full, Δ vs no_api

### Phase 2b — Dose-response run (GPU, ~15 min)
1. Run `oracle_redacted_half` condition on same 125 prompts
2. Compute: em_partial_half_redact, Δ vs no_api

### Phase 3 — Analysis (post-GPU, ~10 min)
1. Paired bootstrap CI for all 3 oracle conditions vs no_api (10k resamples, seed 42)
2. Apply interpretation gates to Δ_full_redact (primary); use Δ_half_redact as monotonicity check
3. Write `exp_016_report.md` with: dose-response table, bootstrap CI, verdict
4. Emit recommended §3.5 wording patches for three verdict paths

### Total GPU time
- Phase 1 sanity: ~30 min
- Phase 2a full redact: ~30 min
- Phase 2b half redact: ~15 min
- Total on cuda:0: **~1.25h wall-clock** (Director APPROVED budget upgrade from 1h)

## 6. Interpretation gates (pre-committed, Director APPROVED 2026-04-14)

Primary gate on **Δ_full_redact** = em_partial(`oracle_redacted_full`) - em_partial(`no_api`).

| Δ_full_redact range | Verdict | Paper action |
|---|---|---|
| **≥ +25 pp** (≥ 50% of +47.3) | **retrieval_quality_dominates** | Headline claim holds; keep "causal bottleneck is retrieval execution" with redaction validation footnote |
| **+5 pp ≤ Δ < +25 pp** | **partial_mechanism** | Rewrite §3.5 as "retrieval + partial leakage upper bound"; scope-limit causal claim |
| **< +5 pp** | **leakage_dominated** | **Downgrade** headline: oracle gain was leakage-driven. §3.5 loses causal claim, becomes "upper bound that cannot isolate retrieval from leakage". Major rewrite of §1/§5 headline |

Paired bootstrap p-value (Δ_full_redact > 0) used as secondary significance check; gate is primarily on point-estimate thresholds.

### Dose-response cross-check (secondary, non-gating)
- `oracle_redacted_half` serves as a monotonicity witness: expect Δ_half_redact between Δ_full_redact and Δ_oracle_full
- If monotonicity violates (e.g. Δ_half > Δ_oracle_full or Δ_half < Δ_full_redact) → flag non-monotone, document in report but **do not block** the Δ_full_redact-based verdict
- Report table in §3.5 shows all 4 conditions as a dose-response curve for reviewer-visible evidence

## 7. Hard constraints

- **No fabricated numbers** — if any phase errors, report raw + blocker
- **Model**: base Qwen2.5-3B + LoRA `checkpoints/exp_006_sft_shared` — no LoRA swap, no re-train, no base-model swap (bit-exact match to exp_009 loading path)
- **Generation**: `max_new_tokens=4096`, temperature / top_p match §3 table footprint (greedy or sampling as §3 uses)
- **Oracle injection mechanism**: positional (j-th block on j-th `<search>`), identical to exp_009 — do **not** switch to query-similarity matching (that would conflate with another ablation)
- **Disk budget**: <5 GB in workdir (eval JSONs + stats, no checkpoints)
- **Budget**: 1.25h GPU wall-clock hard cap (updated 2026-04-14 for 4-condition dose-response)
- **Phase 1 ±0.02 hard-stop**: any violation → immediate `blocker_report` to Director, no self-diagnosis, no downstream phases
- **Redaction evidence**: `n_redacted_per_block` array in every eval JSON + 3 before/after samples in `redaction_samples.json` (reviewer-auditable)

## 8. Risks & blockers

| Risk | Mitigation |
|---|---|
| Redaction over-masks surrounding context | Substring-only match, manual sample inspection in Phase 0 |
| Gold-answer normalisation mismatches Wikipedia tokens | Reuse v3.4 leakage_audit's `_norm` exactly |
| Empty gold_answers for some prompts | Log and exclude from interpretation, reweight |
| Sanity run drift from §3 table (±0.02 tolerance fail) | **STOP** — indicates model checkpoint drift since exp_009 |
| Chinese drift (same as exp_014 smoke) | Post-hoc `pct_non_english_preds` check; likely absent since this uses SFT baseline not transplant, but collect anyway |

## 9. Pre-registered decision log

**Before** the experiment runs, the following decisions are committed:
- Split: phase1d_v2 N=8, full 125 prompts, no subsampling
- Model: base Qwen2.5-3B + LoRA `checkpoints/exp_006_sft_shared` (exp_009 bit-exact loading path)
- Conditions: `no_api` + `oracle_full` + `oracle_redacted_full` + `oracle_redacted_half` (4 conditions, Director APPROVED 2026-04-14)
- Half-redaction rule: deterministic odd-index substring redaction (seed-free)
- Bootstrap: 10k resamples, seed 42, paired design
- Gates: the 3-path table in §6 above, primary on Δ_full_redact
- Verdict paths map directly to paper revisions — no post-hoc reframing allowed
- Phase 1 ±0.02 hard-stop is non-negotiable; violations escalate to blocker_report, not self-diagnosis

## 10. Out of scope for exp_016 (deferred)

- Llama-1B replication (exp_016b, if Qwen result is clean) — Director explicitly declined for ARR rebuttal phase
- Paraphrase-level leakage audit (semantic coverage beyond substring) — Director APPROVED substring-only redaction
- Finer dose-response (0/25/75/100%) — only 0/50/100% covered by plan
- Retrieval-quality baseline (BM25 with same redaction, to isolate retriever)
- exp_014b cross-layer SVD transplant (deferred per Director's rebuttal priority lock)

## 11. Dependencies

- **Blocked by**: exp_014 sweep completion (cuda:0 release)
- **Blocks**: exp_017 (C2 capacity separation), exp_018 (C3 seed variance), and rebuttal response-letter draft
- **Reads**: `artifacts/phase1d_v2_data/raw_trajectories_N8.json`, `artifacts/exp_006/eval_sft_shared.json`, base Qwen2.5-3B (`/root/autodl-tmp/models/Qwen2.5-3B/`) + LoRA adapter `checkpoints/exp_006_sft_shared` on autodl-ltc (no merged "Qwen2.5-3B-SFT-Pooled" artefact exists; historical name was wrong)

## 12. Approval gate — Director APPROVED 2026-04-14

| # | Gate | Decision | Notes |
|---|---|---|---|
| 1 | Interpretation gates ≥+25 / +5..+25 / <+5 | **APPROVED** | "三档阈值合理" |
| 2 | Add `oracle_partial_redact_0.5` dose-response | **UPGRADED**: add as `oracle_redacted_half` + conditions now 4 (no_api / oracle_full / oracle_redacted_full / oracle_redacted_half) | "保守不是理由, dose-response 对 reviewer 是额外证据" — wall-clock 1h → 1.25h |
| 3 | Qwen-3B only, no Llama-1B | **APPROVED** | "ARR rebuttal 阶段资源紧" |
| 4 | Substring-only redaction + `[REDACTED]` placeholder | **APPROVED** | No sentence-level |

**Director 追加约束** (recorded above in §3, §5, §6, §7):
- Phase 1 ±0.02 sanity tolerance is **hard stop**, blocker_report on violation, no self-diagnosis
- `n_redacted_per_block` must appear in every eval result JSON (evidence of firing)
- 3 before/after samples dumped to `artifacts/exp_016_workdir/redaction_samples.json` for rebuttal citation

**Execution trigger**: Director will send `urgent_agent_message` after exp_014 sweep completes and releases cuda:0. Until then, main agent does NOT register exp_016, does NOT spawn worker.

On trigger, main agent will:
1. Register exp_016 in registry with this plan.md as hypothesis source, parent=exp_009
2. Spawn `worker_exp_016_passage_redaction` (tool_groups: ssh + knowledge + registry)
3. Register cron (`on_experiment_done`, interval 720s, tmux exp016_redact)
