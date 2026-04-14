# Response Letter — MemDistill v3.6

Generated: 2026-04-14
Paper version: v3.6 (post-exp_016 passage-redaction decomposition)
Verdict (exp_016 §6 gate, pre-registered): **partial_mechanism** (Δ_full_redact = +11.0pp,
within the +5..+25pp band).

---

## C1 — Oracle recovery conflates retrieval quality with passage-level answer leakage

**Reviewer concern (paraphrased).** The v3.5 paper's headline +47.7pp oracle gain is
uninterpretable as a retrieval-quality upper bound because teacher-replayed
`<information>` blocks contain gold-answer substrings at 98% rate (v3.4 leakage audit).
The reviewer requested a causal disentanglement before accepting the §3.5 ``retrieval is
the bottleneck'' framing.

**Response.** We partially accept this concern and have restructured the relevant claims
in §1, §3.2, §3.5, §5, and Appendix D (redaction ablation) to reflect a new,
pre-registered experiment (exp_016) that directly disentangles the two mechanisms via
passage redaction.

### 1. Experiment (exp_016, pre-registered, Director-approved 2026-04-14)

We evaluated the SFT student (base Qwen2.5-3B + LoRA `checkpoints/exp_006_sft_shared`,
bit-exact to the exp_009 loading path) on the full phase1d_v2 N=8 test split (125
prompts), with four conditions:

- `honest_search` — baseline (no oracle injection)
- `oracle_full` — reproduce exp_009 (sanity gate, ±0.02 tolerance vs §3 table 0.498)
- `oracle_redacted_full` — redact 100% of gold-answer substring matches in each
  `<information>` block with a `[REDACTED]` placeholder
- `oracle_redacted_half` — uniform density halving applied to all 125 oracle
  passages (867 `[REDACTED]` substitutions vs 1663 in `oracle_redacted_full`,
  ratio 0.52 via preprocessing-level `redact_info_block_tag`; deterministic
  dose-response control)

### 2. Results

| Condition | em_partial (N=8) | Δ vs honest_search | 95% CI (pp) |
|---|---|---|---|
| honest_search | 0.021 | — | — |
| oracle_redacted_full | 0.131 | **+11.0** | [+8.8, +13.2] |
| oracle_redacted_half | 0.373 | +35.2 | [+32.2, +38.2] |
| oracle_full | 0.498 | +47.7 | [+44.6, +50.9] |

Paired bootstrap, 10k resamples, seed 42, n=125. Sanity gate: oracle_full 0.498 vs §3
table 0.498 (PASS, within ±0.02). Monotonicity 0.131 < 0.373 < 0.498 holds. All three
oracle conditions are strictly positive at p<0.0001 vs honest_search.

### 3. Decomposition

- **Retrieval-quality component (non-leakage)**: +11.0pp (≈23% of full oracle gain).
  This is the `oracle_redacted_full` − `honest_search` residual, i.e. what remains after
  removing all exact-match gold-answer substrings from teacher passages.
- **Passage-level answer-leakage component**: +36.7pp [+33.5, +40.0] (≈77%; paired
  bootstrap 10k resamples, seed 42, `oracle_full − oracle_redacted_full`, p<0.0001;
  source `exp_016_workdir/bootstrap_redaction_ci.json`
  field `oracle_full_vs_oracle_redacted_full_delta`). This is the gap between
  `oracle_full` and `oracle_redacted_full`, i.e. the portion of the oracle gain
  attributable to substring-level answer exposure enabling short-form decoding.
- **Span-importance asymmetry (TL;DR)**: halving per-passage substitution density
  (applied uniformly across all 125 passages, 867 vs 1663 total substitutions,
  ratio 0.52) retains 73.8% of the oracle gain, indicating a Pareto-like
  distribution of span importance rather than the uniform-leakage model a
  reviewer might assume.

### 4. Paper changes

- **§1 Introduction**: the ``oracle retrieval-quality upper bound'' narrative is
  downgraded to ``leakage-inclusive upper bound'', with the 11/37 pp decomposition
  quoted in-line. The ``catastrophic SFT collapse'' headline (`honest_search = 0.021` floor)
  is **unchanged** — the decomposition does not affect the honest_search floor, which is what
  defines the collapse.
- **§3.2 Capacity ceiling**: the retrieval-quality upper bound is re-anchored to
  `oracle_redacted_full − honest_search = 0.110` rather than the leakage-inclusive 0.477. The
  12–14% capacity bound is now compared against this tighter denominator, and is no
  longer a trivial fraction.
- **§3.5 Causal intervention**: rewritten as ``retrieval is necessary but not
  sufficient''. The teacher–student gap now has two distinct sub-failures — (a)
  retrieval precision, and (b) student-side ability to exploit passages without
  substring hints — corresponding to the 11pp and 37pp components respectively.
- **§5 Findings**: the oracle-related finding is restated with explicit disclosure of
  the redaction ablation and the decomposition.
- **Figure 5**: new dose-response subplot (`figure5_dose_response.pdf`), x = redaction
  level (0/50/100%), y = em_partial, error bars = paired-bootstrap 95% CI.
- **Appendix D (new)**: Redaction Validation — protocol, pre-registered deviation log,
  fire-statistics (125/125 blocks redacted, 1663 total substitutions, mean 13.3/block,
  zero fully-unredactable prompts), 3 before/after samples, bootstrap CI decomposition
  table, explicit limitations (substring-only, not paraphrase-aware).

### 5. Pre-registered deviation disclosure (word-boundary anchoring)

We disclose in full, without spin, a pre-registered protocol deviation from the
Director-approved spec:

- **Registered**: ``substring-only redaction with `_norm` normalisation,
  case-insensitive, replace each gold-answer substring with `[REDACTED]`''.
- **Executed**: same spec **plus word-boundary anchoring (`\b`) and leading/trailing
  quote-strip** on gold answers before regex compile.
- **Why**: Phase 0 dry-run found pure substring matching caused pathological
  over-redaction on short gold answers. A yes/no question with `gold_answer="no"`
  redacted every occurrence of ``November'', ``novelist'', ``now'', ``none'',
  ``nothing'' in the passage, collapsing `oracle_redacted_full` toward the honest_search floor
  regardless of the true mechanism, biasing the decision gate toward the
  `leakage_dominated` branch and voiding the three-way pre-registered test.
- **Minimum-necessary refinement**: word-boundary + quote-strip is the minimal
  semantic interpretation of ``substring match'' that preserves the test's scientific
  validity. We explicitly considered and rejected stemming, lemmatisation, stopword
  filtering, and paraphrase detection. See Appendix D.1 for the full deviation log and
  Director-approved non-deviations list.
- **Guaranteed unchanged**: 4 conditions, ±0.02 Phase 1 hard-stop, three-branch
  decision gate thresholds, bootstrap config, split, model identity, oracle injection
  mechanism, placeholder token, generation config.

### 6. What the decomposition does and does not say

- The ``catastrophic SFT collapse'' headline in §1 **stands**: `honest_search em_partial =
  0.021` at N=8 is unchanged by this experiment, and this is what defines the silent
  failure mode. The decomposition refines *how* oracle recovers this floor, not
  *whether* the student collapses without retrieval.
- Retrieval execution is **necessary**: +11.0pp after full redaction is statistically
  and practically significant, and is larger than both the DPO effect (+1.0pp,
  p=0.163) and the 7B-vs-3B capacity-isolation effect at N=8.
- Retrieval execution is **not sufficient**: the majority (77%) of the oracle gain
  depends on passage-level answer substrings, which realistic retrieval backends will
  not reliably provide. BM25, the deployable mitigation, recovers +12.2pp at N=8 —
  consistent with the retrieval-quality component (≈11pp) being the practical ceiling,
  not the leakage-inclusive 47pp.
- The $+11$pp retrieval-quality residual is **itself an upper bound** (substring
  redaction removes exact-match but not paraphrase hints). The true non-leakage
  retrieval-quality mechanism is ≤ +11pp.

### 7. Why we did not adopt the minimal §3.5-only scope

We explicitly considered and rejected a ``patch §3.5 only'' scope because (a) the
leakage-inclusive framing also appears in the abstract, §1 contributions,
§3.2 capacity bound, §5 findings, and the Figure 5 caption, and leaving any of these
unchanged would attract a subsequent round of criticism for inconsistency; (b) the
11/37 decomposition is substantively important enough (it shifts the
necessary-vs-sufficient status of retrieval) to warrant headline-level disclosure.
The broad-scope revision is what §3.5-only would ultimately be forced to become under
reviewer follow-up, minus the extra round-trip.

### 8. Methodology note — latency disparity between honest_search and oracle_*

The `honest_search` condition runs ~4.72× slower than `oracle_full` (3666.7s vs 777.4s over
125 prompts, 2889.3s delta). This latency disparity is **not** attributable to output length:
average generation length is 325.7 word-tokens for `honest_search` vs 315.4 for `oracle_full`,
a <4% delta (exp_016_workdir/no_api_latency_probe.json). The latency is fully accounted for
by FlashRAG retrieval I/O overhead (~23s/prompt), which the `oracle_*` conditions bypass
entirely by injecting teacher-replayed passages directly. `avg_searches = 1` and
`truncated_rate = 0` for both conditions confirm results are trustworthy — the `honest_search`
samples are not degenerate or pathologically-truncated fallbacks. This disclosure does not
make a root-cause claim about internal retrieval timing variance (TGI warm-cache differences,
retry timeout distribution), only that output-length confounding is excluded.
