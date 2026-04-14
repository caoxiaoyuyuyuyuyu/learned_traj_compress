# Handover — 2026-04-14

> 由 Agent 生成，每次状态变更后自动更新。硬限 150 行。

## 运行中
（无）

## 待处理
（无）

## 近期结论
- **exp_015** (failed): —
  [exp_015_phase0_sample_schema](/api/workspace-files/projects/learned_traj_compress/artifacts/exp_015/intermediate/phase0_sample_schema.txt) | [exp_015_phase0_dryrun_result](/api/workspace-files/projects/learned_traj_compress/artifacts/exp_015/report/phase0_dryrun_result.txt)
- **exp_016_passage_redaction** (negative): C1 blocking rebuttal: passage-redaction oracle replay to disentangle retrieval quality from answer leakage. Reviewer C1 noted that oracle SFT gain (+47.3pp em_partial on N=8) may be dominated by passage-level gold-answer leakage (v3.4 leakage audit: 98% substring hit rate). exp_016 runs 4 conditions on Qwen-3B SFT pooled (base + LoRA adapter checkpoints/exp_006_sft_shared) × 125-prompt phase1d_v2 N=8 test split: (1) no_api baseline, (2) oracle_full (reproduce §3 table, ±0.02 hard-stop sanity gate), (3) oracle_redacted_full (100% gold substring redaction in teacher info blocks), (4) oracle_redacted_half (50% deterministic odd-index redaction, dose-response control). Interpretation gates (pre-committed, Director APPROVED 2026-04-14): primary gate on Delta_full_redact = em_partial(oracle_redacted_full) - em_partial(no_api). ≥+25pp → retrieval_quality_dominates (headline holds); +5 to +25pp → partial_mechanism (scope-limit §3.5); <+5pp → leakage_dominated (downgrade §1/§3/§5 causal claim). Dose-response half-redact as secondary monotonicity witness. Director追加约束: Phase 1 ±0.02 hard stop blocker_report (no self-diagnose), n_redacted_per_block in eval JSON, 3 before/after samples in redaction_samples.json. Worker modified redaction spec to substring + word-boundary + quote-strip (vs plan.md pure substring) to avoid catastrophic over-redaction from short yes/no golds (e.g. "no" matching "November"); pending Director ACK. GPU budget 1.25h cuda:0 on RTX PRO 6000 Blackwell.
- **exp_015_kang2025_protocol_validation** (failed): C1 empirical differentiation + C10 protocol generality: 在 Kang 2025 (agent-distillation, NeurIPS 2025) 的公开 checkpoint 上跑 MemDistill 3-phase 诊断协议, 证明 silent failure mode 非 MEM1-only, 同时给 Kang 对照 empirical teeth. 方案 (b) 压缩 8 GPU-h: 1.5B+7B × {baseline, ftp} × {HotpotQA, GSM8K} = 16 run configs. Teacher 不 deploy (Kang 开源 trajectories). Phase A surface metric collapse curve + Phase C oracle retrieval replay from teacher_trajectories_2k. Outcome: 成功复制 silent failure → §3.8 独立第三方验证 + C1/C10 升 empirical; null (无 silent failure) → narrow C10 claim + 新 finding "ftp partially rescues"; hard-fail → 降级 WebFetch-only C1 差异化. Stop 条件: adapter >8h 无 sanity abort, Phase A 无 pattern 停 Phase C, GPU >20h 强制停. Parent: paper:v3.3 C1 review response. 无训练, 纯 inference + schema adapter. ← 基于 exp_013_w2_cross_backbone_llama1b

## 趋势
成功 15 / 否定 5 / 失败 20 / 共 40

## 下一步
（无活跃 idea，请通过 ideas.md 添加）

## 资源状态
autodl-ltc: 空闲 cuda:0

## 告警
- exp_002_phase1a_collapse_curve: 训练失败 — unknown
- exp_002v2_phase1a_collapse_curve: 训练失败 — unknown
- exp_005_phase1d_2x2_contrast: 训练失败 — unknown
- exp_006_eval_d029: 训练失败 — unknown
- exp_006_eval_d031: 训练失败 — unknown
- exp_006_eval_d031_fix_dpo_N8: 训练失败 — unknown
- exp_006_eval_d030_dpo: 训练失败 — unknown
- exp_006_eval_d031_fix_dpo_N4_v2: 训练失败 — unknown
- exp_006_eval_d031_fix_dpo_N4: 训练失败 — unknown
- exp_006_eval_d031_fix_dpo_N2: 训练失败 — unknown
- exp_006_eval_d031_fix_dpo_N2_v2: 训练失败 — unknown
- exp_006_eval_d031_fix_dpo_N8_v2: 训练失败 — unknown
- exp_008_search_suppression: 训练失败 — unknown
- exp_008_search_suppression_v2: 训练失败 — unknown
- exp_008_search_suppression_v3: 训练失败 — unknown
- exp_008_search_suppression_v4: 训练失败 — unknown
- exp_012_7b_pooled_sft: 训练失败 — unknown
- exp_014_svd_weight_transplant_7b_7b: 训练失败 — unknown
- exp_015_kang2025_protocol_validation: 训练失败 — unknown
- exp_015: 训练失败 — unknown
