# MemDistill 研究路线图

> 最后更新: 2026-04-14 11:25 UTC | 更新原因: W1/W2 修正: D062 已补齐 + D058/D059 gap 调查; GPU 恢复硬截止护栏 (T-20 / 24h auto-dispatch)

## 研究目标

诊断 agent memory consolidation 中 SFT-RL gap 机制, 蒸馏 7B RL memory policy 到 <3B sidecar.

## ⚠️ W3 结构性发现 (D060)

Paper 原 "94.5% drop from teacher's 0.455" 是 oracle-teacher vs honest-student apples-to-oranges. v3.6 paper 已改为 +47.7pp 口径, 不再出现 94.5%. matched condition 数字待 exp_019 补齐.

## 核心发现链

1. EM 衰减 0.62→0.341
2. RL 对齐 base 主方向 2-6x
3. Per-objective 无线性 signal
4. DPO +1.0pp scale-limited
5. Oracle +47.7pp upper bound; **exp_016 negative**: retrieval 23% / leakage 77% (partial_mechanism verdict)
6. Capacity ≤12-14% confound; exp_017 拆解 (deferred)
7. BM25 +6-12pp dose-response
8. ASA=0.606 3-token prefix artifact

## 当前阶段

**📝 ARR Rebuttal Preparation — v3.6 FROZEN @ a85410e** (D062, T-41 天)

Paper v3.6 commit **a85410e** on origin/main. Reviewer 第 3 轮 critical gate PASS. Issue 1/2/3 + P0 关闭; 禁区 0 regression; 算术闭合 (0.110+0.367=0.477).

**exp_016 DONE negative** — C1 verdict: partial_mechanism. Paper §3.5 + Abstract/§1/§5 broad rewrite 反映 headline 降级.

**用户暂停 GPU**: 2026-04-14 ~10:00 UTC 释放 AutoDL. exp_019/017/018 deferred.

## GPU 调度 + 硬截止护栏 (W2 修正)

优先级 (GPU 恢复后): `exp_019 (P1) > exp_017 (P2) > exp_018 (P3)`

**硬截止**:
- **T-20 (2026-05-05)** 前若 GPU 未恢复 → Director 必须重新评估 (a) 主动请用户重启 AutoDL 或 (b) 将 W3 / C2 / C3 降级为 paper future work, rebuttal 中明确承认 limitation
- **GPU 恢复后 24h 内** auto-dispatch exp_019 (P1), 不再等 Director 审批

预算: exp_019 ~4h + exp_017 ~8h + exp_018 ~16h = ~28 GPU-h (+ analysis + rebuttal 集成).

## Pre-registered Decision Trees

### exp_016 (C1) — **DONE negative**

Verdict: partial_mechanism. Δ_full_redact +11.0pp (CI [+8.8,+13.2]) 落入 +5~+25pp 区间. Decomposition retrieval 0.110 / leakage 0.367 / total 0.477. Half-redact 保留 73.8% oracle gain. Residual probe: 0/125 blocks 残留 gold. Deviation 3 (word-boundary + quote-strip) 已披露.

### exp_019 (W3) — deferred, auto-dispatch on GPU return

2 conditions: `teacher_matched_no_api_em_partial_n8` + `teacher_matched_oracle_em_partial_n8`. 3-branch tree on matched_honest (≥0.40 / 0.25-0.40 / <0.25).

### exp_017 (C2) — deferred

relative_capacity_contribution gate (≥25% / 5-25% / <5%). student_no_api = phase1d_v2 N=8 student_3B_LoRA_r32_no_api_em_partial.

### exp_018 (C3) — deferred

3B SFT × 3 seeds × (no_api, oracle, **oracle_redacted_full per D061**). SD(Δ_oracle) gate.

## 短期任务

- [x] v3.5a-e + exp_014/015 清算 + exp_016 scaffold/dry-run/execution/analysis
- [x] exp_016 DONE negative (partial_mechanism)
- [x] Paper v3.6 broad rewrite (sed 1-5) + 3 轮 Reviewer critical gate
- [x] v3.6 freeze commit a85410e + push origin/main
- [x] decisions.yaml D055/D056/D057/D060/D061/D062 已记录 (W1 修正: 原任务说 "D060/D061 补录" 状态过时; D062 今补齐)
- [ ] decisions.yaml D058/D059 编号 gap 调查 (预留 or draft persist 失败)
- [ ] **submit_paper_to_review** → Review Director (EMNLP 2026 Findings, ARR 截止 2026-05-25)
- [ ] paper_provenance.yaml v3.6 sed 链深度更新
- [ ] `_HASH.tex` 6 个 stale 快照清理 (W2 post-freeze polish)
- [ ] tab:eval_matrix cross-ref 脚注 (0.025/0.021, post-freeze polish)
- [ ] report.md 20 个 "训练失败 — unknown" 历史告警过滤 (I5)
- [ ] **GPU 恢复后 24h 内**: auto-dispatch exp_019 → exp_017 → exp_018
- [ ] **T-20 (2026-05-05) 前**: 若 GPU 未恢复, 评估 pivot 到 future-work 降级
- [ ] AutoDL MCP `autodl_list_instances` 修复 (coroutine 未 await)

## 累计 GPU

~100h baseline + ~14h rebuttal (exp_016 含); 预留 ~28h for exp_019/017/018

## 目标 venue

EMNLP 2026 Findings (conditional) | ARR 截止 2026-05-25 (T-41)

## 关键决策日志

| ID | 日期 | 摘要 | 关联实验 |
|----|------|------|----------|
| D063 | 2026-04-14 | D063: paper v3.8 FROZEN @ commit cd11cae, ARR subm | exp_016_passage_redaction, exp_015_kang2025_protocol_validation |
| D062 | 2026-04-14 | D062: paper v3.6 FREEZE @ commit a85410e (origin/m | exp_016_passage_redaction |
| D061 | 2026-04-14 | D061: exp_018 per-seed 扩充 oracle_redacted_full 条件  | exp_018_seed_variance, exp_017_capacity_separation |
| D060 | 2026-04-14 | D060: exp_019 teacher matched replay 必须同时提供 honest | exp_019_teacher_matched_replay, exp_002v3_phase1a_collapse_curve |
| D057 | 2026-04-14 | D057: v3.6 paper revision scope = BROAD (Abstract  | exp_016_passage_redaction |
| D056 | 2026-04-14 | D056: accept exp_016 redaction deviation (substrin | exp_016_passage_redaction |
| D055 | 2026-04-14 | D055: exp_016 verdict = partial_mechanism (leakage | exp_016_passage_redaction |
| D054 | 2026-04-14 | exp_019 提前至 P1, GPU 顺序改为 exp_016 > exp_019 > exp_0 | exp_016_passage_redaction, exp_019_teacher_matched_replay, exp_017_capacity_separation, exp_018_seed_variance |
| D053 | 2026-04-14 | 接受 v3.1 (commit f9107f3) 为 ACL 2026 投稿版，正文 7pp 满足  | exp_013 |
| D052 | 2026-04-13 | D052: 论文叙事重构——从实验报告流改为 thesis-driven，新 thesis 聚焦 a | — |

