# MemDistill 研究路线图

> 最后更新: 2026-04-14 07:13 UTC | 更新原因: Reviewer warning 吸收: exp_017 student_no_api 语义锁定 = 3B LoRA r32 no_api (解读 2 封闭占比); Strongly Expected 列表顺序对齐 GPU 调度; housekeeping 明确化 (D060/D061 补录 + exp_019 plan.md 依赖更新 前置 exp_019 spawn)

## 研究目标

诊断 agent memory consolidation 中 SFT-RL gap 机制, 蒸馏 7B RL memory policy 到 <3B sidecar.

## ⚠️ W3 结构性发现 (D060)

Paper 原 "94.5% drop from teacher's 0.455" 实际上是 **oracle-teacher vs honest-student** 的 apples-to-oranges 比较:
- Teacher 0.455 来自 exp_002v3: 50 samples + gold-passage injection
- Student 0.025 来自 phase1d_v2: 125 samples + real search API

exp_019 必须同时测两个 teacher conditions 产出 matched 数字.

## 核心发现链

1. EM 衰减 0.62→0.341
2. RL 对齐 base 主方向 2-6x (observational)
3. Per-objective 无线性 signal
4. DPO +1.0pp scale-limited
5. Oracle +43-47pp upper bound; exp_016 redaction 隔离 non-leakage
6. Capacity ≤12-14% confound; exp_017 拆解
7. BM25 +6-12pp dose-response
8. ASA=0.606 3-token prefix artifact

## 当前阶段

**📝 ARR Rebuttal Preparation** (41 天)

v3.5 bundle + 4 diagnostic scripts pushed (b5db091). exp_016 RUNNING. 剩余 blocking: C2 exp_017, C3 exp_018. Strongly expected: W3 exp_019.

## GPU 调度优先级 (D054 reorder)

```
exp_016 (RUNNING) > exp_019 > exp_017 > exp_018
```

**理由**: exp_017 gate 阈值依赖 T_honest 分母, exp_019 必须先跑锁定 baseline. exp_017 gate 改为**相对贡献率** (≥25% / 5-25% / <5%).

## Pre-registered Decision Trees

### exp_016 (C1) — RUNNING
Δ_full_redact: ≥+25pp 保留 / +5-+25pp 双机制 / <+5pp **崩塌重写**

**Deviation** (Phase 0 dry-run 发现): substring-only 对短 yes/no gold pathological over-redaction (e.g. "no" 匹配 "November"), 改 word-boundary + quote-strip (commit 7cf099d). Reviewer 独立审查判定科学上更正确, 保持 gate 三分支有效性. plan.md §3 有 Deviation log, response letter 主动披露.

### exp_019 (W3) — P1, 2 conditions

- `teacher_matched_no_api_em_partial_n8` (real search API, honest) — 修 headline 94.5% drop
- `teacher_matched_oracle_em_partial_n8` (oracle injection) — 修 teacher upper bound

**3-branch decision tree** (matched_honest):
- ≥0.40: 局部数字替换
- 0.25-0.40: framing 微调 "retention ~6-10%"
- <0.25: §1 结构重写 "both teacher and student struggle at N=8"

**Cross-experiment coupling**: Figure 5 分母重算 / exp_017 gate rescale / §3.causal ">100%" claim 升级 / student>teacher 预防性叙事.

### exp_017 (C2) — P2, gate depends on exp_019
relative_capacity_contribution = Δ_cap / (T_honest − student_3B_LoRA_r32_no_api):
- ≥25% training-method 贡献 / 5-25% 中间 / <5% tighten. Fallback 7B LoRA rank-32.

**student_no_api 语义锁定**: = `phase1d_v2 N=8 student_3B_LoRA_r32_no_api_em_partial` (解读 2, capacity 拆分封闭占比, 与 exp_017 LoRA vs full 设计一致).

### exp_018 (C3) — P3
3B SFT × 3 seeds × (no_api, oracle). SD(Δ_oracle): ≤2pp robust / 2-5pp mild / >5pp caveat.

## 4 Strongly Expected (按执行顺序)

1. W3 teacher matched → exp_019 (P1, 2 conditions, 3-branch tree)
2. C2 capacity × method → exp_017 (P2, 依赖 exp_019)
3. C3 seed variance → exp_018 (P3)
4. W1/W2 downgrade + W6 DPO softening → v3.5d/e ✓

## 短期任务

- [x] v3.5a-e 修订 + push (b5db091)
- [x] exp_015 FAILED_INFEASIBLE
- [x] exp_014 DONE/failed (autodl on_failed hook, partial snapshot 保留; 与 registry 对齐 2026-04-14)
- [x] exp_016 scaffold + decision tree + Deviation log
- [x] exp_017 decision tree + student_no_api 语义锁定
- [x] exp_018 scope + decision tree
- [x] exp_019 plan.md v2 (146 行)
- [x] D054: exp_019 提前至 P1
- [ ] **exp_016** RUNNING ~07:50Z 预计完成, 等 cron on_experiment_done
- [ ] exp_016 done 后 housekeeping (一次性完成):
  - [ ] 4 stash audit + archive + drop (worker A)
  - [ ] server `git pull --rebase` 拉 b5db091 (worker B, tmux 空闲窗口)
  - [ ] registry exp_015 orphan purge (PREPARING 条目)
  - [ ] exp_019 plan.md §3 line 3 + §12 line 134 "Awaits exp_014" → "Awaits exp_016 cuda:0 release"
  - [ ] exp_014 口径统一 (registry DONE/failed vs roadmap "terminated", 对齐为 DONE with partial scope-narrow)
  - [ ] decisions.yaml 补录 D060 (W3 apples-to-oranges) + D061 (exp_019 spec 扩充)
- [ ] **exp_019** W3, 2-3h GPU, P1 (exp_016 cuda:0 释放后)
- [ ] **exp_017** C2, 2.5h GPU, P2 (依赖 exp_019 T_honest)
- [ ] **exp_018** C3, 4-5h GPU, P3
- [ ] Rebuttal response letter + v3.6 package

## 累计 GPU

~100h + rebuttal ~12 GPU-h

## 目标 venue

EMNLP 2026 Findings (conditional) | ARR 截止 2026-05-25 (41 天)

## 关键决策日志

| ID | 日期 | 摘要 | 关联实验 |
|----|------|------|----------|
| D055 | 2026-04-14 | D055: exp_016 verdict = partial_mechanism (leakage | exp_016_passage_redaction |
| D054 | 2026-04-14 | exp_019 提前至 P1, GPU 顺序改为 exp_016 > exp_019 > exp_0 | exp_016_passage_redaction, exp_019_teacher_matched_replay, exp_017_capacity_separation, exp_018_seed_variance |
| D053 | 2026-04-14 | 接受 v3.1 (commit f9107f3) 为 ACL 2026 投稿版，正文 7pp 满足  | exp_013 |
| D052 | 2026-04-13 | D052: 论文叙事重构——从实验报告流改为 thesis-driven，新 thesis 聚焦 a | — |
| D051 | 2026-04-12 | D051: 审稿终稿 Accept (3.75/5)，进入 camera-ready editori | exp_012_7b_pooled_sft_v2, exp_011_bm25_retrieval, exp_009_student_with_api |
| D050 | 2026-04-12 | D044 addendum: D044 结论 "7B≈3B capacity confound 不成 | exp_010_7b_sft_eval, exp_012_7b_pooled_sft_v2 |
| D049 | 2026-04-12 | D049: exp_012 修正 capacity 叙事——capacity 是 contribut | exp_012_7b_pooled_sft_v2, exp_010_7b_sft_eval, exp_009_student_with_api |
| D048 | 2026-04-12 | Strategy analysis 修正：teacher 本身 >95% 单次搜索，student  | exp_009_student_with_api |
| D047 | 2026-04-12 | D047: 用户授权 8h GPU 提升论文水平。分配：BM25 retrieval (~2h) + | exp_009_student_with_api, exp_010_7b_sft_eval |
| D046 | 2026-04-12 | Round 3 weak accept (3.25/5)，进入 camera-ready 修改。不训 | exp_009_student_with_api, exp_010_7b_sft_eval |

