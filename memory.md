## [2026-04-14 12:05 UTC] v3.7 rebuttal: 两处 Director/任务模板事实修正

v3.7 修订过程中 worker 纠正了两条之前的错误假设（来自 Director 指令模板或我的任务 prompt）:

### Correction 1 — exp_015 不是 half-density run
之前任务模板假设 exp_015 是一次失败的 half-density 实验。**事实**: exp_015 是 Kang 2025 CodeAct vs ReAct 协议族交叉工作验证，在 Phase 0 因 **CodeAct vs ReAct protocol-family mismatch** 被终止。与半密度无关。v3.7 §1 的 exp_015 披露已按真实事实写入，并作为"论文scope 限定在 ReAct 家族"的依据。

**How to apply**: 未来提及 exp_015 必须说 "Kang 2025 cross-work validation, CodeAct/ReAct protocol-family mismatch, Phase 0 aborted"，不要复制旧任务模板里 "failed half-density run" 的错误叙述。

### Correction 2 — oracle_redacted_full 的 retrieval residual 并不 beat BM25
之前 v3.7 task prompt 里我写的 W4 方向 "oracle_redacted_full's retrieval-only residual still beats BM25" **是错的**。实测: N=8 下 oracle_redacted_full = 0.131 < BM25 = 0.147。worker 没有写这个错误 claim，改为把 BM25 定位为"现实 baseline"，oracle 定位为"upper-bound probe"，不做两者的高低对比。

**How to apply**: 描述 retrieval residual 时,不要与 BM25 做"高于"比较。 +11.0pp retrieval residual 是 **相对 honest_search baseline** 的增益,不是相对 BM25。partial_mechanism 23/77 分解 仍然 稳健 (oracle_full 相对 honest_search 的 +47.7pp 分解),但 oracle_redacted_full 的绝对得分低于 BM25 这一事实需要在未来引用时注意。

---

## [2026-04-14 10:00 UTC] phase1d_v2 test set = per-N canonical partitions

phase1d_v2 不是单一 n=125，是 per-N canonical partitions:
- N=2 → 251 prompts
- N=4 → 131 prompts
- N=8 → 125 prompts

exp_006 N=8 (0.025) vs exp_016 N=8 (0.021) = cross-run decoding noise on identical 125 prompts (byte-identical gold_answers verified)，落在 ±0.02 pre-registered tolerance。

**未来引用 phase1d_v2 size 时必须按 N 明示 n_prompts，不能说 "n=125" 代指整个 test set。证据: `artifacts/_project/paper/table3_split_consistency_verification.json`**

---

## [2026-04-14 09:25 UTC] exp_016 residual leakage probe → P0 WARNING CLEARED (<5% 分支)

**结果**:
- `oracle_redacted_full` blocks-only residual = **0/125 = 0.0%** → v3.6 freeze 可继续
- Sanity `oracle_full` blocks-only = 125/125 = 1.00 (匹配 v3.4 audit 98% 基线) ✅
- 诊断分支: **<5%** → 加 footnote 后继续 Reviewer 第 2 轮对抗审查

**Root cause of 753-vs-1663 gap（非 prompt 截断）**:
- `eval_sft_with_api.py:397` 存 `generated_text[:2000]` — 纯 **JSON 磁盘序列化截断**
- `generate_with_retrieval` 本身 `tokenizer(current_text, return_tensors="pt")` 无 `max_length`/`truncation` → **模型看到全部 1663 `[REDACTED]` tokens**
- Worker A 的 753 来自 `json.dumps(details[i]).count("[REDACTED]")` 跨被截断的 `generated_text[:2000]` + `predicted_answer` + `predicted_parts`
- **结论**: 差异是序列化产物，不是 model-input 丢失 → partial_mechanism (11.0pp / 36.7pp) 分解稳健

**60% user_q residual 为什么不是失败**:
- 104/106 gold matches 出现在 "which, X or Y?" 多跳问题主干里（任务构造决定 answer 必出现在 question）
- 剩余 2/106 是常见词假阳性（"American"→"american folklore", "rock"→"rock band"）
- user_q 在 oracle_full / oracle_redacted_full / honest_search 三条件下完全对称 → 条件 delta 里 cancel

**Why**: Reviewer v5 对抗审查发现 Phase 0 dry-run 1663 vs eval details 753 的 55% 差异，若原因是 prompt 截断会让 oracle_redacted_full 保留未遮蔽 gold substring，推翻 +11.0pp retrieval residual。probe worker 用 blocks-only regex 在真实 eval JSON 上 grep gold_answer 验证：0/125 命中 → hypothesis 证伪，警报解除。根因归到纯磁盘序列化 artifact，不影响 model input。

**How to apply (paper v3.6)**:
- §3.5 partial_mechanism 叙事**不降级**，`23% / 77%` 百分比**冻结** (11.0/47.7 = 23.06%, 36.7/47.7 = 76.94%)
- §3.5 或 Appendix 加 footnote: "Phase-0 dry-run counts (1663 `[REDACTED]`) and eval JSON details counts (753) differ due to `generated_text[:2000]` JSON-serialization truncation in `eval_sft_with_api.py:397`; tokenizer calls in `generate_with_retrieval` have no `max_length` cap, so model input contains the full redacted context. Probe verified 0/125 blocks-only residual in `oracle_redacted_full`; see Appendix~\ref{app:redaction} probe report."
- sed 3 scope 追加此 footnote（在 v6 Issue 3 Table 3 caption + app:exp006_full_test stub 之后）
- Reviewer 第 2 轮对抗审查可以启动（等 sed 2 done 后）

**提醒**: 这次 Reviewer 的 P0 hypothesis 是合理怀疑但最终被证伪——validation 流程需保留但结论积极。下次见到跨阶段计数差异时，先检查 serialization boundary（`[:N]` / JSON dumps limits / logging truncation），再怀疑 model input。

---

## [2026-04-14 09:10 UTC] exp_016 redaction apples-to-apples 终审结论

### 发现 1（**已证实**）: half 条件 = per-passage density ~50%（实测 0.43），跨所有 125 passages
- **数据**（`redaction_stats_apples_to_apples.json`，同 regex 同 counting surface）：
  - `full_eval_total_redacted = 753`, avg `6.02/passage`, 125/125 passages touched
  - `half_eval_total_redacted = 325` (✅ sanity 匹配 D 基准), avg `2.60/passage`, 125/125 passages touched
  - `ratio_half_over_full = 0.4316`
- **三条路被排除/证实**：
  - ❌ "half 是 first-N subset"（50% passages 做完整 redaction）**被证伪**：full 和 half 都全 125 touched
  - ❌ "half 是 density-half ~20%"（Reviewer 假设 + 我早先的误读）**被证伪**：实测比例是 0.43 不是 0.20
  - ✅ "half 是 per-passage density ~50%"（density-halved across all passages）**被证实**
- **Plan.md 原文 "odd-indexed 50% subset deterministically" 解读**：数值上 ~50% density 是对的，但 "odd-indexed subset" 这个机制描述错——**不是 subset，是每个 passage 里删掉约半数的 substitutions**
- **Phase 0 dry-run 1663 vs eval details 753 差异**: [2026-04-14 09:25 probe 结论] 纯 JSON 序列化截断 (`generated_text[:2000]` @ `eval_sft_with_api.py:397`)，model input 完整，0/125 residual。非 input-level bug。

### 发现 2（**保留**）: no_api 4.72x 延迟 ≠ 更长 fallback（Reviewer 假设证伪）
- no_api avg 325.7 word-tokens vs oracle_full 315.4 word-tokens，<4% 差异
- Reviewer 新 insight（v4）：延迟 2889.3s 完美对齐 "FlashRAG retrieval I/O overhead ~23s/prompt"
- **命名含义**: `no_api` 是 misnomer，真实是 honest FlashRAG retrieval baseline。v3.6 全局重命名 → `honest_search`

**Why**: 三项均来自 worker_postprocess + worker_verify_apples_to_apples 双重 CPU-only 二次审计。前两次都因为**跨计数面数字直接比较**得出错误结论（325 vs 1663 → 19.5%），只有用同一 counting surface 比较（753 vs 325 → 43%）才是正确的。**提醒**：rebuttal / paper 数字的每次跨源/跨文件比较前，必须先验证两边 counting method 是否一致。这条经验值得写进 ml-experiment-guide.skill.md

### How to apply（paper v3.6 下游）
- **§3.5 L74** 原 "50% (oracle_redacted_half, deterministic odd-indexed substring redaction)"：
  - 改为 "\sim 50\% per-passage density (\texttt{oracle\_redacted\_half}, uniform density-halved substring redaction across all 125 passages: mean 2.60 substitutions/passage vs 6.02 in the full condition, ratio $\approx 0.43$; see Appendix~\ref{app:redaction})"
- **Figure 5 x-axis**: "50%" 作为 oracle_redacted_half 的 label **保留**（per-passage density ~50% 的意思对），但 caption 必须把 "50% = per-passage density reduction, all passages touched" 说清，避免 reviewer 误读为 "50% passages"
- **Insight β narrative**: 原 Reviewer 版本 "~20% density recovers 73.8%" 改为 "~50% per-passage density reduction retains 73.8% of oracle gain → the first half of substitutions per passage contributes ~74% of the leakage mechanism, the second half only ~26%, consistent with Pareto-like span importance but not as extreme as originally hypothesized"
- **γ 命名决议建议**: 保留 `oracle_redacted_half` 命名（per-passage density-halved 的含义符合 "half" 字面）；或改 `oracle_redacted_partial` 更精确但丢失数值含义。我建议**保留 half 命名 + appendix 显式定义**
- **Deviation log 必须**补充："both oracle_redacted_full and oracle_redacted_half redact across all 125/125 passages; half reduces per-passage substitution count to ~43% (6.02 → 2.60), not a subset of passages"

---

## [2026-04-14 08:52 UTC] exp_016 postprocess non-obvious 发现（已被 09:10 终审取代，仅保留作为自检教训记录）

### 发现 1（**撤回**）: 基于 apples-to-oranges 的"~20% density-half"解读
- D worker 325-subs 数字来自 `oracle_redacted_half_eval.json` details 字段的 regex 事后计数
- 对比用的 1663 来自 Phase 0 `redaction_stats.json` 的 dry-run 统计
- 两者计数面不同（eval 输出 vs 预处理统计），apples-to-oranges
- 09:10 终审用同一 counting surface 重数 full，得到 753（不是 1663），ratio 0.43（不是 0.20）
- **教训**：跨计数面数字比较必须先对齐 counting surface 再下结论

### 发现 2: no_api 4.72x 延迟 **不是** 来自更长 fallback 生成（Reviewer 假设证伪）
- no_api avg 325.7 word-tokens vs oracle_full 315.4 word-tokens（差 <4%）
- Reviewer 假设的 "2500 vs 500 tokens" 是错的。延迟差异源于其他未知因素（候选：FlashRAG 冷启动 / retry timeout / TGI warm-cache 差异，未调查）
- avg_searches=1, truncated_rate=0, em_partial 分布正常 → 结果仍 trustworthy
- **How to apply**: response letter / rebuttal 只做 disclosure 不做 root-cause claim。不要写 "因为生成更长 fallback 所以延迟更高"

### 新 CI: Δ_leakage paired bootstrap
- oracle_full − oracle_redacted_full = **+36.7pp, 95% CI [+33.5, +40.0]**, p_boot(>0)=1.0
- 10k resamples, seed=42, paired 125 samples
- 写入 `bootstrap_redaction_ci.json` 新字段 `oracle_full_vs_oracle_redacted_full_delta`，原 3 CI 未改
- **How to apply**: response letter C1 现在可以引用 leakage 36.7pp 的带 CI 的数字，不再是 point estimate

**Why**：这三项来自 worker_postprocess CPU-only 二次审计。前两项是 Agent/Reviewer 首次判断基于假设未验证即 write-through 到 D056/D057/指令——**提醒：涉及 rebuttal 字面描述的每个数字必须 cross-check 实际 artifact，不接受 "第一直觉合理" 的路径**

---

## [2026-04-14 08:35 UTC] exp_016 passage-redaction → C1 headline 降级 (partial_mechanism)

**结果**:Delta_full_redact = em_partial(oracle_redacted_full) − em_partial(no_api) = 0.131 − 0.021 = **+11.0pp**,落入预提交判定的 [+5, +25] **partial_mechanism** bucket。检索质量仅解释 exp_009 oracle SFT +47.7pp 增益的 **~23%**,passage-level gold-answer leakage 解释 **~77%**。Dose-response 单调见证成立:2.1% < 13.1% (redacted_full) < 37.3% (redacted_half) < 49.8% (oracle_full)。Phase-1 ±0.02 复现闸门 PASS。

**Why**:v3.4 leakage audit 已显示 98% gold substring hit rate,exp_016 是 Reviewer C1 质疑 oracle 是"答案泄漏而非检索"的可证伪测试。Worker 把 redaction 从 plan.md 纯 substring 改成 substring+word-boundary+quote-strip (commit 7cf099d),避免 "no"→"November" 级过度遮蔽——偏保守的 redaction 仍然拿到 +11pp,说明结论方向稳健。

**How to apply (对后续 rebuttal 写作)**:
- §1/§3/§5 原 headline "retrieval quality dominates oracle SFT gain" **不得保留**。改写为量化分解:"retrieval quality contributes ~23% of the +47.7pp oracle gain; passage-level answer leakage accounts for the remaining ~77%."
- §3.5 必须加 scope-limit 段落,主动披露 exp_016 redaction 对照(推荐作为"self-discovered limitation"写法,增强诚信信号,比被 reviewer 再次揪出强)
- exp_009 oracle 因果论述不可单独引用,必须与 exp_016 并列出现
- 任何后续 ablation 设计都要问:"是否可能被 gold-in-passage 污染?"——把 passage-level leakage 列为默认 confound 之一
- **教训**:"oracle retrieval" 作为 causal probe 默认假设了 retrieved passages 里没有 gold span。当 teacher trajectories 用 gold→搜索 的反向检索构造时,这个假设天然不成立。下次设计 oracle-style ablation 时,把 redaction baseline 直接内置为预注册对照

## [2026-04-13 18:25 UTC] exp_013 DONE 验证 checklist（Reviewer 中期审查指令）

exp_013 训练完成后**必须**执行 4 项验证，即使超 8h 预算也要跑 oracle 这一条：

1. **Llama student 截断率 vs exp_006 Qwen 截断率对比**
   - Why: 若 Llama 明显偏高，会混淆 "cross-backbone silent failure" 归因
   - How: 论文里必须明确报告截断率差异；否则读者会质疑 cross-backbone 对照不纯

2. **label masking 正确性审计**
   - Why: `pad_token=eos_token` + `PerNDataCollator` 组合下，Llama eos 位置可能被错误覆盖 loss
   - How: 验证 Llama tokenizer eos token id，对比 collator output labels 确认 eos 位置是 -100（masked）不是真实 label

3. **Surface metrics 表（同 exp_006 test split）**
   - Why: W2 完整证据链需要 exp_006 Qwen vs exp_013 Llama 同 test split 逐项对齐
   - How: tool_call_syntax / balanced_xml / non_empty_queries / action_seq / BLEU / ROUGE-L 全跑，在 exp_006 同一 test split（442 样本）

4. **Oracle retrieval ablation on Llama（必做，超预算也跑）**
   - Why: 复现 exp_009 +47.3pp 因果信号是 W2 完整证据链的关键一环
   - How: eval_sft_with_api.py --retrieval_mode oracle on Llama adapter，对比 no-API baseline 的 em_partial delta

**worker done ≠ 训练完成**：exp_013 真正 DONE 以 `training_meta.json` + final adapter 落盘为准，不是 pipeline worker 进程退出。registry status 由 cron / 在 eval 完成后手动切换，**不要提前切**。

## [2026-04-13 17:50 UTC] D053: lineno v9 正式关闭（accepted residual）

投稿前 line-number 对齐问题：d1f058d PDF 仍存在 2 行 cosmetic 瑕疵（第 42-43 行在 x=253 中间列位置而非右列 x=570，PyMuPDF 验证：x=12 有 41 行 [1-41]，x=253 有 2 行 [42-43]，x=570 有 38 行 [44-81]）。

**Why**：根源是 `\twocolumn[\@maketitle\abstract]` 头尾切换时 `\if@firstcolumn` 状态在极少数行上错判。v8 前多轮 Plan A/B/C 尝试均未根除。Director D053 决定接受残留，不再投入 Plan（v9 absolute-position \rlap 方案取消）。

**How to apply**：
- 论文正文**不**加 footnote 或任何解释——正文保持干净
- 投稿 cover letter / submission notes 里用一句话说明："Two line numbers (42-43) appear in the gutter between columns due to a known interaction with the ACL `\twocolumn[\@maketitle]` header-body transition; this is a purely cosmetic artifact and does not affect reviewing."
- v9 worker 已 cancel；acl.sty 保持 HEAD baseline，不做任何修改

## [2026-04-12 13:30 UTC] exp_009: Oracle retrieval 因果验证成功 + n_searches=1 策略简化发现

### 因果证据
给 SFT student 接入 oracle retrieval（从 teacher 轨迹注入真实 `<information>` 块）后，em_partial 从 N=8 的 0.025 暴涨到 0.498（+47.3pp）。trunc 从 83.2% 降到 0.0%。**behavioral mismatch 因果性确认**：student 学会了搜索策略但缺乏执行能力。

**⚠️ 2026-04-14 update**:此 +47.3pp 因果 claim 被 exp_016 passage-redaction 部分推翻——真实检索质量贡献仅 ~23% (+11pp),剩余 ~77% 来自 passage-level gold leakage。引用本条时必须与 exp_016 并列。

### n_searches=1 策略简化发现
所有 507 prompt avg_searches=1.0。SFT 只学到"搜索一次→回答"的简化模式，未学到 RL teacher 的多轮迭代搜索策略。这是 SFT collapse 的额外维度——不仅是执行能力缺失，还有策略简化（strategy simplification）。

### 论文写作注意
- N=8 student(0.498) vs teacher(0.455) 不可直接对比（不同 prompt 集）。论文只报告 with-API vs without-API 的 delta。
- Oracle 模式是性能上界（用 teacher 真实检索结果），非独立检索系统。论文需说明这一点。
- em_full 仍然低（N=8=0.008），因为单次搜索只能覆盖部分 objectives。

## [2026-04-12 07:05 UTC] §3 写作框架约束（Reviewer 审定）
- **exp_008 结论措辞上限**：即使 suppression 恢复 EM，论文只能说 search token 是 "contributing factor to distillation failure"，不能说 "root cause of behavioral mismatch"。原因：因果链不完整——模型容量不足也可能是原因，suppression 实验无法排除
- **§3 分析结构**（强制顺序）：
  1. **SFT suppressed vs SFT unsuppressed**（首要对比）→ 量化 search token 对 truncation 的贡献。如果 SFT suppressed 也恢复 EM，说明 truncation 本身是主 confound，与 DPO/SFT 方法差异无关
  2. **DPO suppressed** → 验证 DPO 是否也有同样的 behavioral pattern
  3. **综合讨论** behavioral mismatch 作为 contributing factor（hedged 表述）
- **禁止**：将 DPO vs SFT 作为 §3 首要对比框架

## [2026-04-10 23:14 UTC] D028: matched-sample 文档 drift 教训 + DPO 小数据最佳实践
- **hypothesis drift 根因**：registry 写 "109 pairs each" 但实际训练是 114/109/114（matched 在 raw pre-split 级别，非 post-split train 级别）。5% 差异在论文层面会被审稿人质疑诚信
- **修复方案**：改文本（方案 A），不改代码。hypothesis 改为声明 "matched at raw-pair level"，论文 Methods 需同步显式声明
- **DPO 小数据（~110 train）超参教训**：
  - epochs 2 过多（saturation@epoch1.70），1 epoch 足够
  - beta=0.1 过高导致 chosen-suppression mode（chosen reward 从 +0.04 降到 -1.91），降到 0.05
  - lr=5e-5 导致 loss 0.66→0.13 过快，降到 2e-5
  - 必须 max_grad_norm=1.0 做梯度裁剪
  - **禁止依赖 wandb offline run 做可观测性**——必须 dump trainer.state.log_history 到 JSON

## [2026-04-10 23:05 UTC] D026 审计教训：Worker 手动 nohup vs 编排脚本 + rate-limit 恢复
- **根因**：Worker 绕过现有 `scripts/exp_006_stage1_run.sh`（正确的 for-loop + set -euo pipefail），手动逐个 nohup 发 SFT/DPO 命令。被 rate limit 中断时 N4/N8 未启动
- **retry 恢复 bug**：框架 auto-retry worker 仅按磁盘文件存在性判断状态，未校验 ckpt mtime vs upstream ckpt mtime，导致用旧 SFT base 的 N4/N8 ckpt 跑 eval（污染结果）
- **教训 1**：Worker prompt 应明确要求 "优先使用项目已有的编排脚本，而非手动逐步 nohup"——shell 脚本在 LLM session 挂掉后能继续后台执行
- **教训 2**：Worker resume 时必须校验下游 ckpt 的 mtime ≥ 上游 ckpt 的 mtime，否则标记 stale 并拒绝使用
- **教训 3**：DPO 小数据（~114 train samples, 30 steps）容易出现 chosen-reward 抑制模式（chosen reward 下降而非上升，margin 增长全靠 suppress rejected）。N4/N8 重训应 epochs 2→1

## [2026-04-10 18:50 UTC] D023: SFT 入选准则 perfect-only → partial-credit (total_em ≥ 1)
- **触发**：Stage 0 真值 N=8 perfect=2 (0.4%) → SFT pool 仅 2 样本，"matched N=8 DPO−SFT ≥ 3pp" 因 SFT starvation 结构性不可达
- **决议**：SFT 入选从 `is_perfect == True` 改为 `total_em ≥ 1`（partial-credit）。每 prompt 取 max(total_em) 的 response 去重。从既有 raw_trajectories 重提取，不重跑 RL teacher
- **新数字**：N=2 SFT 490→**806**，N=4 76→**471**，N=8 2→**487**，total 568→**1764**。瓶颈解除
- **matched_size 拆分**：sft=mi
