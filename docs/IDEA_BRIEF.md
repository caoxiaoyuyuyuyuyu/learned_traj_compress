# Idea Brief: Why Agent Memory Consolidation Amplifies the SFT-RL Gap: Mechanistic Diagnosis and Preference Distillation for Tiny Sidecars


- **Idea ID**: `idea_028`
- **Created**: 2026-04-08
- **Approved**: 2026-04-09
- **Source**: web_research
- **Level**: top-tier
- **Target Venues**: AAAI 2027 (abstract July 25, 2026; submission August 1, 2026 — primary target, 4 months), NeurIPS 2026 Workshops (deadlines TBD, likely Aug-Sep 2026), EMNLP 2026 (ARR cycle, stretch goal)

## Seed Topic

帮我生成和LLM agent、harness engineering、LLM+搜索、LLM+graphics相关的idea，要求能中顶会，用双卡RTX PRO 6000（单卡96GB显存）及以下的资源（GPU 100h以下），尽量不用API key，不自己构建benchmark，不需要人工标注，尽量不要做RL训练（显存占用过高）！

## Generation Journey

1. **Start** — 开始生成，seed: '帮我生成和LLM agent、harness engineering、LLM+搜索、LLM+graphics相关的idea，要求能中顶会，用双卡RTX PRO 6000（单卡96GB显存）及以下的资源（GPU 100h以下），尽量不用API key，不自己构建benchmark，不需要人工标注，尽量不要做RL训练（显存占用过高）！'
2. **Problem Discovery** (714s) — 发现 4 个研究方向
   - SFT Distillation of Interleaved Search-Reasoning for Small LLMs
   - Rendering-Grounded Rejection Sampling for Visual Code SFT
   - Learned Online Trajectory Compression for LLM Agents
   - Multi-Format Visual Code Refinement via Difference-Aligned SFT
3. **Method Synthesis** (281s) — 完成: "SFT Distillation of Interleaved Search-Reasoning for Small LLMs" [solid] → NeurIPS 2026 (abstract May 4, full paper May 6), EMNLP 2026 (ARR submission May 25)
3. **Method Synthesis** (182s) — 完成: "Rendering-Grounded Rejection Sampling for Visual Code SFT" [solid] → NeurIPS 2026 (paper deadline May 6, 2026), EMNLP 2026 (ARR deadline May 25, 2026)
3. **Method Synthesis** (393s) — 完成: "Learned Online Trajectory Compression for LLM Agents" [solid] → NeurIPS 2026 (abstract May 4, paper May 6, 2026), EMNLP 2026 (ARR submission May 25, 2026; commitment Aug 2, 2026)
3. **Method Synthesis** (426s) — 完成: "Multi-Format Visual Code Refinement via Difference-Aligned SFT" [solid] → NeurIPS 2026 (abstract May 4, paper May 6), EMNLP 2026 (ARR submission May 25)

## Core Problem

RL-based agent memory methods (MEM1, MemPO, Memex(RL), Mem-α, Memory-R1) train 7B models for online context consolidation, but deploying a 7B sidecar alongside the main agent LLM is impractical. SFT distillation to smaller models collapses catastrophically — MEM1's ablation shows the SFT-RL gap grows from 1.4x at 1 objective to 18.5x at 6 objectives and complete collapse (0.000 EM) at 16 objectives. 'SFT Memorizes, RL Generalizes' (ICML 2025) establishes the general mechanism: SFT performs hard singular vector rotation that memorizes surface patterns while RL achieves soft re-alignment that generalizes OOD. However, 'Debunk the Myth of SFT Generalization' (arXiv 2510.00237) shows that prompt diversity + CoT supervision can close this gap on single-objective tasks (Sokoban, GeneralPoints). This raises a critical question: why does the gap remain CATASTROPHIC for agent memory consolidation despite MEM1's SFT models being trained on diverse multi-turn trajectories? We hypothesize that agent memory consolidation is structurally distinct: it requires simultaneously representing competing trade-offs across N objectives (what to retain, compress, or discard), and SFT's mode-seeking behavior collapses this multi-modal decision landscape into a single consolidation strategy. Prompt diversity and CoT fix surface-level memorization artifacts, but cannot address the fundamental mode collapse when the decision space grows combinatorially with objective count. This mode collapse is benign when N≤6 (the trade-off space is small enough for one strategy) but catastrophic when N>6 (the exponentially growing trade-off space cannot be captured by a single mode). No existing work diagnoses this domain-specific amplification mechanism or provides a distillation method informed by it.

## Opportunity / Why Now

MEM1 (ICLR 2026) releases both its 7B RL policy weights and its training framework, and critically uses a rule-based exact-match (EM) reward — no learned reward model exists or is needed. This means (1) we can probe the RL policy's internal representations directly, (2) we can construct preference pairs by evaluating candidate consolidations against downstream EM accuracy, and (3) we can do on-policy distillation with verifiable reward signals. The ICML 2025 mechanistic framework (singular vector rotation analysis) provides the analytical tools, but has only been applied to single-objective tasks (arithmetic games, navigation, Sokoban). The 'Debunk' paper (arXiv 2510.00237) demonstrates that SFT's generalization failure on those tasks is fixable with data curation — which actually STRENGTHENS the case that multi-objective consolidation represents a fundamentally harder problem requiring different solutions. BOND (Best-of-N Distillation) provides a clean distillation framework when verifiable rewards are available. The convergence of released weights + verifiable reward + established mechanistic tools + a clear gap in multi-objective analysis creates a unique opportunity to both understand and solve the agent memory distillation problem.

## Landscape (SOTA & Limitations)

Agent memory management has converged on RL-based training in early 2026. MEM1 (ICLR 2026, PPO, 7B) achieves 3.5x performance gain with 3.7x memory reduction via generative internal state consolidation; its SFT ablation shows a gap that grows from 1.4x at 1 objective to 18.5x at 6 objectives to complete collapse at 16 — the key motivating finding. The reward is pure EM accuracy. MemPO (Feb 2026, GRPO, 7B) gains 25.98% F1 via memory-specific advantage computation. Memex(RL) (Mar 2026, GRPO) introduces indexed experience memory with 24→86% success rate. EMPO2 (Feb 2026) uses hybrid on/off-policy RL for memory-augmented agents, achieving +128.6% on ScienceWorld. Mem-α (arXiv 2509.25911, GRPO) trains agents to manage core/episodic/semantic memory via RL, achieving 13x generalization beyond training length. Memory-R1 (arXiv 2508.19828, PPO+GRPO) achieves strong memory management with only 152 training examples.

On SFT-vs-RL mechanisms: 'SFT Memorizes, RL Generalizes' (ICML 2025) establishes that SFT memorizes surface patterns while RL generalizes OOD, tested on GeneralPoints and V-IRL. Its followup 'RL Fine-Tuning Heals OOD Forgetting' (arXiv 2509.12235) shows singular vector rotation drives both forgetting and recovery. Crucially, 'Debunk the Myth of SFT Generalization' (arXiv 2510.00237) challenges this narrative by showing that prompt diversity + CoT supervision makes SFT match RL on Sokoban and GeneralPoints — but these are single-objective tasks where the decision space does not grow combinatorially. MEM1's SFT models were trained with diverse multi-turn trajectories (not frozen prompts), yet still collapse at >6 objectives, suggesting a deeper failure mechanism beyond the prompt artifacts that the 'Debunk' paper addresses.

On distillation without reward models: BOND (Best-of-N Distillation) generates N candidates, selects best via any scoring function (including rule-based), and distills the BoN distribution into a student. DeepSeek-R1-Distill demonstrates that rejection sampling with rule-based correctness rewards suffices for reasoning distillation. The OPD Survey (Apr 2026) shows on-policy distillation mathematically equals maximizing token-level reward from teacher log-probabilities with KL penalty.

On mode collapse: 'The Price of Format: Diversity Collapse in LLMs' documents alignment-induced diversity contraction. GRPO exhibits diversity collapse, amplifying single solution strategies. PRISM (ICLR 2026) shows SFT causes entropy collapse that constrains exploration.

Critical gaps: (1) No analysis of why the SFT-RL gap is catastrophically amplified for agent memory consolidation vs. merely degraded (or even fixable) for single-objective tasks. (2) No distillation method designed for multi-objective consolidation policies with verifiable rewards. (3) No capacity scaling analysis separating model size limitations from mode collapse at varying objective counts.

## Proposed Approach

We propose MemDistill, a framework that first diagnoses WHY agent memory consolidation catastrophically amplifies the general SFT-RL gap (even when prompt diversity + CoT fixes it for single-objective tasks), then applies that understanding to design preference distillation with verifiable rewards for tiny (<3B) sidecars. Three phases on single RTX PRO 6000 (96GB).

**Phase 1 — Domain-Specific Amplification Diagnosis (~20h GPU):** Building on the ICML 2025 finding that SFT memorizes / RL generalizes via singular vector rotation, and the 'Debunk' paper showing this is fixable for single-objective tasks, we ask: what is structurally different about multi-objective consolidation that makes the gap persist and grow catastrophically?
(a) *Singular vector rotation analysis per objective count*: Following arXiv 2509.12235's spectral SVD methodology, measure singular vector rotation magnitude in MEM1-7B vs SFT students (Qwen2.5-0.5B, 1.5B, 3B) at 4/8/12/16 objectives. Hypothesis: rotation magnitude scales superlinearly with objective count for SFT (greedy hard alignment across conflicting objectives) but sublinearly for RL (soft re-alignment finds a compromise).
(b) *Mode collapse quantification*: For each objective count, generate 100 consolidation outputs from teacher and SFT-student for the same inputs. Measure output diversity (distinct n-grams, embedding variance, strategy clustering). Hypothesis: SFT diversity collapses sharply at >6 objectives — the student converges to one consolidation strategy — while the teacher maintains multiple modes.
(c) *Per-objective information probing*: Apply linear probes at multiple layers to measure how much per-objective information is retained. Identify WHICH objectives' information SFT loses first and whether it correlates with objective conflict (measured by gradient cosine similarity between objectives).
(d) *Capacity vs. mode collapse disentanglement*: If Qwen2.5-3B SFT still collapses at >6 objectives but DPO-1.5B doesn't, the problem is mode collapse, not capacity. This is the critical experiment separating our story from a 'just use a bigger model' narrative.
(e) *Prompt diversity + CoT control*: Re-train SFT students with augmented prompt diversity and CoT traces (following arXiv 2510.00237's methodology). Confirm that these interventions, which fix single-objective tasks, do NOT fix multi-objective consolidation — establishing that the failure is structural, not artifactual.

**Phase 2 — EM-Grounded Preference Distillation (~30h GPU):** Since MEM1 uses rule-based EM reward (no learned reward model needed), we construct preference pairs via verifiable downstream accuracy:
(a) *Best-of-N pair construction*: For each input, generate N=16 candidate consolidations from the student. Evaluate each by running downstream QA with the consolidated memory and computing EM accuracy. Rank candidates; top-k become positives, bottom-k become negatives. This is on-policy (student's own distribution) and grounded in verifiable reward (EM accuracy). **Concrete budget**: MEM1's training set contains ~2,000 episodes. At N=16 candidates per episode, this yields 32,000 student forward passes. Qwen2.5-1.5B generates ~200 tokens/s on RTX PRO 6000; at ~500 tokens per consolidation output, each pass takes ~2.5s → 32K × 2.5s ≈ 22h student inference. Downstream QA evaluation per candidate (short-form EM) adds ~0.5s per candidate → 32K × 0.5s ≈ 4.5h. Total pair construction: ~27h. To fit within budget, we subsample to 1,200 episodes (19.2K student passes ≈ 13h + 2.7h QA ≈ 16h), reserving 800 for validation.
(b) *Teacher-likelihood filtering*: Additionally score candidates under MEM1-7B's log-likelihood. Pairs where EM and teacher-likelihood agree get higher weight; disagreements are flagged for analysis (they reveal where the student finds valid strategies the teacher doesn't use — potentially beneficial diversity).
(c) *Cold-start initialization*: SFT the student on easy tasks (≤6 objectives) where SFT doesn't collapse, establishing stable output format and basic consolidation ability. Then switch to preference optimization for high-objective tasks. This follows the ICML 2025 insight that SFT provides format stabilization that enables subsequent RL/preference optimization.
(d) *Progressive objective curriculum*: Train preference optimization starting from 7 objectives and progressively increasing to 16, preventing the student from being overwhelmed by the full multi-objective trade-off space immediately.
(e) *Calibrated teacher deferral*: When the student's EM accuracy on a consolidation drops below a calibrated threshold (set on validation data), defer to the 7B teacher. Report the deferral rate as a function of student size and objective count.

**Phase 3 — Evaluation (~15h GPU):**
(a) *Primary evaluation (MEM1's QA domain)*: MemDistill students (Qwen2.5-0.5B/1.5B) vs. MEM1-7B teacher vs. vanilla SFT students vs. prompt-diverse+CoT SFT students vs. heuristic baselines (ACON, AgentDiet, Focus). Test across 4/8/12/16 objective counts, extending MEM1's collapse curve.
(b) *Transfer evaluation (coding domain)*: Evaluate on SWE-smith coding trajectories to test whether the distilled consolidation policy transfers beyond QA. Single transfer domain to keep scope manageable.
(c) *Key ablations*: (i) DPO vs EM-grounded preference pairs — does verifiable reward outperform teacher-only signal? (ii) On-policy (student-generated) vs off-policy (teacher-generated) pairs. (iii) Cold-start vs random init. (iv) Student size scaling (Qwen2.5-0.5B/1.5B/3B). (v) N in Best-of-N (4/8/16/32).
(d) *Mechanistic validation*: Re-run Phase 1's mode collapse and probing analyses on the preference-distilled students. Confirm that the distillation method specifically addresses the diagnosed failure mechanism (restores multi-modal diversity, retains per-objective information).

Key metrics: EM accuracy preservation (% of teacher), mode diversity (distinct strategies per objective count), per-objective probe accuracy, token reduction ratio, inference latency, deferral rate.

**Broader implications**: Our analysis suggests that any task requiring simultaneous optimization of multiple competing objectives — not just memory consolidation, but also multi-criteria code generation, multi-stakeholder negotiation, multi-constraint planning — will exhibit similar SFT-RL gap amplification that prompt diversity and CoT cannot fix. The mode collapse mechanism we diagnose is not specific to memory; it arises whenever the optimal policy must maintain multiple modes over a combinatorially growing trade-off space. We leave verification of this broader hypothesis to future work, but our framework (SVD rotation analysis + diversity quantification + per-objective probing) provides a reusable diagnostic toolkit for any domain where SFT-to-RL distillation fails unexpectedly.

## Resource Estimate

- **auto_research**: {'claude_api_cost': '$180-380 (~2200-4000 turns)', 'estimated_gpu_cost': '$0', 'gpu_hours': '~95-115h RTX PRO 6000: thorough hyperparameter sweeps, more N values for Best-of-N, additional probing layers', 'gpu_utilization': 'High (~80-90%). Single 96GB card handles all workloads without model sharding.', 'risk_note': 'Agent may over-explore N values and probing configurations. Mitigation: gate on establishing the mode-collapse finding before exhaustive probing.', 'speed_bottleneck': 'Best-of-N generation (N=16 × 1,200 episodes) is the main bottleneck; 96GB enables large batch sizes. Concrete: ~16h for pair construction with Qwen2.5-1.5B.', 'team': '0 (initial setup: download MEM1 weights, prepare QA + coding datasets)', 'timeline': '3-4 weeks with RTX PRO 6000'}
- **human_in_loop**: {'claude_api_cost': '$100-250 (~1200-3000 turns)', 'estimated_gpu_cost': '$0', 'gpu_hours': '~80-95h RTX PRO 6000: agent manages pipeline, human reviews SVD analysis and diversity metrics', 'gpu_utilization': 'High (~65-75%). 96GB VRAM allows MEM1-7B + student on single card for efficient Best-of-N generation.', 'risk_note': "Key risk: if SVD analysis shows agent memory rotation patterns are NOT qualitatively different from single-objective tasks, the 'domain-specific amplification' narrative weakens. Mitigation: (1) mode collapse quantification provides an independent diagnostic, (2) Phase 1e's prompt-diversity+CoT control directly tests whether the known single-objective fix transfers, and (3) even a negative result (the fix works!) is publishable as extending the 'Debunk' findings to multi-objective settings.", 'team': '1 researcher, ~30 min/day check-in', 'timeline': '5-6 weeks (mechanistic week 1-2, distillation weeks 2-3, evaluation week 4, paper weeks 5-6)'}
- **manual**: {'estimated_gpu_cost': "$0 (user's own RTX PRO 6000, 96GB)", 'gpu_hours': '~70-85h RTX PRO 6000 (96GB): ~20h mechanistic analysis (SVD + probing + diversity + prompt-diversity control), ~16h Best-of-N pair construction (1,200 episodes × N=16, concrete budget above), ~15h preference training (0.5B-3B students), ~15h evaluation + ablations, ~5h supplementary', 'gpu_utilization': 'Moderate (~50-60%). Human bottleneck: interpreting SVD rotation patterns, analyzing mode collapse visualizations, paper writing.', 'team': '1 researcher with ~1-2h/day active work', 'timeline': '6-7 weeks (1.5 weeks mechanistic analysis, 1.5 weeks pair construction + training, 1.5 weeks evaluation + ablations, 2 weeks paper writing)'}

## Key References

- MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents (ICLR 2026, arXiv 2506.15841)
- SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training (ICML 2025, arXiv 2501.17161)
- Debunk the Myth of SFT Generalization (arXiv 2510.00237, Oct 2025)
- RL Fine-Tuning Heals OOD Forgetting in SFT (arXiv 2509.12235)
- MemPO: Self-Memory Policy Optimization for Long-Horizon Agents (arXiv 2603.00680, Feb 2026)
- Memex(RL): Scaling Long-Horizon LLM Agents via Indexed Experience Memory (arXiv 2603.04257, Mar 2026)
- EMPO2: Exploratory Memory-Augmented Policy Optimization (arXiv 2602.23008, Feb 2026)
- Mem-α: Learning Memory Construction via Reinforcement Learning (arXiv 2509.25911)
- Memory-R1: Enhancing LLM Agents to Manage and Utilize Memories via Reinforcement Learning (arXiv 2508.19828)
- MemFactory: Unified Inference & Training Framework for Agent Memory (arXiv 2603.29493, Mar 2026)
- PRISM: Consolidation or Adaptation? Disentangling SFT and RL Data via Gradient Concentration (arXiv 2601.07224, ICLR 2026)
- BOND: Best-of-N Distillation for Aligning LLMs (OpenReview, 2024)
- A Survey of On-Policy Distillation for Large Language Models (arXiv 2604.00626, Apr 2026)
- Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes (arXiv 2603.25562, Mar 2026)
- DGPO: Distillation-Guided Policy Optimization for Compact Language Models in Agentic RAG (arXiv 2508.20324, NeurIPS 2025)
- The Price of Format: Diversity Collapse in LLMs (2024)
- ACON: Optimizing Context Compression for Long-horizon LLM Agents (arXiv 2510.00615, Oct 2025)
- Direct Preference Optimization: Your Language Model is Secretly a Reward Model (NeurIPS 2023, arXiv 2305.18290)
- ADPA: Advantage-Guided Distillation for Preference Alignment in Small Language Models (arXiv 2502.17927)
- AgentDiet: Improving the Efficiency of LLM Agent Systems through Trajectory Reduction (arXiv 2025)
- Focus: Active Context Compression in LLM Agents (arXiv 2026)

## Reviewer Evaluation

**Scores**:
  - clarity: **{'rationale': "Excellent three-phase structure with concrete, falsifiable hypotheses (e.g., 'SFT diversity collapses sharply at >6 objectives'). Clear engagement with both supporting and opposing prior work. The prompt-diversity+CoT control experiment (Phase 1e) is particularly well-motivated. Resource estimates include concrete token/time calculations.", 'score': 4}**/5
  - feasibility: **{'rationale': 'MEM1 weights are public, rule-based EM reward eliminates reward model dependency, Qwen2.5 student sizes are correct, 96GB VRAM fits teacher+student on one card. The ~85h GPU budget is concrete with detailed per-phase breakdown (20h diagnosis + 16h pair construction + 15h training + 15h eval). Best-of-N budget calculation (1,200 episodes × N=16 × 2.5s ≈ 13h) is realistic.', 'score': 4}**/5
  - impact: **{'rationale': 'Addresses a practical deployment problem (7B sidecar alongside main agent LLM is impractical) in a rapidly growing area (5+ RL-memory papers in early 2026). The diagnostic toolkit (SVD rotation + diversity quantification + per-objective probing) is reusable. OPD Survey explicitly identifies agent-level distillation as open. Broader implications for multi-objective SFT-RL gap are plausible if speculative.', 'score': 4}**/5
  - novelty: **{'rationale': "The specific diagnosis of why multi-objective consolidation catastrophically amplifies the SFT-RL gap (mode collapse that prompt diversity + CoT cannot fix) occupies a genuinely unoccupied niche. Well-differentiated from both 'SFT Memorizes, RL Generalizes' (general mechanism) and 'Debunk' (single-objective fix). The EM-grounded preference distillation with BOND-inspired pair construction is a natural but non-trivial methodological contribution.", 'score': 4}**/5
