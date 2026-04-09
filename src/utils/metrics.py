"""Unified EM evaluation and reward utilities for MemDistill.

Uses normalize_answer() (SQuAD standard with article removal) consistently
across training reward and evaluation, resolving the MEM1 train/eval discrepancy.
"""

import re
import string
from collections import Counter
from typing import Optional


# ---------------------------------------------------------------------------
# Text normalization (SQuAD standard — matches MEM1 training reward)
# ---------------------------------------------------------------------------

def normalize_answer(text: str) -> str:
    """Normalize answer text using SQuAD standard.

    Pipeline: lowercase -> remove punctuation -> remove articles -> collapse whitespace.
    This matches MEM1's training reward `normalize_answer()`, NOT `eval.py`'s
    `preprocess_text()` which omits article removal.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


# ---------------------------------------------------------------------------
# Single-question EM
# ---------------------------------------------------------------------------

def em_check(prediction: str, golden_answers: list[str]) -> int:
    """Check exact match between prediction and any golden answer.

    Returns 1 if normalized prediction matches any normalized golden answer, else 0.
    """
    norm_pred = normalize_answer(prediction)
    for golden in golden_answers:
        if normalize_answer(golden) == norm_pred:
            return 1
    return 0


# ---------------------------------------------------------------------------
# Answer extraction from multi-turn output
# ---------------------------------------------------------------------------

def extract_solution(text: str, min_answer_tags: int = 2) -> Optional[str]:
    """Extract final answer from model output.

    MEM1 requires >= min_answer_tags <answer> tags to consider the output valid
    (multi-turn design: intermediate answers exist, only the last one counts).

    Args:
        text: Full model output text.
        min_answer_tags: Minimum number of <answer> tags required. Default 2
            (MEM1 convention). Set to 1 for single-turn evaluation.

    Returns:
        Extracted answer string, or None if fewer than min_answer_tags found.
    """
    matches = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL))
    if len(matches) < min_answer_tags:
        return None
    return matches[-1].group(1).strip()


# ---------------------------------------------------------------------------
# Multi-objective reward computation
# ---------------------------------------------------------------------------

def compute_em_reward(
    model_output: str,
    ground_truth_targets: list[list[str]],
    min_answer_tags: int = 2,
) -> int:
    """Compute EM reward for a multi-objective episode.

    Matches MEM1 training reward: extract answer -> split by semicolon ->
    per-question binary EM -> sum.

    Args:
        model_output: Full model output containing <answer> tags.
        ground_truth_targets: List of lists, where ground_truth_targets[i] is
            the list of acceptable answers for question i.
        min_answer_tags: Minimum <answer> tags required (see extract_solution).

    Returns:
        Integer reward in [0, N] where N = len(ground_truth_targets).
        Returns 0 if answer extraction fails.
    """
    answer = extract_solution(model_output, min_answer_tags=min_answer_tags)
    if answer is None:
        return 0

    parts = [a.strip() for a in answer.split(";")]
    score = 0
    for idx, golden_answers in enumerate(ground_truth_targets):
        if idx < len(parts):
            score += em_check(parts[idx], golden_answers)
    return score


# ---------------------------------------------------------------------------
# Reward normalization for Best-of-N pair construction (Phase 2a)
# ---------------------------------------------------------------------------

def normalize_rewards_minmax(rewards: list[float]) -> list[float]:
    """Min-max normalize rewards to [0, 1] within an episode's candidates.

    Use this when comparing candidates within a single episode to remove
    the effect of objective count on reward scale (N=1 -> [0,1] vs N=16 -> [0,16]).

    Args:
        rewards: List of raw rewards for N candidates of one episode.

    Returns:
        Normalized rewards in [0, 1]. Returns all zeros if min == max.
    """
    if not rewards:
        return []
    r_min = min(rewards)
    r_max = max(rewards)
    if r_max == r_min:
        return [0.0] * len(rewards)
    return [(r - r_min) / (r_max - r_min) for r in rewards]


def rank_candidates(rewards: list[float]) -> list[int]:
    """Rank candidates by reward (highest = rank 0).

    Use this for rank-based pair construction that doesn't depend on
    absolute reward values.

    Args:
        rewards: List of raw rewards for N candidates.

    Returns:
        List of ranks (0-indexed, 0 = best).
    """
    indexed = sorted(enumerate(rewards), key=lambda x: -x[1])
    ranks = [0] * len(rewards)
    for rank, (idx, _) in enumerate(indexed):
        ranks[idx] = rank
    return ranks


def construct_preference_pairs(
    candidates: list[str],
    rewards: list[float],
    method: str = "best_worst",
) -> list[tuple[str, str]]:
    """Construct preference pairs from ranked candidates.

    Args:
        candidates: List of candidate outputs.
        rewards: Corresponding rewards.
        method: Pair construction strategy.
            - "best_worst": Single pair of (best, worst).
            - "best_vs_rest": Best vs every non-best candidate.
            - "adjacent": All adjacent pairs in ranking order.

    Returns:
        List of (chosen, rejected) tuples.
    """
    if len(candidates) < 2:
        return []

    ranked = sorted(zip(rewards, candidates), key=lambda x: -x[0])

    if method == "best_worst":
        return [(ranked[0][1], ranked[-1][1])]
    elif method == "best_vs_rest":
        best = ranked[0][1]
        return [(best, r[1]) for r in ranked[1:] if r[0] < ranked[0][0]]
    elif method == "adjacent":
        pairs = []
        for i in range(len(ranked) - 1):
            if ranked[i][0] > ranked[i + 1][0]:
                pairs.append((ranked[i][1], ranked[i + 1][1]))
        return pairs
    else:
        raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------

def compute_f1(prediction: str, golden_answers: list[str]) -> float:
    """Compute token-level F1 score, taking max over golden answers.

    Uses Counter-based overlap (SQuAD official), not set-based,
    so duplicate tokens are counted correctly.
    """
    pred_tokens = normalize_answer(prediction).split()
    best_f1 = 0.0
    for golden in golden_answers:
        gold_tokens = normalize_answer(golden).split()
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_common = sum(common.values())
        if num_common == 0:
            continue
        precision = num_common / len(pred_tokens) if pred_tokens else 0
        recall = num_common / len(gold_tokens) if gold_tokens else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        best_f1 = max(best_f1, f1)
    return best_f1


def evaluate_multi_objective(
    model_output: str,
    ground_truth_targets: list[list[str]],
    min_answer_tags: int = 2,
) -> dict:
    """Full evaluation of a multi-objective episode.

    Returns per-question and aggregated EM, F1 metrics.
    """
    n_objectives = len(ground_truth_targets)
    answer = extract_solution(model_output, min_answer_tags=min_answer_tags)

    if answer is None:
        return {
            "em_total": 0,
            "em_mean": 0.0,
            "f1_mean": 0.0,
            "n_objectives": n_objectives,
            "extraction_failed": True,
            "per_question": [],
        }

    parts = [a.strip() for a in answer.split(";")]
    per_question = []
    for idx, golden_answers in enumerate(ground_truth_targets):
        pred = parts[idx] if idx < len(parts) else ""
        em = em_check(pred, golden_answers)
        f1 = compute_f1(pred, golden_answers)
        per_question.append({"em": em, "f1": f1, "prediction": pred})

    em_total = sum(q["em"] for q in per_question)
    return {
        "em_total": em_total,
        "em_mean": em_total / n_objectives if n_objectives > 0 else 0.0,
        "f1_mean": sum(q["f1"] for q in per_question) / n_objectives if n_objectives > 0 else 0.0,
        "n_objectives": n_objectives,
        "extraction_failed": False,
        "per_question": per_question,
    }
