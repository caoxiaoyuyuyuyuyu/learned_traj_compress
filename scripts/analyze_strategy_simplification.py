"""Strategy Simplification 定量分析

分析 SFT student 的 strategy simplification 现象：
1. Teacher search 轮数分布
2. Student vs Teacher search query 质量对比
3. Per-objective EM breakdown（验证"只搜一次"假说）
"""

import json
import re
import os
import numpy as np
from collections import Counter, defaultdict

BASE = "/root/autodl-tmp/learned_traj_compress/artifacts"
TEACHER_DIR = f"{BASE}/phase1d_v2_data"
STUDENT_FILE = f"{BASE}/exp_009_eval/eval_sft_shared_api_oracle.json"
OUTPUT_FILE = f"{BASE}/exp_009_eval/strategy_analysis.json"


def extract_search_queries(text):
    """Extract all search queries from generated text."""
    return re.findall(r'<search>(.*?)</search>', text, re.DOTALL)


def compute_token_overlap(q1, q2):
    """Compute token-level Jaccard overlap between two queries."""
    t1 = set(q1.lower().split())
    t2 = set(q2.lower().split())
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def analyze_teacher_searches(n_obj):
    """Analyze teacher search round distribution for a given N."""
    path = f"{TEACHER_DIR}/raw_trajectories_N{n_obj}.json"
    data = json.load(open(path))
    results = data['results']

    all_searches = []
    teacher_queries_by_prompt = {}  # prompt_idx -> list of queries (from best response)

    for r in results:
        prompt_idx = r['prompt_idx']
        # Pick the best response (first perfect one, or first one)
        best_resp = None
        for resp in r['responses']:
            if resp.get('is_perfect'):
                best_resp = resp
                break
        if best_resp is None:
            best_resp = r['responses'][0]

        text = best_resp['full_assistant']
        queries = extract_search_queries(text)
        n_search = len(queries)
        all_searches.append(n_search)
        teacher_queries_by_prompt[prompt_idx] = queries

    all_searches = np.array(all_searches)
    # Distribution
    counter = Counter(all_searches.tolist())
    dist = {int(k): int(v) for k, v in sorted(counter.items())}
    total = len(all_searches)
    dist_pct = {int(k): round(v / total * 100, 1) for k, v in sorted(counter.items())}

    stats = {
        "n_prompts": total,
        "mean": round(float(all_searches.mean()), 3),
        "median": float(np.median(all_searches)),
        "std": round(float(all_searches.std()), 3),
        "min": int(all_searches.min()),
        "max": int(all_searches.max()),
        "distribution_count": dist,
        "distribution_pct": dist_pct,
    }
    return stats, teacher_queries_by_prompt


def analyze_student_oracle():
    """Analyze student oracle results."""
    data = json.load(open(STUDENT_FILE))
    results = {}

    for n_key in ["N2", "N4", "N8"]:
        details = data['details'][n_key]
        n_obj = int(n_key[1:])

        searches = []
        per_obj_correct = defaultdict(list)  # obj_position -> list of 0/1
        student_queries_by_idx = {}

        for i, det in enumerate(details):
            n_s = det.get('n_searches', 0)
            searches.append(n_s)

            # Per-objective EM
            per_q = det.get('per_q_em', [])
            for pos, em in enumerate(per_q):
                per_obj_correct[pos].append(em)

            # Extract student queries
            gen_text = det.get('generated_text', '')
            queries = extract_search_queries(gen_text)
            student_queries_by_idx[i] = queries

        searches = np.array(searches)
        search_dist = Counter(searches.tolist())

        # Per-objective accuracy
        per_obj_acc = {}
        for pos in sorted(per_obj_correct.keys()):
            vals = per_obj_correct[pos]
            per_obj_acc[f"obj_{pos+1}"] = {
                "n": len(vals),
                "correct": sum(vals),
                "accuracy": round(sum(vals) / len(vals), 4) if vals else 0,
            }

        results[n_key] = {
            "n_prompts": len(details),
            "avg_searches": round(float(searches.mean()), 3),
            "search_distribution": {int(k): int(v) for k, v in sorted(search_dist.items())},
            "per_objective_accuracy": per_obj_acc,
            "student_queries": student_queries_by_idx,
        }

    return results, data


def analyze_query_overlap(teacher_queries_by_prompt, student_data, n_key, raw_teacher_data):
    """Compare student and teacher search queries for matching prompts."""
    # Need to match prompts between teacher and student
    # Teacher has prompt_idx, student details are in order of test set
    # Load teacher data to get prompt_idx -> questions mapping
    n_obj = int(n_key[1:])
    teacher_path = f"{TEACHER_DIR}/raw_trajectories_N{n_obj}.json"
    teacher_data = json.load(open(teacher_path))

    # Build teacher question->queries lookup
    teacher_by_questions = {}
    for r in teacher_data['results']:
        q_key = tuple(r['questions'])
        best_resp = None
        for resp in r['responses']:
            if resp.get('is_perfect'):
                best_resp = resp
                break
        if best_resp is None:
            best_resp = r['responses'][0]
        teacher_by_questions[q_key] = extract_search_queries(best_resp['full_assistant'])

    # Student details - try to match via gold_answers
    student_details = student_data['details'][n_key]

    overlaps = []
    matched = 0
    student_first_query_matches_teacher_first = 0
    student_single_vs_teacher_multi = []

    for det in student_details:
        # Extract student queries
        gen_text = det.get('generated_text', '')
        s_queries = extract_search_queries(gen_text)
        if not s_queries:
            continue

        # Try to find matching teacher prompt by gold_answers
        gold = tuple(tuple(g) for g in det['gold_answers'])

        # Search teacher data for matching gold_answers
        t_queries = None
        for r in teacher_data['results']:
            t_gold = tuple(tuple(g) for g in r['gold_answers'])
            if t_gold == gold:
                best_resp = None
                for resp in r['responses']:
                    if resp.get('is_perfect'):
                        best_resp = resp
                        break
                if best_resp is None:
                    best_resp = r['responses'][0]
                t_queries = extract_search_queries(best_resp['full_assistant'])
                break

        if t_queries is None or not t_queries:
            continue

        matched += 1

        # Compare first query
        if s_queries and t_queries:
            overlap = compute_token_overlap(s_queries[0], t_queries[0])
            overlaps.append(overlap)
            if overlap > 0.5:
                student_first_query_matches_teacher_first += 1

        # Track cases where teacher used multiple searches
        if len(t_queries) > 1 and len(s_queries) == 1:
            student_single_vs_teacher_multi.append({
                "teacher_n_searches": len(t_queries),
                "teacher_queries": t_queries,
                "student_query": s_queries[0],
                "em_full": det.get('em_full', 0),
                "per_q_em": det.get('per_q_em', []),
            })

    return {
        "matched_prompts": matched,
        "avg_first_query_overlap": round(float(np.mean(overlaps)), 4) if overlaps else None,
        "median_first_query_overlap": round(float(np.median(overlaps)), 4) if overlaps else None,
        "first_query_match_rate_gt05": round(student_first_query_matches_teacher_first / matched, 4) if matched else None,
        "student_single_teacher_multi_count": len(student_single_vs_teacher_multi),
        "student_single_teacher_multi_pct": round(len(student_single_vs_teacher_multi) / matched * 100, 1) if matched else None,
        "examples_single_vs_multi": student_single_vs_teacher_multi[:5],  # first 5 examples
    }


def main():
    print("=" * 60)
    print("Strategy Simplification Analysis")
    print("=" * 60)

    output = {}

    # 1. Teacher search distribution
    print("\n### 1. Teacher Search Round Distribution ###")
    teacher_stats = {}
    teacher_queries = {}
    for n in [2, 4, 8]:
        stats, queries = analyze_teacher_searches(n)
        teacher_stats[f"N{n}"] = stats
        teacher_queries[n] = queries
        print(f"\nN={n}:")
        print(f"  Prompts: {stats['n_prompts']}")
        print(f"  Mean searches: {stats['mean']}")
        print(f"  Median: {stats['median']}, Std: {stats['std']}")
        print(f"  Min: {stats['min']}, Max: {stats['max']}")
        print(f"  Distribution: {stats['distribution_pct']}")

    output["teacher_search_distribution"] = teacher_stats

    # 2. Student oracle analysis
    print("\n### 2. Student Oracle - Per-Objective Breakdown ###")
    student_results, student_data = analyze_student_oracle()

    for n_key in ["N2", "N4", "N8"]:
        sr = student_results[n_key]
        print(f"\n{n_key}:")
        print(f"  Avg searches: {sr['avg_searches']}")
        print(f"  Search distribution: {sr['search_distribution']}")
        print(f"  Per-objective accuracy:")
        for obj, acc in sr['per_objective_accuracy'].items():
            print(f"    {obj}: {acc['correct']}/{acc['n']} = {acc['accuracy']:.1%}")

    # Store without the bulky student_queries
    student_summary = {}
    for n_key in ["N2", "N4", "N8"]:
        sr = student_results[n_key].copy()
        del sr['student_queries']
        student_summary[n_key] = sr
    output["student_oracle_per_objective"] = student_summary

    # 3. Query overlap analysis
    print("\n### 3. Student vs Teacher Query Overlap ###")
    query_overlap = {}
    for n in [2, 4, 8]:
        n_key = f"N{n}"
        teacher_path = f"{TEACHER_DIR}/raw_trajectories_N{n}.json"
        raw_data = json.load(open(teacher_path))
        overlap = analyze_query_overlap(teacher_queries[n], student_data, n_key, raw_data)
        query_overlap[n_key] = overlap
        print(f"\n{n_key}:")
        print(f"  Matched prompts: {overlap['matched_prompts']}")
        print(f"  First query Jaccard overlap: mean={overlap['avg_first_query_overlap']}, median={overlap['median_first_query_overlap']}")
        print(f"  First query match rate (>0.5): {overlap['first_query_match_rate_gt05']}")
        print(f"  Student 1-search vs Teacher multi-search: {overlap['student_single_teacher_multi_count']} ({overlap['student_single_teacher_multi_pct']}%)")

    output["query_overlap"] = query_overlap

    # 4. Key hypothesis test: does per-obj accuracy drop for later objectives?
    print("\n### 4. Hypothesis: First Objective >> Later Objectives ###")
    for n_key in ["N2", "N4", "N8"]:
        accs = student_summary[n_key]['per_objective_accuracy']
        acc_list = [accs[k]['accuracy'] for k in sorted(accs.keys())]
        drop = acc_list[0] - acc_list[-1] if len(acc_list) > 1 else 0
        print(f"  {n_key}: obj accuracies = {acc_list}, drop first→last = {drop:.4f}")
        output.setdefault("hypothesis_first_vs_later", {})[n_key] = {
            "per_obj_accuracies": acc_list,
            "first_obj_acc": acc_list[0],
            "last_obj_acc": acc_list[-1],
            "drop": round(drop, 4),
        }

    # Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
