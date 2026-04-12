#!/bin/bash
# Exp 008: Search token suppression eval (D034)
#
# Tests behavioral mismatch hypothesis: suppress <search>/<information> tokens
# during generation via bad_words_ids bigrams. If em_partial recovers >5pp and
# truncated_rate drops >20pp, behavioral mismatch is confirmed.
#
# Models: SFT shared + merged DPO N8 (baseline results already exist)
# Eval: N=2,4,8 test prompts with --suppress_search_tokens
#
# Usage:
#   bash scripts/exp_008_search_suppression_eval.sh          # both models
#   SKIP_DPO=1 bash scripts/exp_008_search_suppression_eval.sh  # SFT only
set -euo pipefail

# --- Absolute paths (AutoDL) ---
PROJECT_DIR="/root/autodl-tmp/learned_traj_compress"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-3B}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/artifacts/phase1d_v2_data}"
CKPT_DIR="${CKPT_DIR:-${PROJECT_DIR}/checkpoints}"
EVAL_DIR="${EVAL_DIR:-${PROJECT_DIR}/artifacts/exp_008_eval}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export HF_HOME="${HF_HOME:-/root/autodl-tmp/.hf_cache}"

echo "============================================"
echo "Exp 008: Search Token Suppression Eval"
echo "============================================"
echo "Project:  ${PROJECT_DIR}"
echo "Model:    ${MODEL_PATH}"
echo "Data:     ${DATA_DIR}"
echo "Eval out: ${EVAL_DIR}"
echo ""

mkdir -p "$EVAL_DIR"

# --- Eval 1: SFT shared with suppression ---
SFT_DIR="${CKPT_DIR}/exp_006_sft_shared"
if [ -d "$SFT_DIR" ]; then
    echo "--- Evaluating: sft_shared_suppressed ---"
    python3 -u "${SCRIPT_DIR}/exp_006_stage1_eval.py" \
        --model_path "$MODEL_PATH" \
        --sft_adapter "$SFT_DIR" \
        --model_name "sft_shared_suppressed" \
        --data_dir "$DATA_DIR" \
        --output_dir "$EVAL_DIR" \
        --suppress_search_tokens
    echo "[OK] SFT shared suppressed eval complete"
    echo ""
else
    echo "[ERROR] SFT adapter not found: ${SFT_DIR}"
    exit 1
fi

# --- Eval 2: Merged DPO N8 with suppression ---
if [ "${SKIP_DPO:-0}" = "1" ]; then
    echo "[SKIP] DPO eval skipped (SKIP_DPO=1)"
else
    # Try merged checkpoint first (avoids OOM from stacking adapters)
    MERGED_DPO_N8="${CKPT_DIR}/merged_dpo_N8"
    DPO_N8_ADAPTER="${CKPT_DIR}/exp_006_dpo_N8"
    if [ -d "$MERGED_DPO_N8" ]; then
        echo "--- Evaluating: merged_dpo_N8_suppressed (merged checkpoint) ---"
        python3 -u "${SCRIPT_DIR}/exp_006_stage1_eval.py" \
            --model_path "$MERGED_DPO_N8" \
            --model_name "merged_dpo_N8_suppressed" \
            --data_dir "$DATA_DIR" \
            --output_dir "$EVAL_DIR" \
            --suppress_search_tokens
        echo "[OK] Merged DPO N8 suppressed eval complete"
    elif [ -d "$DPO_N8_ADAPTER" ]; then
        echo "--- Evaluating: dpo_N8_suppressed (adapter stacking) ---"
        echo "[WARN] Using adapter stacking — may OOM. Prefer merged checkpoint."
        python3 -u "${SCRIPT_DIR}/exp_006_stage1_eval.py" \
            --model_path "$MODEL_PATH" \
            --sft_adapter "$SFT_DIR" \
            --dpo_adapter "$DPO_N8_ADAPTER" \
            --model_name "dpo_N8_suppressed" \
            --data_dir "$DATA_DIR" \
            --output_dir "$EVAL_DIR" \
            --suppress_search_tokens
        echo "[OK] DPO N8 suppressed eval complete"
    else
        echo "[SKIP] No DPO N8 checkpoint found (tried ${MERGED_DPO_N8} and ${DPO_N8_ADAPTER})"
    fi
fi

# --- Summary ---
echo ""
echo "============================================"
echo "Exp 008 complete!"
echo "============================================"
echo "Results:"
ls -lh "${EVAL_DIR}"/eval_*suppressed*.json 2>/dev/null || echo "  (no results found)"
echo ""
echo "Compare with baselines in: ${PROJECT_DIR}/artifacts/exp_006_eval/"
