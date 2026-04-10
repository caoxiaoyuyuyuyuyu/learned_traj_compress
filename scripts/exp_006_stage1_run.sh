#!/bin/bash
# Exp 006 Stage 1: Full pipeline — SFT → DPO per-N → Evaluation (D019)
#
# DEFAULT (main paper result, addresses reviewer Warning 1 + 2):
#   - SFT: natural pooled (all perfect SFT samples from N=2/4/8 combined) with
#          per-N train loss logging to diagnose N=2 domination.
#   - DPO: **matched-sample** per-N (each N subsamples to min(pair count) so
#          cross-N EM comparison is fair).
#
# Ablation knobs (env vars):
#   BALANCED_SFT=1  → run balanced SFT (down-sample N=2 to ~BALANCED_N2_TARGET)
#                     as a Warning-2 ablation. Writes to exp_006_sft_shared_balanced.
#   FULL_DPO=1      → run full-data DPO (all pairs per N) as a Warning-1 upper-bound
#                     reference. Writes to exp_006_dpo_N{2,4,8}_full.
#
# Usage:
#   bash scripts/exp_006_stage1_run.sh                       # main: matched DPO
#   BALANCED_SFT=1 bash scripts/exp_006_stage1_run.sh        # balanced SFT ablation
#   FULL_DPO=1 bash scripts/exp_006_stage1_run.sh            # full-data DPO ablation
#   SKIP_SFT=1 bash scripts/exp_006_stage1_run.sh            # skip SFT
#   SKIP_N2=1 bash scripts/exp_006_stage1_run.sh             # skip DPO N=2
#   SKIP_SFT=1 SKIP_N2=1 bash scripts/exp_006_stage1_run.sh # retrain N4/N8 only
#   SKIP_TRAIN=1 bash scripts/exp_006_stage1_run.sh          # eval only
set -euo pipefail

# --- Paths ---
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-3B}"
DATA_DIR="${DATA_DIR:-/root/autodl-tmp/learned_traj_compress/artifacts/phase1d_v2_data}"
CKPT_DIR="${CKPT_DIR:-/root/autodl-tmp/learned_traj_compress/checkpoints}"
EVAL_DIR="${EVAL_DIR:-/root/autodl-tmp/learned_traj_compress/artifacts/exp_006_eval}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SPLIT_SEED="${SPLIT_SEED:-42}"

# --- Ablation flags ---
BALANCED_SFT="${BALANCED_SFT:-0}"
BALANCED_N2_TARGET="${BALANCED_N2_TARGET:-50}"
FULL_DPO="${FULL_DPO:-0}"
SKIP_N2="${SKIP_N2:-0}"

# --- DPO hyperparams (D028: tuned for small-data ~110 train) ---
DPO_EPOCHS="${DPO_EPOCHS:-1}"
DPO_LR="${DPO_LR:-2e-5}"
DPO_BETA="${DPO_BETA:-0.05}"
DPO_MAX_GRAD_NORM="${DPO_MAX_GRAD_NORM:-1.0}"

# Checkpoint dir names reflect ablation mode so main and ablation runs coexist
if [ "$BALANCED_SFT" = "1" ]; then
    SFT_DIR="${CKPT_DIR}/exp_006_sft_shared_balanced"
else
    SFT_DIR="${CKPT_DIR}/exp_006_sft_shared"
fi
if [ "$FULL_DPO" = "1" ]; then
    DPO_SUFFIX="_full"
else
    DPO_SUFFIX=""
fi

export WANDB_PROJECT="learned_traj_compress"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/.hf_cache}"

echo "============================================"
echo "Exp 006 Stage 1: SFT + DPO + Eval Pipeline"
echo "============================================"
echo "Model:        ${MODEL_PATH}"
echo "Data:         ${DATA_DIR}"
echo "Checkpoints:  ${CKPT_DIR}"
echo "Eval output:  ${EVAL_DIR}"
echo "Split seed:   ${SPLIT_SEED}"
echo "Balanced SFT: ${BALANCED_SFT} (N=2 target=${BALANCED_N2_TARGET})"
echo "Full DPO:     ${FULL_DPO}"
echo "Skip N=2:     ${SKIP_N2}"
echo "DPO epochs:   ${DPO_EPOCHS}"
echo "DPO lr:       ${DPO_LR}"
echo "DPO beta:     ${DPO_BETA}"
echo "DPO grad_norm:${DPO_MAX_GRAD_NORM}"
echo "SFT dir:      ${SFT_DIR}"
echo ""

# --- Verify data exists ---
echo "Checking data files..."
for N in 2 4 8; do
    sft_file="${DATA_DIR}/sft_data_N${N}.json"
    dpo_file="${DATA_DIR}/dpo_data_N${N}.json"
    raw_file="${DATA_DIR}/raw_trajectories_N${N}.json"
    if [ -f "$sft_file" ]; then
        count=$(python3 -c "import json; print(len(json.load(open('$sft_file'))))")
        echo "  sft_data_N${N}.json: ${count} samples"
    else
        echo "  WARNING: ${sft_file} not found"
    fi
    if [ -f "$dpo_file" ]; then
        count=$(python3 -c "import json; print(len(json.load(open('$dpo_file'))))")
        echo "  dpo_data_N${N}.json: ${count} pairs"
    else
        echo "  WARNING: ${dpo_file} not found"
    fi
    if [ -f "$raw_file" ]; then
        echo "  raw_trajectories_N${N}.json: exists (needed for eval)"
    else
        echo "  WARNING: ${raw_file} not found (eval will skip N=${N})"
    fi
done
echo ""

if [ "${SKIP_TRAIN:-0}" = "1" ]; then
    echo "[SKIP] All training skipped (SKIP_TRAIN=1), jumping to eval"
else

# ============================================
# Step 1: Shared SFT Training
# ============================================
if [ "${SKIP_SFT:-0}" = "1" ]; then
    echo "[SKIP] SFT training skipped (SKIP_SFT=1)"
    if [ ! -d "$SFT_DIR" ]; then
        echo "ERROR: SFT checkpoint not found at ${SFT_DIR}"
        exit 1
    fi
else
    echo "============================================"
    echo "Step 1: Shared SFT Training (balanced=${BALANCED_SFT})"
    echo "============================================"
    SFT_EXTRA_ARGS=()
    if [ "$BALANCED_SFT" = "1" ]; then
        SFT_EXTRA_ARGS+=(--balanced_sft --balanced_n2_target "$BALANCED_N2_TARGET")
    fi
    python3 "${SCRIPT_DIR}/exp_006_stage1_sft.py" \
        --model_path "$MODEL_PATH" \
        --data_dir "$DATA_DIR" \
        --output_dir "$SFT_DIR" \
        --num_epochs 3 \
        --lr 2e-4 \
        --batch_size 1 \
        --grad_accum 8 \
        --split_seed "$SPLIT_SEED" \
        "${SFT_EXTRA_ARGS[@]}"

    echo ""
    echo "[OK] SFT training complete. Adapter at: ${SFT_DIR}"
    echo ""
fi

# ============================================
# Step 2: Per-N DPO Training (default: matched-sample, Warning 1 fix)
# ============================================
for N in 2 4 8; do
    # Skip N=2 DPO when retrain-only mode (D028: N2 already done, only retrain N4/N8)
    if [ "$N" = "2" ] && [ "$SKIP_N2" = "1" ]; then
        echo "[SKIP] DPO N=2 skipped (SKIP_N2=1)"
        continue
    fi

    DPO_DIR="${CKPT_DIR}/exp_006_dpo_N${N}${DPO_SUFFIX}"
    DPO_DATA="${DATA_DIR}/dpo_data_N${N}.json"

    if [ ! -f "$DPO_DATA" ]; then
        echo "[SKIP] DPO N=${N}: data file not found (${DPO_DATA})"
        continue
    fi

    echo "============================================"
    echo "Step 2.${N}: DPO Training (N=${N}, full=${FULL_DPO})"
    echo "============================================"
    DPO_EXTRA_ARGS=()
    if [ "$FULL_DPO" = "1" ]; then
        DPO_EXTRA_ARGS+=(--full_data)
    fi
    # D028: max_grad_norm only passed when set (avoids no-clip default)
    if [ -n "$DPO_MAX_GRAD_NORM" ]; then
        DPO_EXTRA_ARGS+=(--max_grad_norm "$DPO_MAX_GRAD_NORM")
    fi
    python3 "${SCRIPT_DIR}/exp_006_stage1_dpo.py" \
        --model_path "$MODEL_PATH" \
        --sft_adapter "$SFT_DIR" \
        --data_dir "$DATA_DIR" \
        --n_objectives "$N" \
        --output_dir "$DPO_DIR" \
        --num_epochs "$DPO_EPOCHS" \
        --lr "$DPO_LR" \
        --beta "$DPO_BETA" \
        --batch_size 1 \
        --grad_accum 8 \
        --split_seed "$SPLIT_SEED" \
        "${DPO_EXTRA_ARGS[@]}"

    echo ""
    echo "[OK] DPO N=${N} complete. Adapter at: ${DPO_DIR}"
    echo ""
done

fi  # end SKIP_TRAIN

# ============================================
# Step 3: Evaluation
# ============================================
echo "============================================"
echo "Step 3: Evaluation on held-out test prompts"
echo "============================================"
mkdir -p "$EVAL_DIR"

# Tag eval run names with ablation mode suffix so results don't collide
SFT_EVAL_NAME="sft_shared"
if [ "$BALANCED_SFT" = "1" ]; then
    SFT_EVAL_NAME="sft_shared_balanced"
fi

# Eval 1: Shared SFT baseline
if [ -d "$SFT_DIR" ]; then
    echo ""
    echo "--- Evaluating: ${SFT_EVAL_NAME} ---"
    python3 "${SCRIPT_DIR}/exp_006_stage1_eval.py" \
        --model_path "$MODEL_PATH" \
        --sft_adapter "$SFT_DIR" \
        --model_name "$SFT_EVAL_NAME" \
        --data_dir "$DATA_DIR" \
        --output_dir "$EVAL_DIR" \
        --split_seed "$SPLIT_SEED"
fi

# Eval 2-4: DPO per-N
for N in 2 4 8; do
    DPO_DIR="${CKPT_DIR}/exp_006_dpo_N${N}${DPO_SUFFIX}"
    DPO_EVAL_NAME="dpo_N${N}${DPO_SUFFIX}"
    if [ "$BALANCED_SFT" = "1" ]; then
        DPO_EVAL_NAME="${DPO_EVAL_NAME}_balSFT"
    fi
    if [ -d "$DPO_DIR" ]; then
        echo ""
        echo "--- Evaluating: ${DPO_EVAL_NAME} ---"
        python3 "${SCRIPT_DIR}/exp_006_stage1_eval.py" \
            --model_path "$MODEL_PATH" \
            --sft_adapter "$SFT_DIR" \
            --dpo_adapter "$DPO_DIR" \
            --model_name "$DPO_EVAL_NAME" \
            --data_dir "$DATA_DIR" \
            --output_dir "$EVAL_DIR" \
            --split_seed "$SPLIT_SEED"
    else
        echo "[SKIP] Eval ${DPO_EVAL_NAME}: checkpoint not found (${DPO_DIR})"
    fi
done

# ============================================
# Summary
# ============================================
echo ""
echo "============================================"
echo "All Stage 1 complete!"
echo "============================================"
echo "Checkpoints:"
echo "  SFT:        ${SFT_DIR}"
for N in 2 4 8; do
    echo "  DPO N=${N}:   ${CKPT_DIR}/exp_006_dpo_N${N}${DPO_SUFFIX}"
done
echo ""
echo "Eval results: ${EVAL_DIR}/"
ls -lh "${EVAL_DIR}"/eval_*.json 2>/dev/null || echo "  (no eval results found)"
