#!/bin/bash
# Phase 1d v2: 3B-only SFT→DPO Contrast Experiment
# Run all stages sequentially on a single GPU.
#
# Usage:
#   bash scripts/phase1d_run_all.sh [--dry_run]
#
# Changes from v1:
#   - 1.5B removed (3B-only)
#   - DPO inits from SFT checkpoint (--sft_checkpoint)
#   - LoRA rank=32, alpha=64 (default in train script)
#   - Data dir updated to phase1d_v2_data

set -euo pipefail

# --- Paths ---
MODELS_ROOT="/root/autodl-tmp/models"
ARTIFACTS="/root/autodl-tmp/learned_traj_compress/artifacts"
DATA_DIR="${ARTIFACTS}/phase1d_v2_data"
EVAL_DIR="${ARTIFACTS}/phase1d_v2_eval"

BASE_3B="${MODELS_ROOT}/Qwen2.5-3B"
RL_DIR="${MODELS_ROOT}/MEM1"

OUT_3B_SFT="${MODELS_ROOT}/phase1d_v2/Qwen2.5-3B-SFT"
OUT_3B_DPO="${MODELS_ROOT}/phase1d_v2/Qwen2.5-3B-DPO"

DRY_RUN_FLAG=""
if [[ "${1:-}" == "--dry_run" ]]; then
    DRY_RUN_FLAG="--dry_run"
    echo "*** DRY RUN MODE ***"
fi

cd /root/autodl-tmp/learned_traj_compress

echo "================================================"
echo "Phase 1d v2: 3B SFT→DPO Contrast"
echo "Start: $(date)"
echo "================================================"

# --- Stage 1: SFT Training ---
echo ""
echo "=== STAGE 1: 3B SFT ==="
echo "Start: $(date)"
python scripts/phase1d_train.py \
    --base_dir "${BASE_3B}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUT_3B_SFT}" \
    --method sft
echo "Stage 1 done: $(date)"

# --- Stage 2: DPO Training (from SFT checkpoint) ---
echo ""
echo "=== STAGE 2: 3B DPO (init from SFT) ==="
echo "Start: $(date)"
python scripts/phase1d_train.py \
    --base_dir "${BASE_3B}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUT_3B_DPO}" \
    --method dpo \
    --sft_checkpoint "${OUT_3B_SFT}"
echo "Stage 2 done: $(date)"

# --- Stage 3: Evaluation ---
echo ""
echo "=== STAGE 3: Evaluation ==="
echo "Start: $(date)"
python -u scripts/phase1d_evaluate.py \
    --models \
        "3B-SFT:${OUT_3B_SFT}" \
        "3B-DPO:${OUT_3B_DPO}" \
    --base_dirs \
        "${BASE_3B}" \
        "${BASE_3B}" \
    --output_dir "${EVAL_DIR}" \
    ${DRY_RUN_FLAG:+--dry_run}
echo "Stage 3 done: $(date)"

echo ""
echo "================================================"
echo "Phase 1d v2 COMPLETE: $(date)"
echo "Results: ${EVAL_DIR}/phase1d_results.json"
echo "================================================"
