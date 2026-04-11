#!/bin/bash
# Phase 1d: Resume from Stage 2b (1.5B DPO) after fixing DPOConfig bug
# Stage 1 (data gen) and Stage 2a (1.5B SFT) already completed.

set -euo pipefail

MODELS_ROOT="/root/autodl-tmp/models"
ARTIFACTS="/root/autodl-tmp/learned_traj_compress/artifacts"
DATA_DIR="${ARTIFACTS}/phase1d_data"
EVAL_DIR="${ARTIFACTS}/phase1d_eval"

BASE_1_5B="${MODELS_ROOT}/Qwen2.5-1.5B"
BASE_3B="${MODELS_ROOT}/Qwen2.5-3B"

OUT_1_5B_SFT="${MODELS_ROOT}/phase1d/Qwen2.5-1.5B-SFT"
OUT_1_5B_DPO="${MODELS_ROOT}/phase1d/Qwen2.5-1.5B-DPO"
OUT_3B_SFT="${MODELS_ROOT}/phase1d/Qwen2.5-3B-SFT"
OUT_3B_DPO="${MODELS_ROOT}/phase1d/Qwen2.5-3B-DPO"

cd /root/autodl-tmp/learned_traj_compress

echo "================================================"
echo "Phase 1d: RESUME from Stage 2b (DPOConfig fix)"
echo "Start: $(date)"
echo "================================================"
echo "Stage 1 (data gen): ALREADY DONE"
echo "Stage 2a (1.5B SFT): ALREADY DONE"

# 2b: 1.5B DPO (retry)
echo ""
echo "--- 2b: 1.5B DPO (retry) ---"
echo "Start: $(date)"
python scripts/phase1d_train.py \
    --base_dir "${BASE_1_5B}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUT_1_5B_DPO}" \
    --method dpo
echo "2b done: $(date)"

# 2c: 3B SFT
echo "--- 2c: 3B SFT ---"
echo "Start: $(date)"
python scripts/phase1d_train.py \
    --base_dir "${BASE_3B}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUT_3B_SFT}" \
    --method sft
echo "2c done: $(date)"

# 2d: 3B DPO
echo "--- 2d: 3B DPO ---"
echo "Start: $(date)"
python scripts/phase1d_train.py \
    --base_dir "${BASE_3B}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUT_3B_DPO}" \
    --method dpo
echo "2d done: $(date)"

# Stage 3: Evaluation
echo ""
echo "=== STAGE 3: Evaluation ==="
echo "Start: $(date)"
python -u scripts/phase1d_evaluate.py \
    --models \
        "1.5B-SFT:${OUT_1_5B_SFT}" \
        "1.5B-DPO:${OUT_1_5B_DPO}" \
        "3B-SFT:${OUT_3B_SFT}" \
        "3B-DPO:${OUT_3B_DPO}" \
    --base_dirs \
        "${BASE_1_5B}" \
        "${BASE_1_5B}" \
        "${BASE_3B}" \
        "${BASE_3B}" \
    --output_dir "${EVAL_DIR}"
echo "Stage 3 done: $(date)"

echo ""
echo "================================================"
echo "Phase 1d COMPLETE: $(date)"
echo "Results: ${EVAL_DIR}/phase1d_results.json"
echo "================================================"
