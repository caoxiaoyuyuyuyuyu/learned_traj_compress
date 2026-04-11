#!/bin/bash
# Phase 3: Evaluation pipeline
set -euo pipefail

CONFIG="configs/evaluation.yaml"
BASE_CONFIG="configs/base.yaml"

echo "=== Phase 3a: EM evaluation & collapse curves ==="
python -u -m src.evaluation.em_eval --config "$CONFIG" --base-config "$BASE_CONFIG"

echo "=== Phase 3b: Transfer evaluation ==="
python -u -m src.evaluation.transfer_eval --config "$CONFIG" --base-config "$BASE_CONFIG"

echo "=== Phase 3c: Ablation studies ==="
python -u -m src.evaluation.ablation --config "$CONFIG" --base-config "$BASE_CONFIG"

echo "=== Phase 3d: Mechanistic analysis ==="
python -u -m src.evaluation.mechanistic --config "$CONFIG" --base-config "$BASE_CONFIG"

echo "Phase 3 complete."
