#!/bin/bash
# Phase 2: Distillation pipeline
set -euo pipefail

CONFIG="configs/distillation.yaml"
BASE_CONFIG="configs/base.yaml"

echo "=== Phase 2a: Generate Best-of-N candidates ==="
python -m src.distillation.bon_pairs --config "$CONFIG" --base-config "$BASE_CONFIG"

echo "=== Phase 2b: Teacher filtering ==="
python -m src.distillation.teacher_filter --config "$CONFIG" --base-config "$BASE_CONFIG"

echo "=== Phase 2c: Cold-start training ==="
python -m src.distillation.cold_start --config "$CONFIG" --base-config "$BASE_CONFIG"

echo "=== Phase 2d: Curriculum DPO training ==="
python -m src.distillation.curriculum --config "$CONFIG" --base-config "$BASE_CONFIG"

echo "Phase 2 complete."
