#!/bin/bash
# Phase 1: Diagnosis experiments
set -euo pipefail

CONFIG="configs/diagnosis.yaml"

echo "=== Phase 1a: SVD rotation analysis ==="
python -m src.diagnosis.svd_rotation --config "$CONFIG"

echo "=== Phase 1b: Mode collapse detection ==="
python -m src.diagnosis.mode_collapse --config "$CONFIG"

echo "=== Phase 1c: Information probing ==="
python -m src.diagnosis.info_probing --config "$CONFIG"

echo "=== Phase 1d: Capacity experiments ==="
python -m src.diagnosis.capacity_test --config "$CONFIG"

echo "=== Phase 1e: Prompt/CoT controls ==="
python -m src.diagnosis.prompt_cot_control --config "$CONFIG"

echo "Phase 1 complete."
