#!/bin/bash
export HF_HOME=/root/autodl-tmp/.hf_cache
export HF_HUB_DISABLE_XET=1
source /root/miniconda3/bin/activate
cd /root/autodl-tmp/learned_traj_compress

WORKDIR=artifacts/exp_016_workdir
mkdir -p $WORKDIR
LOG=$WORKDIR/run_all_phases.log
ERR_FLAG=$WORKDIR/SANITY_FAIL

SCRIPT="scripts/exp_016_passage_redaction.py"
PY=/root/miniconda3/bin/python
MODEL=/root/autodl-tmp/models/Qwen2.5-3B
ADAPTER=checkpoints/exp_006_sft_shared
ORACLE=artifacts/phase1d_v2_data/raw_trajectories_N8.json
N=125

# Clean any stale fail flag / done flag from previous attempt
rm -f $ERR_FLAG $WORKDIR/ALL_PHASES_DONE

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] exp_016 run_all_phases START (python=$($PY --version 2>&1))" >> $LOG

run_phase() {
  local label="$1"; local cond="$2"; local out="$3"
  echo "[$(date -u +%H:%M:%SZ)] $label ($cond)" >> $LOG
  $PY $SCRIPT --phase eval --condition $cond \
    --model_path $MODEL --sft_adapter $ADAPTER \
    --oracle_source $ORACLE \
    --output_json $out \
    --n_test $N --split phase1d_v2 >> $LOG 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(date -u +%H:%M:%SZ)] PHASE_FAIL $label rc=$rc" >> $LOG
    exit $rc
  fi
}

# Phase 0.5: no_api baseline
run_phase "Phase 0.5 no_api" no_api $WORKDIR/no_api_eval.json

# Phase 1: oracle_full (sanity)
run_phase "Phase 1 oracle_full" oracle_full $WORKDIR/oracle_full_eval.json

$PY - << 'PYEOF' >> $LOG 2>&1
import json, sys
d = json.load(open("artifacts/exp_016_workdir/oracle_full_eval.json"))
ep = d["em_partial"]
print(f"[gate] oracle_full em_partial={ep} expected=0.498 tol=0.02")
if abs(ep - 0.498) > 0.02:
    open("artifacts/exp_016_workdir/SANITY_FAIL", "w").write(
        f"SANITY_FAIL oracle_full em_partial={ep} outside 0.498 +/- 0.02\n")
    sys.exit(42)
PYEOF
if [ -f $ERR_FLAG ]; then
  echo "[$(date -u +%H:%M:%SZ)] SANITY GATE FAILED — abort, blocker_report to Director" >> $LOG
  exit 42
fi

# Phase 2a: oracle_redacted_full
run_phase "Phase 2a oracle_redacted_full" oracle_redacted_full $WORKDIR/oracle_redacted_full_eval.json

# Phase 2b: oracle_redacted_half
run_phase "Phase 2b oracle_redacted_half" oracle_redacted_half $WORKDIR/oracle_redacted_half_eval.json

# Phase 3: aggregate + bootstrap
echo "[$(date -u +%H:%M:%SZ)] Phase 3 aggregate" >> $LOG
$PY $SCRIPT --phase aggregate --output_dir $WORKDIR >> $LOG 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ALL PHASES DONE" >> $LOG
touch $WORKDIR/ALL_PHASES_DONE
