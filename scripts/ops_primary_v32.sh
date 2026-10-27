#!/usr/bin/env bash
# ============================================================================
# ops_primary_v32.sh -- v32 FRESH RUN: 10 repeats x 2 seeds, whole-dataset
# coverage, four external cohorts in-run. Derived verbatim from
# ops_primary_regen_29ag.sh (frozen recipe) with only names changed
# (scratch/_primary_v32, ckpt/_primary_v32, gate_v32.log,
# results_v32_defog.json) and SEEDS/REPEATS defaults.
#
# SMOKE: SEEDS=1 REPEATS=1 EPOCHS=3 bash src/training/scripts/ops_primary_v32.sh
# FULL:  nohup bash src/training/scripts/ops_primary_v32.sh > v32_run.log 2>&1 &
# ============================================================================
set -u
set -o pipefail
PY=/home/danish/miniconda3/envs/mamba_env/bin/python
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRATCH="$ROOT/_primary_v32"
mkdir -p "$SCRATCH" || exit 8
cd "$SCRATCH" || { echo ABORT_CD; exit 8; }
export PYTHONPATH="$ROOT/models:$ROOT/src/evaluation/calibration:${PYTHONPATH:-}"
export NEUROMAMBA_CKPT_DIR="$ROOT/_primary_v32/ckpt"
mkdir -p "$NEUROMAMBA_CKPT_DIR"
[ -x "$PY" ] || { echo ABORT_NO_PYTHON; exit 8; }
if pgrep -f neuromamba_v25_sota.py >/dev/null 2>&1; then echo ABORT_PROCESS_ALIVE; exit 8; fi
SEEDS=${SEEDS:-2}
REPEATS=${REPEATS:-10}
EPOCHS=${EPOCHS:-80}
FAMILIES=${FAMILIES:-"defog"}

echo $$ > gate_v32.pid

# --- storm-era stability ---
export NEUROMAMBA_NOTYPE_FOG_W=1.0
export NEUROMAMBA_GRAD_GUARD_DYN_CAP=1000
export NEUROMAMBA_GRAD_GUARD_WARMUP_EPOCHS=10
export NEUROMAMBA_GRAD_GUARD_FROZEN_FLOOR=700
export NEUROMAMBA_GRAD_GUARD_MAX_ABORT_FRAC=0.25
export NEUROMAMBA_GRAD_GUARD_BAD_EPOCH_PATIENCE=4
# --- SH/Walk enhancement stack (v25.58e + v25.59f/g) ---
export NEUROMAMBA_TOKEN_POS_WEIGHT_MODE=rare
export NEUROMAMBA_TOKEN_POS_WEIGHT_CAP=8
export NEUROMAMBA_DOMAIN_DANN=0.05
export NEUROMAMBA_HEAD_PRIOR_INIT=1
export NEUROMAMBA_WALK_CROP_PROB=0.40
export NEUROMAMBA_TOKEN_ASL=1
export NEUROMAMBA_WALK_HARDNEG_W=2
export NEUROMAMBA_BOOTSTRAP_N=50
export NEUROMAMBA_SMALL_CALIB_EP_F1_DEPLOY=1
export NEUROMAMBA_CALIB_USE_OOF_POOL=1
export NEUROMAMBA_DEPLOY_FARH_SAFETY=0.75
export NEUROMAMBA_PRIOR_CLAMP_DELTA=0.15
export NEUROMAMBA_MLLS_CORRECT=1
export NEUROMAMBA_EXTERNAL_PRIOR_MATCH=1
export NEUROMAMBA_SELECT_MAP_BLEND=0.3
export NEUROMAMBA_WINNER_TOKEN_LOSS=1
export NEUROMAMBA_SUBJECT_FIRST_SAMPLER=1
export NEUROMAMBA_SAMPLER_QUOTA=0.20,0.20,0.10,0.25,0.25
export NEUROMAMBA_TURN_PLAIN=0
export NEUROMAMBA_EPOCH_T_FIXED=1
export NEUROMAMBA_SUBTYPE_TEMPERATURE=0
export NEUROMAMBA_MASK_DEFOG_SH=0
export NEUROMAMBA_SH_CONTRASTIVE=0
export NEUROMAMBA_SH_SUBJECT_CAP=0
export NEUROMAMBA_GATE_MODE=multiply
export NEUROMAMBA_GATE_FLOOR=0.25
export NEUROMAMBA_GATE_SH_COND=1
export NEUROMAMBA_GATE_MASK_SH=1
export NEUROMAMBA_TLVMC_ROOT="/mnt/d/Installed/Datasets/tlvmc-parkinsons-freezing-gait-prediction"
export NEUROMAMBA_DAPHNET_ROOT=/mnt/d/Installed/Datasets/daphnet
export NEUROMAMBA_XUANWU_ROOT="/mnt/d/Installed/Datasets/r8gmbtv7w2-3/Filtered Data"
export NEUROMAMBA_WEARGAIT_ROOT=/mnt/d/Installed/Datasets/synapse
export NEUROMAMBA_FOGSTAR_ROOT="/mnt/d/Installed/Datasets/FoG-STAR"
# --- primary arm: hazard block absent by design ---
unset FOGBENCH_TLS FOGBENCH_TLS_W FOGBENCH_TLS_PARAMS
unset NEUROMAMBA_NP_CONSTRAINT NEUROMAMBA_NP_ALPHA NEUROMAMBA_NP_START
unset NEUROMAMBA_HAZARD_STREAM
unset NEUROMAMBA_LORA_RANK NEUROMAMBA_LORA_ALPHA NEUROMAMBA_LORA_DROPOUT NEUROMAMBA_LORA_FREEZE_BACKBONE
unset NEUROMAMBA_WINNER_D NEUROMAMBA_WINNER_HEADS NEUROMAMBA_WINNER_DEPTH
unset NEUROMAMBA_WINNER_PATCH NEUROMAMBA_TRAIN_WINDOW
unset NEUROMAMBA_WARMUP_EPOCHS NEUROMAMBA_SUB_W_RAMP_EPOCHS

run_family () {
  local fam=$1 out=$2 log=$3
  local model_for_fam="winner"
  if [ "$fam" = "defog" ]; then
    model_for_fam="sequence"
    export NEUROMAMBA_WARM_START_FAMILY=tdcsfog
    export NEUROMAMBA_WARM_EPOCHS=15
    export NEUROMAMBA_TURN_PLAIN=1
    export NEUROMAMBA_WARM_REPLAY_REPEAT=2
  else
    export NEUROMAMBA_TURN_PLAIN=0
    unset NEUROMAMBA_WARM_REPLAY_REPEAT
    unset NEUROMAMBA_WARM_START_FAMILY NEUROMAMBA_WARM_EPOCHS
  fi
  echo "[v32] === family $fam (model=$model_for_fam, seeds=$SEEDS, repeats=$REPEATS) ==="
  "$PY" "$ROOT/models/neuromamba_v25_sota.py" \
    --model "$model_for_fam" --datasets tlvmc --families "$fam" \
    --split-mode shared_pooled --family-shared-val \
    --n-splits 1 --n-seeds "$SEEDS" --n-repeats "$REPEATS" --epochs "$EPOCHS" \
    --run-baseline-comparator \
    --external daphnet xuanwu weargait fogstar \
    --resume \
    --output "$out" 2>&1 | tee "$log"
  echo "[v32] === family $fam done -> $out ==="
}

for fam in $FAMILIES; do
  case "$fam" in
    defog) run_family defog results_v32_defog.json gate_v32.log ;;
    *) echo "[v32] unknown family '$fam'" ;;
  esac
done
[ -f "results_v32_defog.json" ] || { echo "[v32] RESULTS JSON MISSING -- STOP"; exit 9; }
echo "[v32] ALL DONE."
