#!/usr/bin/env bash
set -euo pipefail
trap 'RUN_RC=$?; echo "[DONE] Teacher-v10 block exit=$RUN_RC; this terminal remains open"' EXIT

AMDM_ROOT="$(pwd -P)"
PACKAGE_ROOT="$(cd "$(dirname "$0")" && pwd -P)"
PATCH="$PACKAGE_ROOT"

python "$PATCH/prepare/validate_teacher_lora_v10_supervised_capacity_package.py"

MISSING=0
for FILE in fewshot_cdm_common.py fewshot_cdm_lora.py relational_teacher_v9_all_sittable_contract.py relational_teacher_v9_all_sittable_metrics.py relational_teacher_v9_lora_objective.py relational_teacher_v9_lora_preflight_contract.py relational_teacher_v9_lora_runtime.py train_fewshot_cdm.py; do
  if [ ! -f "$AMDM_ROOT/prepare/$FILE" ]; then
    echo "[STOP] missing prepare/$FILE"
    MISSING=1
  fi
done
if [ "$MISSING" -ne 0 ]; then
  echo "[STOP] dependencies are incomplete; do not run CUDA"
  exit 1
fi

cp "$PATCH"/prepare/*.py "$AMDM_ROOT/prepare/"
cd "$AMDM_ROOT"
python prepare/test_relational_teacher_v10_supervised_capacity_contract.py

SRC=data/history_affordance_relational_teacher_v7_hd
DATA=data/history_affordance_relational_teacher_v9_all_sittable_v1
V5DATA=data/history_affordance_v1
V5="$V5DATA/experiments/fewshot_cdm_chair23_bed2_whiteboard12_v5r4"
FAILED="$DATA/experiments/teacher_lora_v9812/two_scene_multiupdate_calibration_s20261028_v1/summary.json"
OUT="$DATA/experiments/teacher_lora_v10/supervised_capacity_s20261029_v1"

test -f "$FAILED" || { echo "[STOP] missing v9.8.12 failure: $FAILED"; exit 1; }
test ! -e "$OUT" || { echo "[STOP] output already exists; preserve it and use a new versioned OUT"; exit 1; }

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
ARGS=(--failed-calibration-summary "$FAILED" --source-dataset-root "$SRC" --dataset-root "$DATA" --index "$DATA/index.json" --v5-dataset-root "$V5DATA" --v5-split "$V5DATA/splits/chair23_bed2_whiteboard12_multistart24_v1.json" --stats-file data/Mean_Std_Cont_HumanML3D_HUMANISE_PROX_contact_cont_joints_0.8_fur.npz --original-checkpoint outputs/CDM-Perceiver-ALL/ckpt/model300000.pt --v5-checkpoint "$V5/fewshot_cdm.pt" --v5-evidence-report "$V5/gate0a_report.json" --output-dir "$OUT" --diffusion-steps 500 --steps 1200 --lr 0.0002 --weight-decay 0 --grad-clip 1 --lora-rank 16 --lora-alpha 16 --seed 20261029 --device cuda:0 --no-progress)
python -u prepare/run_relational_teacher_v10_supervised_capacity.py "${ARGS[@]}"
python prepare/validate_relational_teacher_v10_supervised_capacity.py --summary "$OUT/summary.json"
python prepare/summarize_relational_teacher_v10_supervised_capacity.py --summary "$OUT/summary.json"

