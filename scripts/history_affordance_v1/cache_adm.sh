#!/usr/bin/env bash
set -euo pipefail

SCENE_ID="${1:-room_0002}"
DATASET_ROOT="${2:-data/history_affordance_v1}"
EXP_DIR="${3:-outputs/CDM-Perceiver-ALL}"
STATS_FILE="${4:-data/Mean_Std_Cont_HumanML3D_HUMANISE_PROX_contact_cont_joints_0.8_fur.npz}"

DATASET_ROOT="$(realpath "${DATASET_ROOT}")"
EXP_DIR="$(realpath "${EXP_DIR}")"
STATS_FILE="$(realpath "${STATS_FILE}")"
CHECKPOINT="$(find "${EXP_DIR}/ckpt" -maxdepth 1 -type f -name 'model*.pt' | sort -V | tail -1)"

if [[ -z "${CHECKPOINT}" || ! -f "${CHECKPOINT}" ]]; then
  echo "No CDM checkpoint found under ${EXP_DIR}/ckpt" >&2
  exit 1
fi

python test.py hydra/job_logging=none hydra/hydra_logging=none \
  exp_dir="${EXP_DIR}" \
  seed=20260807 \
  output_dir=outputs \
  diffusion.steps=500 \
  task=contact_gen \
  model=cdm \
  model.arch=Perceiver \
  task.dataset.sigma=0.8 \
  task.dataset.name=HistoryAffordanceV1ContactMapDataset \
  +task.dataset.dataset_root="${DATASET_ROOT}" \
  +task.dataset.scene_id="${SCENE_ID}" \
  +task.dataset.stats_file="${STATS_FILE}" \
  task.evaluator.eval_metrics=[] \
  task.evaluator.k_samples=0 \
  task.evaluator.num_k_samples=0 \
  task.evaluator.eval_nbatch=1 \
  task.test.batch_size=1 \
  task.test.num_workers=0

LATEST_EVAL="$(find "${EXP_DIR}/eval" -mindepth 1 -maxdepth 1 -type d -name 'test-*' \
  -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
INFO_SET="history_affordance_v1_${SCENE_ID}"
PRED_FILE="${LATEST_EVAL}/${INFO_SET}/pred_contact/00000.npy"

if [[ ! -f "${PRED_FILE}" ]]; then
  mapfile -t PREDICTIONS < <(find "${LATEST_EVAL}" -type f -path '*/pred_contact/00000.npy' | sort)
  if [[ "${#PREDICTIONS[@]}" -ne 1 ]]; then
    echo "Expected one prediction under ${LATEST_EVAL}, found ${#PREDICTIONS[@]}" >&2
    exit 1
  fi
  PRED_FILE="${PREDICTIONS[0]}"
fi

python prepare/cache_history_affordance_v1_adm.py \
  --pred "${PRED_FILE}" \
  --dataset-root "${DATASET_ROOT}" \
  --scene-id "${SCENE_ID}" \
  --sigma 0.8 \
  --checkpoint "${CHECKPOINT}" \
  --eval-dir "${LATEST_EVAL}"

echo "[DONE] eval_dir=${LATEST_EVAL}"
