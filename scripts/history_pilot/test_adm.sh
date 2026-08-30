#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="${1:-outputs/CDM-Perceiver-ALL}"
PILOT_DIR="${2:-data/history_affordance_pilot/pilot_0001}"
STATS_FILE="${3:-data/Mean_Std_Cont_HumanML3D_HUMANISE_PROX_contact_cont_joints_0.8_fur.npz}"

EXP_DIR="$(realpath "${EXP_DIR}")"
PILOT_DIR="$(realpath "${PILOT_DIR}")"
STATS_FILE="$(realpath "${STATS_FILE}")"

python prepare/history_pilot_adm_input.py --pilot-dir "${PILOT_DIR}"

python test.py hydra/job_logging=none hydra/hydra_logging=none \
  exp_dir="${EXP_DIR}" \
  seed=20260807 \
  output_dir=outputs \
  diffusion.steps=500 \
  task=contact_gen \
  model=cdm \
  model.arch=Perceiver \
  task.dataset.sigma=0.8 \
  task.dataset.name=HistoryPilotContactMapDataset \
  +task.dataset.pilot_dir="${PILOT_DIR}" \
  +task.dataset.pilot_stats_file="${STATS_FILE}" \
  task.evaluator.eval_metrics=[] \
  task.evaluator.k_samples=0 \
  task.evaluator.num_k_samples=0 \
  task.evaluator.eval_nbatch=1 \
  task.test.batch_size=1 \
  task.test.num_workers=0

LATEST_EVAL="$(find "${EXP_DIR}/eval" -mindepth 1 -maxdepth 1 -type d -name 'test-*' \
  -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
PRED_FILE="${LATEST_EVAL}/history_pilot/pred_contact/00000.npy"

python prepare/validate_history_pilot_adm.py \
  --pred "${PRED_FILE}" \
  --pilot-dir "${PILOT_DIR}" \
  --sigma 0.8

echo "[DONE] eval_dir=${LATEST_EVAL}"
