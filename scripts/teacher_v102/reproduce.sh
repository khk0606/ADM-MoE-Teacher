#!/usr/bin/env bash
set -euo pipefail
trap 'RUN_RC=$?; echo "[DONE] Teacher-v10.2 reproduction exit=$RUN_RC"' EXIT

ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"
cd "$ROOT"

python -c 'import clip, numpy, pytorch3d, torch; print("[PASS] core ML imports"); assert torch.cuda.is_available(), "Teacher-v10.2 requires an NVIDIA CUDA device"'
bash scripts/teacher_v102/fetch_assets.sh
python research_history/teacher_lora_packages/teacher_lora_v102_dense_instance_patch/prepare/validate_teacher_lora_v102_dense_instance_package.py
python prepare/test_relational_teacher_v102_dense_instance_contract.py

DATA="data/history_affordance_relational_teacher_v9_all_sittable_v1"
FAILED="$DATA/experiments/teacher_lora_v101/fullfield_supervision_s20261030_v1/summary.json"
OUT="$DATA/experiments/teacher_lora_v102/dense_instance_supervision_s20261031_v1"

python prepare/validate_relational_teacher_v101_fullfield_supervision.py --summary "$FAILED"
if [[ -e "$OUT" ]]; then
  echo "[STOP] Teacher-v10.2 output already exists; preserve it before reproducing"
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
ARGS=(--failed-v101-summary "$FAILED" --output-dir "$OUT" --device cuda:0 --no-progress)
python -u prepare/run_relational_teacher_v102_dense_instance_supervision.py "${ARGS[@]}"
python prepare/validate_relational_teacher_v102_dense_instance_supervision.py --summary "$OUT/summary.json"
python prepare/summarize_relational_teacher_v102_dense_instance_supervision.py --summary "$OUT/summary.json"

echo "[NEXT] bash scripts/teacher_v102/view.sh"
