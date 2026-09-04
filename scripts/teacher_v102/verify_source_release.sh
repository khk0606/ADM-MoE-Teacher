#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"
cd "$ROOT"
PACKAGE_ROOT="research_history/teacher_lora_packages"

required_files=(
  prepare/fewshot_cdm_common.py
  prepare/fewshot_cdm_lora.py
  prepare/train_fewshot_cdm.py
  prepare/relational_teacher_v9_all_sittable_contract.py
  prepare/relational_teacher_v9_all_sittable_metrics.py
  prepare/relational_teacher_v9_lora_objective.py
  prepare/relational_teacher_v9_lora_preflight_contract.py
  prepare/relational_teacher_v9_lora_runtime.py
  prepare/relational_teacher_v10_affordance_viewer_common.py
  prepare/relational_teacher_v10_supervised_capacity_contract.py
  prepare/relational_teacher_v10_supervised_objective.py
  prepare/run_relational_teacher_v10_supervised_capacity.py
  prepare/validate_relational_teacher_v10_supervised_capacity.py
  prepare/visualize_relational_teacher_v10_supervised_capacity_viser.py
  prepare/relational_teacher_v101_fullfield_contract.py
  prepare/relational_teacher_v101_fullfield_objective.py
  prepare/run_relational_teacher_v101_fullfield_supervision.py
  prepare/validate_relational_teacher_v101_fullfield_supervision.py
  prepare/relational_teacher_v102_dense_instance_contract.py
  prepare/relational_teacher_v102_dense_instance_objective.py
  prepare/run_relational_teacher_v102_dense_instance_supervision.py
  prepare/validate_relational_teacher_v102_dense_instance_supervision.py
  prepare/summarize_relational_teacher_v102_dense_instance_supervision.py
  prepare/test_relational_teacher_v102_dense_instance_contract.py
  prepare/visualize_relational_teacher_v102_dense_instance_viser.py
  prepare/export_relational_teacher_v102_public_bundle.py
  prepare/visualize_relational_teacher_v102_public_viser.py
  prepare/teacher_v102_portable_assets.py
  prepare/fetch_teacher_v102_assets.py
  prepare/package_teacher_v102_assets.py
  prepare/test_teacher_v102_portable_assets.py
  prepare/test_package_teacher_v102_assets.py
  "$PACKAGE_ROOT/teacher_lora_v10_supervised_capacity_patch/run_training.sh"
  "$PACKAGE_ROOT/teacher_lora_v101_fullfield_supervision_patch/run_training.sh"
  "$PACKAGE_ROOT/teacher_lora_v102_dense_instance_patch/run_training.sh"
  "$PACKAGE_ROOT/teacher_lora_v102_dense_instance_patch/run_viewer.sh"
  docs/TEACHER_EXPERIMENT_HISTORY.md
  docs/results/teacher_v102/metrics_public.json
  docs/assets/teacher_v102/teacher_v102_step1000_room0101_watch_generation0.png
  scripts/teacher_v102/export_public_result.sh
  scripts/teacher_v102/bootstrap.sh
  scripts/teacher_v102/fetch_assets.sh
  scripts/teacher_v102/reproduce.sh
  scripts/teacher_v102/view.sh
  scripts/teacher_v102/package_assets.sh
  scripts/teacher_v102/publish_assets.sh
  environment.teacher-v102.yml
  docs/TEACHER_V102_RELEASE.md
)

missing=0
for path in "${required_files[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "[FAIL] missing $path"
    missing=1
  fi
done
if [[ "$missing" -ne 0 ]]; then
  exit 1
fi
echo "[PASS] runnable Teacher-v10.2 source inventory"

package_count=$(find "$PACKAGE_ROOT" -mindepth 1 -maxdepth 1 -type d \( -name 'teacher_lora_v9*_patch' -o -name 'teacher_lora_v10*_patch' \) | wc -l | tr -d ' ')
if [[ "$package_count" -ne 34 ]]; then
  echo "[FAIL] expected 34 Teacher-v9-v10.3.1 packages; found $package_count"
  exit 1
fi
echo "[PASS] 34 versioned Teacher source packages"

python "$PACKAGE_ROOT/teacher_lora_v10_supervised_capacity_patch/prepare/validate_teacher_lora_v10_supervised_capacity_package.py"
python "$PACKAGE_ROOT/teacher_lora_v101_fullfield_supervision_patch/prepare/validate_teacher_lora_v101_fullfield_package.py"
python "$PACKAGE_ROOT/teacher_lora_v102_dense_instance_patch/prepare/validate_teacher_lora_v102_dense_instance_package.py"
python "$PACKAGE_ROOT/teacher_lora_v103_onpolicy_response_patch/prepare/validate_teacher_lora_v103_onpolicy_response_package.py"
python "$PACKAGE_ROOT/teacher_lora_v1031_onpolicy_calibration6_patch/prepare/validate_teacher_lora_v1031_onpolicy_calibration6_package.py"
python "$PACKAGE_ROOT/teacher_lora_v1031_affordance_viewer_patch/prepare/validate_teacher_lora_v1031_affordance_viewer_package.py"

python -m py_compile \
  prepare/relational_teacher_v9_all_sittable_contract.py \
  prepare/relational_teacher_v9_all_sittable_metrics.py \
  prepare/relational_teacher_v9_lora_objective.py \
  prepare/relational_teacher_v10_supervised_capacity_contract.py \
  prepare/relational_teacher_v101_fullfield_contract.py \
  prepare/relational_teacher_v102_dense_instance_contract.py \
  prepare/relational_teacher_v102_dense_instance_objective.py \
  prepare/run_relational_teacher_v102_dense_instance_supervision.py \
  prepare/validate_relational_teacher_v102_dense_instance_supervision.py \
  prepare/visualize_relational_teacher_v102_dense_instance_viser.py \
  prepare/export_relational_teacher_v102_public_bundle.py \
  prepare/visualize_relational_teacher_v102_public_viser.py \
  prepare/teacher_v102_portable_assets.py \
  prepare/fetch_teacher_v102_assets.py \
  prepare/package_teacher_v102_assets.py
echo "[PASS] core Python syntax"

PYTHONPATH=prepare python prepare/test_teacher_v102_portable_assets.py
PYTHONPATH=prepare python prepare/test_package_teacher_v102_assets.py

forbidden=$(find . -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.safetensors' -o -name '*.npy' -o -name '*.npz' -o -name '*.pkl' -o -name '*.pickle' -o -name '*.zip' -o -name '*.sha256' \) -not -path './.git/*' -print)
if [[ -n "$forbidden" ]]; then
  echo "[FAIL] forbidden generated/checkpoint payloads found"
  echo "$forbidden"
  exit 1
fi
echo "[PASS] no generated maps, checkpoints, or transport archives"

large=$(find . -type f -size +50M -not -path './.git/*' -print)
if [[ -n "$large" ]]; then
  echo "[FAIL] files larger than 50 MiB found"
  echo "$large"
  exit 1
fi
echo "[PASS] no file exceeds 50 MiB"

leaks=$(find . -type f \( -name '*.py' -o -name '*.sh' -o -name '*.md' -o -name '*.json' -o -name '*.yaml' -o -name '*.yml' \) -not -path './.git/*' -not -path './docs/UPLOAD_CHECKLIST.md' -not -path './scripts/teacher_v102/verify_source_release.sh' -print0 | xargs -0 grep -nE '/home/kang|/Users/kanghyunkyu|DESKTOP-KANG|kang-gpu-server|BEGIN (RSA|OPENSSH|EC) PRIVATE KEY|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}' || true)
if [[ -n "$leaks" ]]; then
  echo "[FAIL] machine-local path or credential pattern found"
  echo "$leaks"
  exit 1
fi
echo "[PASS] no machine-local path or credential pattern"

echo "[SOURCE_RELEASE_PASS] Teacher-v9 through Teacher-v10.3.1"
