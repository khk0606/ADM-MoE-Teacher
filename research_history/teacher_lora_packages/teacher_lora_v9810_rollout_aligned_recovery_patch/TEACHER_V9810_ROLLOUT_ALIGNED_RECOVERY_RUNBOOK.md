# Teacher-v9.8.10: actual two-scene K=3 recovery bracket

Copy the ZIP and checksum into `~/AMDM`, then paste this entire block once.
There are no shell continuation backslashes. Errors are reported without
closing the interactive terminal.

```bash
run_v9810_rollout_aligned_recovery() {
  cd ~/AMDM || return 1
  conda activate afford || return 1
  set +e
  set +u
  set +o pipefail
  sha256sum -c teacher_lora_v9810_rollout_aligned_recovery_v1.zip.sha256 || return 1
  unzip -q -o teacher_lora_v9810_rollout_aligned_recovery_v1.zip || return 1
  PATCH=teacher_lora_v9810_rollout_aligned_recovery_patch
  python "$PATCH/prepare/validate_teacher_lora_v9810_rollout_aligned_recovery_package.py" || return 1
  MISSING=0
  for F in fewshot_cdm_common.py fewshot_cdm_lora.py evaluate_relational_teacher_v987_two_scene_radius_response.py evaluate_relational_teacher_v989_rollout_aligned_radius.py preflight_relational_teacher_v98_rollout_state_response.py preflight_relational_teacher_v983_preservation_direction.py preflight_relational_teacher_v985_two_scene_step4.py preflight_relational_teacher_v988_rollout_aligned_direction.py relational_teacher_v9_all_sittable_contract.py relational_teacher_v9_all_sittable_metrics.py relational_teacher_v9_lora_objective.py relational_teacher_v9_lora_preflight_contract.py relational_teacher_v9_lora_runtime.py relational_teacher_v91_active_support_objective.py relational_teacher_v94_common_descent.py relational_teacher_v97_early_rollout_k3_contract.py relational_teacher_v98_rollout_state_contract.py relational_teacher_v982_rollout_state_calibration6_contract.py relational_teacher_v984_preservation_calibration6_contract.py relational_teacher_v985_two_scene_step4_preflight_contract.py relational_teacher_v988_rollout_aligned_direction_contract.py relational_teacher_v989_rollout_aligned_radius_contract.py run_relational_teacher_v91_corrected_one_scene_overfit.py run_relational_teacher_v982_rollout_state_calibration6.py run_relational_teacher_v984_preservation_calibration6.py train_fewshot_cdm.py validate_relational_teacher_v989_rollout_aligned_radius.py; do
    if [ ! -f "prepare/$F" ]; then
      echo "[STOP] missing prepare/$F"
      MISSING=1
    fi
  done
  if [ "$MISSING" -ne 0 ]; then
    echo "[STOP] dependencies are incomplete; do not run CUDA"
    return 1
  fi
  echo "[OK] dependencies present"
  cp "$PATCH"/prepare/*.py prepare/ || return 1
  python prepare/test_relational_teacher_v9810_rollout_aligned_recovery_contract.py || return 1
  SRC=data/history_affordance_relational_teacher_v7_hd
  DATA=data/history_affordance_relational_teacher_v9_all_sittable_v1
  V5DATA=data/history_affordance_v1
  V5=$V5DATA/experiments/fewshot_cdm_chair23_bed2_whiteboard12_v5r4
  V989=$DATA/experiments/teacher_lora_v989/rollout_aligned_radius_s20261025_v1/summary.json
  OUT=$DATA/experiments/teacher_lora_v9810/rollout_aligned_recovery_s20261026_v1
  if [ -e "$OUT" ]; then
    echo "[STOP] OUT already exists; preserve it and change only the final version suffix"
    return 1
  fi
  python prepare/validate_relational_teacher_v989_rollout_aligned_radius.py --summary "$V989" || return 1
  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
  python -u prepare/evaluate_relational_teacher_v9810_rollout_aligned_recovery.py --failed-radius-summary "$V989" --source-dataset-root "$SRC" --dataset-root "$DATA" --index "$DATA/index.json" --v5-dataset-root "$V5DATA" --v5-split "$V5DATA/splits/chair23_bed2_whiteboard12_multistart24_v1.json" --stats-file data/Mean_Std_Cont_HumanML3D_HUMANISE_PROX_contact_cont_joints_0.8_fur.npz --original-checkpoint outputs/CDM-Perceiver-ALL/ckpt/model300000.pt --v5-checkpoint "$V5/fewshot_cdm.pt" --v5-evidence-report "$V5/gate0a_report.json" --output-dir "$OUT" --diffusion-steps 500 --seed 20261016 --lora-rank 4 --lora-alpha 8 --device cuda:0 --no-progress
  RUN_RC=$?
  if [ ! -f "$OUT/summary.json" ]; then
    echo "[STOP] CUDA run did not produce summary.json (exit=$RUN_RC)"
    return 1
  fi
  python prepare/validate_relational_teacher_v9810_rollout_aligned_recovery.py --summary "$OUT/summary.json" || return 1
  python prepare/summarize_relational_teacher_v9810_rollout_aligned_recovery.py --summary "$OUT/summary.json" || return 1
  return "$RUN_RC"
}
run_v9810_rollout_aligned_recovery
V9810_RC=$?
unset -f run_v9810_rollout_aligned_recovery
echo "[DONE] Teacher-v9.8.10 block exit=$V9810_RC; this terminal remains open"
```

PASS authorizes only fresh two-scene rollout-aligned calibration. FAIL
authorizes only cross-scene training-objective redesign. Neither result
authorizes a checkpoint, `room_0201`, long training or paper-test access.
