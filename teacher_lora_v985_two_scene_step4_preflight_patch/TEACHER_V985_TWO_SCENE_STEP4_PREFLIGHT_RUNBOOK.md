# Teacher-v9.8.5: exact step-4 cross-scene preflight

Copy the ZIP and checksum into `~/AMDM`, then paste the entire block once. No
line ends in a shell continuation character, and an error returns from the
function without closing the interactive terminal.

```bash
run_v985_two_scene_step4_preflight() {
  cd ~/AMDM || return 1
  conda activate afford || return 1
  set +e
  set +u
  set +o pipefail
  sha256sum -c teacher_lora_v985_two_scene_step4_preflight_v1.zip.sha256 || return 1
  unzip -q -o teacher_lora_v985_two_scene_step4_preflight_v1.zip || return 1
  PATCH=teacher_lora_v985_two_scene_step4_preflight_patch
  python "$PATCH/prepare/validate_teacher_lora_v985_two_scene_step4_preflight_package.py" || return 1
  MISSING=0
  for F in fewshot_cdm_common.py fewshot_cdm_lora.py preflight_relational_teacher_v98_rollout_state_response.py preflight_relational_teacher_v983_preservation_direction.py relational_teacher_v9_all_sittable_contract.py relational_teacher_v9_all_sittable_metrics.py relational_teacher_v9_lora_objective.py relational_teacher_v9_lora_preflight_contract.py relational_teacher_v9_lora_runtime.py relational_teacher_v91_active_support_objective.py relational_teacher_v94_common_descent.py relational_teacher_v97_early_rollout_k3_contract.py relational_teacher_v98_rollout_state_contract.py relational_teacher_v982_rollout_state_calibration6_contract.py relational_teacher_v984_preservation_calibration6_contract.py run_relational_teacher_v91_corrected_one_scene_overfit.py run_relational_teacher_v982_rollout_state_calibration6.py run_relational_teacher_v984_preservation_calibration6.py train_fewshot_cdm.py validate_relational_teacher_v984_preservation_calibration6.py; do
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
  python prepare/test_relational_teacher_v985_two_scene_step4_preflight_contract.py || return 1
  SRC=data/history_affordance_relational_teacher_v7_hd
  DATA=data/history_affordance_relational_teacher_v9_all_sittable_v1
  V5DATA=data/history_affordance_v1
  V5=$V5DATA/experiments/fewshot_cdm_chair23_bed2_whiteboard12_v5r4
  V984=$DATA/experiments/teacher_lora_v984/preservation_calibration6_s20261020_v1/summary.json
  OUT=$DATA/experiments/teacher_lora_v985/two_scene_step4_preflight_s20261021_v1
  if [ -e "$OUT" ]; then
    echo "[STOP] OUT already exists; preserve it and change only the final version suffix"
    return 1
  fi
  python prepare/validate_relational_teacher_v984_preservation_calibration6.py --summary "$V984" || return 1
  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
  python -u prepare/preflight_relational_teacher_v985_two_scene_step4.py --calibration-summary "$V984" --source-dataset-root "$SRC" --dataset-root "$DATA" --index "$DATA/index.json" --v5-dataset-root "$V5DATA" --v5-split "$V5DATA/splits/chair23_bed2_whiteboard12_multistart24_v1.json" --stats-file data/Mean_Std_Cont_HumanML3D_HUMANISE_PROX_contact_cont_joints_0.8_fur.npz --original-checkpoint outputs/CDM-Perceiver-ALL/ckpt/model300000.pt --v5-checkpoint "$V5/fewshot_cdm.pt" --v5-evidence-report "$V5/gate0a_report.json" --output-dir "$OUT" --diffusion-steps 500 --seed 20261016 --lora-rank 4 --lora-alpha 8 --device cuda:0 --no-progress
  RUN_RC=$?
  if [ ! -f "$OUT/preflight.json" ]; then
    echo "[STOP] CUDA run did not produce preflight.json (exit=$RUN_RC)"
    return 1
  fi
  python prepare/validate_relational_teacher_v985_two_scene_step4_preflight.py --report "$OUT/preflight.json" || return 1
  python prepare/summarize_relational_teacher_v985_two_scene_step4_preflight.py --report "$OUT/preflight.json" || return 1
  return "$RUN_RC"
}
run_v985_two_scene_step4_preflight
V985_RC=$?
unset -f run_v985_two_scene_step4_preflight
echo "[DONE] Teacher-v9.8.5 block exit=$V985_RC; this terminal remains open"
```

The run creates six room_0102 Base trajectories, six room_0101 design
trajectories plus exact resumes, and six room_0102 step-4 partial resumes. A
PASS only authorizes a fresh two-scene preservation response preflight. A FAIL
only authorizes cross-scene direction diagnosis. Neither outcome authorizes a
checkpoint, room_0201, long training or paper-test access.
