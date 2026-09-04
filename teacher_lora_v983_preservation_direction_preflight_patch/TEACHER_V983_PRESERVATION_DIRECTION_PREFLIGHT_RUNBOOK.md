# Teacher-v9.8.3: preservation-aware direction preflight

This is delivery v2. It fixes only the v1 candidate-log formatting exception;
the scientific protocol and authorization are unchanged. Preserve the failed
v1 output directory and use the new v2 output directory below.

The v9.8.2 result is not a prompt-invariance failure. Update 1 remains valid,
while update 2 first crosses the fixed-v5 1% retention cap and later updates
also increase explicit-negative means. This gate therefore adds those missing
preservation objectives to the rollout-state direction itself.

It performs no optimizer update and saves no model state. It reconstructs the
accepted update-1 LoRA state from fresh v5r4, computes one eleven-task direction
at the sealed update-2 design state, and tests five hypothetical radii using
actual K=3 continuations from t50 to t0.

## Run everything with one paste

Paste the complete block below once. No line ends in a continuation backslash.
The function uses explicit `return` gates and leaves fail-fast shell options
disabled, so the terminal remains open when a scientific gate returns FAIL.

```bash
run_v983_preservation_preflight() {
  cd ~/AMDM || return 1
  conda activate afford || return 1
  set +e
  set +u
  set +o pipefail
  sha256sum -c teacher_lora_v983_preservation_direction_preflight_v2.zip.sha256 || return 1
  unzip -q -o teacher_lora_v983_preservation_direction_preflight_v2.zip || return 1
  PATCH=teacher_lora_v983_preservation_direction_preflight_patch
  python "$PATCH/prepare/validate_teacher_lora_v983_preservation_direction_package.py" || return 1
  MISSING=0
  for F in fewshot_cdm_common.py fewshot_cdm_lora.py preflight_relational_teacher_v98_rollout_state_response.py relational_teacher_v9_all_sittable_contract.py relational_teacher_v9_all_sittable_metrics.py relational_teacher_v9_lora_objective.py relational_teacher_v9_lora_preflight_contract.py relational_teacher_v9_lora_runtime.py relational_teacher_v91_active_support_objective.py relational_teacher_v91_corrected_overfit_contract.py relational_teacher_v94_common_descent.py relational_teacher_v97_early_rollout_k3_contract.py relational_teacher_v98_rollout_state_contract.py relational_teacher_v981_rollout_state_response6_contract.py relational_teacher_v982_rollout_state_calibration6_contract.py run_relational_teacher_v91_corrected_one_scene_overfit.py run_relational_teacher_v982_rollout_state_calibration6.py train_fewshot_cdm.py validate_relational_teacher_v982_rollout_state_calibration6.py; do
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
  python prepare/test_relational_teacher_v983_preservation_direction_contract.py || return 1
  SRC=data/history_affordance_relational_teacher_v7_hd
  DATA=data/history_affordance_relational_teacher_v9_all_sittable_v1
  V5DATA=data/history_affordance_v1
  V5=$V5DATA/experiments/fewshot_cdm_chair23_bed2_whiteboard12_v5r4
  FAILED982=$DATA/experiments/teacher_lora_v982/rollout_state_calibration6_s20261018_v1/summary.json
  OUT=$DATA/experiments/teacher_lora_v983/preservation_direction_preflight_s20261019_v2
  if [ -e "$OUT" ]; then
    echo "[STOP] OUT already exists; preserve it and change only the final version suffix"
    return 1
  fi
  python prepare/validate_relational_teacher_v982_rollout_state_calibration6.py --summary "$FAILED982" || return 1
  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
  python -u prepare/preflight_relational_teacher_v983_preservation_direction.py --failed-calibration-summary "$FAILED982" --source-dataset-root "$SRC" --dataset-root "$DATA" --index "$DATA/index.json" --v5-dataset-root "$V5DATA" --v5-split "$V5DATA/splits/chair23_bed2_whiteboard12_multistart24_v1.json" --stats-file data/Mean_Std_Cont_HumanML3D_HUMANISE_PROX_contact_cont_joints_0.8_fur.npz --original-checkpoint outputs/CDM-Perceiver-ALL/ckpt/model300000.pt --v5-checkpoint "$V5/fewshot_cdm.pt" --v5-evidence-report "$V5/gate0a_report.json" --output-dir "$OUT" --diffusion-steps 500 --seed 20261016 --lora-rank 4 --lora-alpha 8 --device cuda:0 --no-progress
  RUN_RC=$?
  if [ ! -f "$OUT/preflight.json" ]; then
    echo "[STOP] CUDA run did not produce preflight.json (exit=$RUN_RC)"
    return 1
  fi
  python prepare/validate_relational_teacher_v983_preservation_direction.py --report "$OUT/preflight.json" || return 1
  python prepare/summarize_relational_teacher_v983_preservation_direction.py --report "$OUT/preflight.json" || return 1
  return "$RUN_RC"
}
run_v983_preservation_preflight
V983_RC=$?
unset -f run_v983_preservation_preflight
echo "[DONE] Teacher-v9.8.3 block exit=$V983_RC; this terminal remains open"
```

The first validator is expected to print the sealed v9.8.2 `FAIL` conclusion
while returning successfully because it is validating an authentic failure
artifact. That is not a reason to stop.

The CUDA run reproduces six Base trajectories, two update-2 design trajectories
and exact resumes, six exact update-1 partial resumes, and thirty candidate
partial resumes. GPU work is sequential for a 6 GB card.

`PRESERVATION_DIRECTION_PREFLIGHT_PASS` requires at least one radius to satisfy
all locked gates. A PASS authorizes only a new fresh preservation-aware
calibration-6 package. A FAIL is still a valid diagnosis and authorizes only an
audit of this artifact. Neither result authorizes `room_0102`, `room_0201`, a
checkpoint, long training, or paper-test access.
