# Teacher-v9.8.1: selected t50/radius-0.003 K=3 response

This gate confirms that the rollout-state direction selected by v9.8 is not a
generation-0 accident. It uses the same three sealed v9.7 generation seeds and
both object-agnostic Sit prompts. Generation 0 is reused exactly; only
generations 1 and 2 are newly generated.

The selected LoRA response is inactive from t499 through t51 and active from
t50 through t0. That schedule isolates the response measured by v9.8. It is
not yet a final Teacher checkpoint or inference policy.

Every command below is one complete line. No command ends in a continuation
backslash. Fail-fast options are disabled so a legitimate FAIL cannot close
the interactive terminal.

## 1. Verify, install and run CPU contracts

```bash
cd ~/AMDM
conda activate afford
set +e
set +u
set +o pipefail
sha256sum -c teacher_lora_v981_rollout_state_response6_v1.zip.sha256
unzip -q -o teacher_lora_v981_rollout_state_response6_v1.zip
PATCH=teacher_lora_v981_rollout_state_response6_patch
python "$PATCH/prepare/validate_teacher_lora_v981_rollout_state_response6_package.py"
MISSING=0; for F in fewshot_cdm_common.py fewshot_cdm_lora.py preflight_relational_teacher_v98_rollout_state_response.py relational_teacher_v9_all_sittable_contract.py relational_teacher_v9_all_sittable_metrics.py relational_teacher_v9_lora_objective.py relational_teacher_v9_lora_preflight_contract.py relational_teacher_v9_lora_runtime.py relational_teacher_v91_active_support_objective.py relational_teacher_v91_corrected_overfit_contract.py relational_teacher_v94_common_descent.py relational_teacher_v97_early_rollout_k3_contract.py relational_teacher_v98_rollout_state_contract.py run_relational_teacher_v91_corrected_one_scene_overfit.py train_fewshot_cdm.py validate_relational_teacher_v98_rollout_state_response.py; do if [ ! -f "prepare/$F" ]; then echo "[STOP] missing prepare/$F"; MISSING=1; fi; done; if [ "$MISSING" -eq 0 ]; then echo "[OK] dependencies present"; else echo "[STOP] paste the missing-file lines; do not run CUDA"; fi
cp "$PATCH"/prepare/*.py prepare/
python prepare/test_relational_teacher_v981_rollout_state_response6_contract.py
```

Continue only after the package validator prints three PASS lines, the
dependency check prints `[OK] dependencies present`, and the CPU contract
prints three PASS lines.

## 2. Bind sealed inputs and a fresh output directory

```bash
SRC=data/history_affordance_relational_teacher_v7_hd
DATA=data/history_affordance_relational_teacher_v9_all_sittable_v1
V5DATA=data/history_affordance_v1
V5=$V5DATA/experiments/fewshot_cdm_chair23_bed2_whiteboard12_v5r4
V98=$DATA/experiments/teacher_lora_v98/rollout_state_preflight_s20261016_v1/preflight.json
OUT=$DATA/experiments/teacher_lora_v981/selected_t50_r003_train_k3_s20261017_v1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
if [ -e "$OUT" ]; then echo "[STOP] OUT already exists; preserve it and use a new versioned OUT"; else echo "[OK] OUT is new"; fi
```

Do not continue unless the last line is `[OK] OUT is new`.

## 3. Revalidate the selected v9.8 authority

```bash
python prepare/validate_relational_teacher_v98_rollout_state_response.py --report "$V98"
```

The conclusion must be `ROLLOUT_STATE_PREFLIGHT_PASS`, the selected candidate
must be `t50_radius_0p003`, and all four evidence lines must print PASS.

## 4. Run the CUDA response-6 gate

```bash
python -u prepare/evaluate_relational_teacher_v981_rollout_state_response6.py --v98-report "$V98" --source-dataset-root "$SRC" --dataset-root "$DATA" --index "$DATA/index.json" --v5-dataset-root "$V5DATA" --v5-split "$V5DATA/splits/chair23_bed2_whiteboard12_multistart24_v1.json" --stats-file data/Mean_Std_Cont_HumanML3D_HUMANISE_PROX_contact_cont_joints_0.8_fur.npz --original-checkpoint outputs/CDM-Perceiver-ALL/ckpt/model300000.pt --v5-checkpoint "$V5/fewshot_cdm.pt" --v5-evidence-report "$V5/gate0a_report.json" --output-dir "$OUT" --diffusion-steps 500 --seed 20261016 --lora-rank 4 --lora-alpha 8 --device cuda:0 --no-progress
```

The run must first print `SELECTED_DIRECTION_REPRODUCTION_PASS`. It then makes
four full frozen-Base draws, four selected candidate partial resumes and one
partial determinism repeat. All work is sequential for a 6 GB GPU.

## 5. Deep validation and readable diagnosis

Run both commands whether the result is PASS or FAIL:

```bash
python prepare/validate_relational_teacher_v981_rollout_state_response6.py --summary "$OUT/summary.json"
python prepare/summarize_relational_teacher_v981_rollout_state_response6.py --summary "$OUT/summary.json"
```

`ROLLOUT_STATE_RESPONSE6_PASS` authorizes only a fresh six-update rollout-state
calibration. It does not claim that absolute three-object presence has already
been achieved, and it does not authorize a checkpoint. A legitimate FAIL
authorizes only an audit of this response artifact.

Neither result authorizes `room_0102`, `room_0201`, long training or paper-test
access.
