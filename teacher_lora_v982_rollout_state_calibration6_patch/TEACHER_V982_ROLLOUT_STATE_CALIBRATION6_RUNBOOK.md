# Teacher-v9.8.2: fresh rollout-state calibration-6

The sealed v9.8.1 K=3 response proves that the t50/radius-0.003 direction is
not a generation-0 accident. This gate now starts over from the sealed v5r4
Teacher and applies six online common-descent LoRA parameter updates.

Update 1 exactly reproduces v9.8.1. Updates 2 through 6 use new deterministic
frozen-Base t50 design states. Every update is followed by actual t50-to-t0
generation for K=3 and both Sit prompts. No optimizer state or model
checkpoint is saved.

Every command below is one complete line. No command ends in a continuation
backslash. Fail-fast shell options are disabled, so a legitimate FAIL does not
close the interactive terminal.

## 1. Verify, install and run CPU contracts

```bash
cd ~/AMDM
conda activate afford
set +e
set +u
set +o pipefail
sha256sum -c teacher_lora_v982_rollout_state_calibration6_v1.zip.sha256
unzip -q -o teacher_lora_v982_rollout_state_calibration6_v1.zip
PATCH=teacher_lora_v982_rollout_state_calibration6_patch
python "$PATCH/prepare/validate_teacher_lora_v982_rollout_state_calibration6_package.py"
MISSING=0; for F in fewshot_cdm_common.py fewshot_cdm_lora.py preflight_relational_teacher_v98_rollout_state_response.py relational_teacher_v9_all_sittable_contract.py relational_teacher_v9_all_sittable_metrics.py relational_teacher_v9_lora_objective.py relational_teacher_v9_lora_preflight_contract.py relational_teacher_v9_lora_runtime.py relational_teacher_v91_active_support_objective.py relational_teacher_v91_corrected_overfit_contract.py relational_teacher_v94_common_descent.py relational_teacher_v97_early_rollout_k3_contract.py relational_teacher_v98_rollout_state_contract.py relational_teacher_v981_rollout_state_response6_contract.py run_relational_teacher_v91_corrected_one_scene_overfit.py train_fewshot_cdm.py validate_relational_teacher_v981_rollout_state_response6.py; do if [ ! -f "prepare/$F" ]; then echo "[STOP] missing prepare/$F"; MISSING=1; fi; done; if [ "$MISSING" -eq 0 ]; then echo "[OK] dependencies present"; else echo "[STOP] paste the missing-file lines; do not run CUDA"; fi
cp "$PATCH"/prepare/*.py prepare/
python prepare/test_relational_teacher_v982_rollout_state_calibration6_contract.py
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
V981=$DATA/experiments/teacher_lora_v981/selected_t50_r003_train_k3_s20261017_v1/summary.json
OUT=$DATA/experiments/teacher_lora_v982/rollout_state_calibration6_s20261018_v1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
if [ -e "$OUT" ]; then echo "[STOP] OUT already exists; preserve it and use a new versioned OUT"; else echo "[OK] OUT is new"; fi
```

Do not continue unless the final line is `[OK] OUT is new`.

## 3. Revalidate the v9.8.1 authority

```bash
python prepare/validate_relational_teacher_v981_rollout_state_response6.py --summary "$V981"
```

The conclusion must be `ROLLOUT_STATE_RESPONSE6_PASS`, every evidence line
must print PASS, and failed checks must be empty.

## 4. Run fresh CUDA calibration-6

```bash
python -u prepare/run_relational_teacher_v982_rollout_state_calibration6.py --response6-summary "$V981" --source-dataset-root "$SRC" --dataset-root "$DATA" --index "$DATA/index.json" --v5-dataset-root "$V5DATA" --v5-split "$V5DATA/splits/chair23_bed2_whiteboard12_multistart24_v1.json" --stats-file data/Mean_Std_Cont_HumanML3D_HUMANISE_PROX_contact_cont_joints_0.8_fur.npz --original-checkpoint outputs/CDM-Perceiver-ALL/ckpt/model300000.pt --v5-checkpoint "$V5/fewshot_cdm.pt" --v5-evidence-report "$V5/gate0a_report.json" --output-dir "$OUT" --diffusion-steps 500 --seed 20261016 --updates 6 --step-radius 0.003 --lora-rank 4 --lora-alpha 8 --device cuda:0 --no-progress
```

The run first reproduces all six frozen-Base K=3 maps. It then creates ten
new Base design draws and exact t50 resumes. Finally, each of six updates is
checked with six paired partial generation resumes. All GPU work is sequential
for a 6 GB GPU. Update 1 must print `UPDATE1_REPRODUCTION_PASS`.

## 5. Deep validation and readable diagnosis

Run both commands whether the result is PASS or FAIL:

```bash
python prepare/validate_relational_teacher_v982_rollout_state_calibration6.py --summary "$OUT/summary.json"
python prepare/summarize_relational_teacher_v982_rollout_state_calibration6.py --summary "$OUT/summary.json"
```

`ROLLOUT_STATE_CALIBRATION6_PASS` requires at least one admissible state after
update 1. Each admissible state must improve pooled recall and MAE for Bed,
normal Chair, and High Chair while retaining every individual map, prompt
invariance, explicit negatives, and the fixed v5 probe. At most two steps are
shortlisted, but no model state is serialized.

A PASS authorizes only a preflight for a fresh two-train-scene rollout-state
calibration. It does not itself authorize `room_0102` array access, a
checkpoint, development room `room_0201`, long training, or paper-test access.
A legitimate FAIL authorizes only an audit of this calibration artifact.
