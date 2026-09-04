# Teacher-v9.8: rollout-state final-map response preflight

Teacher-v9.7 failed at the real decision unit: after actual 500-step reverse
diffusion, Bed/normal Chair/High Chair simultaneous presence was `0/3` for both
watch and write at steps 3, 6 and 12. Teacher-v9.8 does not extend that failed
training. It starts from sealed v5r4 with a fresh zero-output LoRA and asks
whether a small direction computed on states actually visited by the sampler
can improve the final generated map.

The gate captures state and RNG at timesteps 400, 200 and 50. For each capture
it builds one minimum-norm common-descent direction over six tasks (three
objects x two prompts), tests radii 0.001, 0.003 and 0.01, then resumes the
original audit trajectory to timestep 0. Selection uses those final maps only.

Every command below is one complete line and no line ends in a continuation
backslash. Fail-fast flags are disabled so a legitimate FAIL cannot close the
interactive terminal.

## 1. Verify, install and run CPU contracts

```bash
cd ~/AMDM
conda activate afford
set +e
set +u
set +o pipefail
sha256sum -c teacher_lora_v98_rollout_state_preflight_v1.zip.sha256
unzip -q -o teacher_lora_v98_rollout_state_preflight_v1.zip
PATCH=teacher_lora_v98_rollout_state_preflight_patch
python "$PATCH/prepare/validate_teacher_lora_v98_rollout_state_package.py"
MISSING=0; for F in fewshot_cdm_common.py fewshot_cdm_lora.py relational_teacher_v9_all_sittable_contract.py relational_teacher_v9_all_sittable_metrics.py relational_teacher_v9_lora_objective.py relational_teacher_v9_lora_preflight_contract.py relational_teacher_v9_lora_runtime.py relational_teacher_v91_active_support_objective.py relational_teacher_v91_corrected_overfit_contract.py relational_teacher_v94_common_descent.py relational_teacher_v97_early_rollout_k3_contract.py run_relational_teacher_v91_corrected_one_scene_overfit.py train_fewshot_cdm.py validate_relational_teacher_v9_all_sittable_lora_preflight.py validate_relational_teacher_v97_early_rollout_k3.py; do if [ ! -f "prepare/$F" ]; then echo "[STOP] missing prepare/$F"; MISSING=1; fi; done; if [ "$MISSING" -eq 0 ]; then echo "[OK] dependencies present"; else echo "[STOP] paste the missing-file lines; do not run CUDA"; fi
cp "$PATCH"/prepare/*.py prepare/
python prepare/test_relational_teacher_v98_rollout_state_contract.py
```

Continue only after the package validator prints three PASS lines, the
dependency check prints `[OK] dependencies present`, and the CPU test prints
three PASS lines.

## 2. Bind sealed inputs and a fresh output directory

```bash
SRC=data/history_affordance_relational_teacher_v7_hd
DATA=data/history_affordance_relational_teacher_v9_all_sittable_v1
V5DATA=data/history_affordance_v1
V5=$V5DATA/experiments/fewshot_cdm_chair23_bed2_whiteboard12_v5r4
PREFLIGHT=$DATA/experiments/teacher_lora_v9/preflight_s20261005_v3/preflight.json
FAILED97=$DATA/experiments/teacher_lora_v97/early_rollout_k3_s20261015_v1/summary.json
OUT=$DATA/experiments/teacher_lora_v98/rollout_state_preflight_s20261016_v1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
if [ -e "$OUT" ]; then echo "[STOP] OUT already exists; preserve it and use a new versioned OUT"; else echo "[OK] OUT is new"; fi
```

Do not continue unless the last line is `[OK] OUT is new`.

## 3. Revalidate the two authorities

```bash
python prepare/validate_relational_teacher_v9_all_sittable_lora_preflight.py --report "$PREFLIGHT"
python prepare/validate_relational_teacher_v97_early_rollout_k3.py --summary "$FAILED97"
```

The first conclusion must be `PREFLIGHT_PASS`. The second must be the already
sealed `EARLY_ROLLOUT_K3_FAIL` integrity result with all four PASS evidence
lines. That FAIL is the input diagnosis, not an execution error.

## 4. Run the CUDA rollout-state grid

```bash
python -u prepare/preflight_relational_teacher_v98_rollout_state_response.py --failed-v97-summary "$FAILED97" --source-dataset-root "$SRC" --dataset-root "$DATA" --index "$DATA/index.json" --v5-dataset-root "$V5DATA" --v5-split "$V5DATA/splits/chair23_bed2_whiteboard12_multistart24_v1.json" --stats-file data/Mean_Std_Cont_HumanML3D_HUMANISE_PROX_contact_cont_joints_0.8_fur.npz --original-checkpoint outputs/CDM-Perceiver-ALL/ckpt/model300000.pt --v5-checkpoint "$V5/fewshot_cdm.pt" --v5-evidence-report "$V5/gate0a_report.json" --preflight-report "$PREFLIGHT" --output-dir "$OUT" --diffusion-steps 500 --seed 20261016 --lora-rank 4 --lora-alpha 8 --device cuda:0 --no-progress
```

On a 6 GB GPU this intentionally runs sequentially. It performs four full Base
draws, six exact Base partial resumes, and eighteen candidate partial resumes.
No rollout batch is retained on the GPU and no optimizer is created.

## 5. Deep validation and readable diagnosis

Run both commands whether the preflight reports PASS or FAIL:

```bash
python prepare/validate_relational_teacher_v98_rollout_state_response.py --report "$OUT/preflight.json"
python prepare/summarize_relational_teacher_v98_rollout_state_response.py --report "$OUT/preflight.json"
```

`ROLLOUT_STATE_PREFLIGHT_PASS` means at least one timestep/radius improves High
Chair recall and MAE, the worst-object recall and macro MAE, while retaining
each object/prompt, prompt invariance, explicit negatives and the v5 probe.
Only then is a six-update confirmation of the selected rollout-state policy
authorized. PASS does not itself create or authorize a checkpoint.

A legitimate FAIL authorizes only an audit of this response artifact. Neither
result authorizes `room_0102`, `room_0201`, long training or paper-test access.
