# Teacher-v9.5: multi-timestep/noise gradient consensus

The sealed v9.4.1 result showed a timestep-specific conflict: normal Chair
regressed only on the `t=50` watch row, Bed regressed on both `t=425` rows,
and High Chair did not regress. This gate therefore replaces the one-panel
direction with a 24-task direction spanning four design timesteps.

Run every command as one complete line. No command below ends in a backslash.

## 1. Verify, install, and run the CPU contract

```bash
cd ~/AMDM
conda activate afford
set +e
set +u
set +o pipefail
sha256sum -c teacher_lora_v95_multitimestep_consensus_v1.zip.sha256
unzip -q -o teacher_lora_v95_multitimestep_consensus_v1.zip
PATCH=teacher_lora_v95_multitimestep_consensus_patch
python "$PATCH/prepare/validate_teacher_lora_v95_multitimestep_consensus_package.py"
for F in fewshot_cdm_common.py fewshot_cdm_lora.py preflight_relational_teacher_v94_common_descent.py relational_teacher_v9_all_sittable_contract.py relational_teacher_v9_lora_preflight_contract.py relational_teacher_v9_lora_runtime.py relational_teacher_v93_exact_topk_objective.py relational_teacher_v94_common_descent.py run_relational_teacher_v91_corrected_one_scene_overfit.py train_fewshot_cdm.py; do test -f "prepare/$F" || echo "[STOP] missing prepare/$F"; done
cp "$PATCH"/prepare/*.py prepare/
python prepare/test_relational_teacher_v95_multitimestep_consensus_contract.py
```

The package validator must print two PASS lines. The CPU test must print three
PASS lines. If any `[STOP] missing` line appears, stop and paste that output.

## 2. Bind the sealed inputs

```bash
SRC=data/history_affordance_relational_teacher_v7_hd
DATA=data/history_affordance_relational_teacher_v9_all_sittable_v1
V5DATA=data/history_affordance_v1
V5=$V5DATA/experiments/fewshot_cdm_chair23_bed2_whiteboard12_v5r4
PREFLIGHT=$DATA/experiments/teacher_lora_v9/preflight_s20261005_v3/preflight.json
FAILED941=$DATA/experiments/teacher_lora_v941/metric_replication_s20261012_v1/summary.json
OUT=$DATA/experiments/teacher_lora_v95/multitimestep_consensus_s20261013_v1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
if [ -e "$OUT" ]; then echo "[STOP] OUT already exists; choose a new versioned OUT"; else echo "[OK] OUT is new"; fi
```

Do not continue unless the last line is `[OK] OUT is new`.

## 3. Run the CUDA gate

```bash
python -u prepare/preflight_relational_teacher_v95_multitimestep_consensus.py --failed-v941-summary "$FAILED941" --source-dataset-root "$SRC" --dataset-root "$DATA" --index "$DATA/index.json" --v5-dataset-root "$V5DATA" --v5-split "$V5DATA/splits/chair23_bed2_whiteboard12_multistart24_v1.json" --stats-file data/Mean_Std_Cont_HumanML3D_HUMANISE_PROX_contact_cont_joints_0.8_fur.npz --original-checkpoint outputs/CDM-Perceiver-ALL/ckpt/model300000.pt --v5-checkpoint "$V5/fewshot_cdm.pt" --v5-evidence-report "$V5/gate0a_report.json" --preflight-report "$PREFLIGHT" --output-dir "$OUT" --diffusion-steps 500 --seed 20261013 --device cuda:0
```

The run prints exactly three `[CONSENSUS R=...]` rows, followed by either
`MULTITIMESTEP_CONSENSUS_PASS` or a legitimate fail-closed result. It computes
the direction on timesteps 50/125/275/425, then selects only from results on
new timesteps 25/175/350/475.

## 4. Deep validation

Run this even when the CUDA gate reports FAIL:

```bash
python prepare/validate_relational_teacher_v95_multitimestep_consensus.py --report "$OUT/preflight.json"
```

`MULTITIMESTEP_CONSENSUS_PASS` authorizes only a fresh six-update consensus
response gate using the selected radius. It does not authorize a checkpoint,
rollout, 120-step overfit, calibration, `room_0102`, `room_0201`, development,
or paper-test access.
