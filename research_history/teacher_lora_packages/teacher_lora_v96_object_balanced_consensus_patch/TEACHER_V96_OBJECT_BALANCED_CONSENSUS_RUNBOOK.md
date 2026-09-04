# Teacher-v9.6: object-balanced consensus CUDA gate

Run every command as one complete line. No command ends with a backslash.

## 1. Verify and install

```bash
cd ~/AMDM
conda activate afford
set +e
set +u
set +o pipefail
sha256sum -c teacher_lora_v96_object_balanced_consensus_v1.zip.sha256
unzip -q -o teacher_lora_v96_object_balanced_consensus_v1.zip
PATCH=teacher_lora_v96_object_balanced_consensus_patch
python "$PATCH/prepare/validate_teacher_lora_v96_object_balanced_consensus_package.py"
for F in fewshot_cdm_common.py fewshot_cdm_lora.py preflight_relational_teacher_v94_common_descent.py preflight_relational_teacher_v95_multitimestep_consensus.py relational_teacher_v9_all_sittable_contract.py relational_teacher_v9_lora_preflight_contract.py relational_teacher_v9_lora_runtime.py relational_teacher_v93_exact_topk_objective.py relational_teacher_v94_common_descent.py relational_teacher_v95_multitimestep_consensus_contract.py run_relational_teacher_v91_corrected_one_scene_overfit.py train_fewshot_cdm.py; do if [ ! -f "prepare/$F" ]; then echo "[STOP] missing prepare/$F"; fi; done
cp "$PATCH"/prepare/*.py prepare/
python prepare/test_relational_teacher_v96_object_balanced_consensus_contract.py
```

The package validator must print two PASS lines and the CPU contract must print
three PASS lines. Stop if any `[STOP] missing` line appears.

## 2. Bind sealed inputs

```bash
SRC=data/history_affordance_relational_teacher_v7_hd
DATA=data/history_affordance_relational_teacher_v9_all_sittable_v1
V5DATA=data/history_affordance_v1
V5=$V5DATA/experiments/fewshot_cdm_chair23_bed2_whiteboard12_v5r4
PREFLIGHT=$DATA/experiments/teacher_lora_v9/preflight_s20261005_v3/preflight.json
FAILED95=$DATA/experiments/teacher_lora_v95/multitimestep_consensus_s20261013_v1/preflight.json
OUT=$DATA/experiments/teacher_lora_v96/object_balanced_consensus_s20261014_v1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
if [ -e "$OUT" ]; then echo "[STOP] OUT already exists; choose a new versioned OUT"; else echo "[OK] OUT is new"; fi
```

Continue only after `[OK] OUT is new`.

## 3. Run the CUDA gate

```bash
python -u prepare/preflight_relational_teacher_v96_object_balanced_consensus.py --failed-v95-report "$FAILED95" --source-dataset-root "$SRC" --dataset-root "$DATA" --index "$DATA/index.json" --v5-dataset-root "$V5DATA" --v5-split "$V5DATA/splits/chair23_bed2_whiteboard12_multistart24_v1.json" --stats-file data/Mean_Std_Cont_HumanML3D_HUMANISE_PROX_contact_cont_joints_0.8_fur.npz --original-checkpoint outputs/CDM-Perceiver-ALL/ckpt/model300000.pt --v5-checkpoint "$V5/fewshot_cdm.pt" --v5-evidence-report "$V5/gate0a_report.json" --preflight-report "$PREFLIGHT" --output-dir "$OUT" --diffusion-steps 500 --seed 20261014 --device cuda:0
```

The run computes 48 design gradients and prints exactly 16
`[BALANCED ... R=...]` rows. It never loads a failed candidate state.

## 4. Deep validation

Run this for both PASS and FAIL outcomes:

```bash
python prepare/validate_relational_teacher_v96_object_balanced_consensus.py --report "$OUT/preflight.json"
```

`OBJECT_BALANCED_CONSENSUS_PASS` authorizes only a fresh six-update response
gate using the selected direction and radius. It does not authorize checkpoint
writing, rollout, overfit120, calibration, `room_0102`, `room_0201`,
development evaluation, long training, or paper-test access.
