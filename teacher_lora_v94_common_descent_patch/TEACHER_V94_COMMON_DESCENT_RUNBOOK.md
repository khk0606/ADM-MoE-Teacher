# Teacher-v9.4: six-task common-descent preflight

Teacher-v9.3 proved that the aggregate exact-top-k loss could improve Bed while
normal-Chair and High-Chair swap losses became worse. This gate does not run a
longer training job. It first asks whether the fresh shared LoRA has a single
gradient direction that locally decreases all six object/prompt tasks.

## 1. Verify and install

Run each command as a complete line. There are no trailing backslashes.

```bash
cd ~/AMDM
conda activate afford
set +e
set +u
set +o pipefail
sha256sum -c teacher_lora_v94_common_descent_preflight_v1.zip.sha256
unzip -q -o teacher_lora_v94_common_descent_preflight_v1.zip
PATCH=teacher_lora_v94_common_descent_patch
python "$PATCH/prepare/validate_teacher_lora_v94_common_descent_package.py"
cp "$PATCH"/prepare/*.py prepare/
python prepare/test_relational_teacher_v94_common_descent_contract.py
```

The package validator and the three CPU-contract PASS lines are required.

## 2. Run the CUDA geometry gate

```bash
SRC=data/history_affordance_relational_teacher_v7_hd
DATA=data/history_affordance_relational_teacher_v9_all_sittable_v1
V5DATA=data/history_affordance_v1
V5=$V5DATA/experiments/fewshot_cdm_chair23_bed2_whiteboard12_v5r4
PREFLIGHT=$DATA/experiments/teacher_lora_v9/preflight_s20261005_v3/preflight.json
FAILED93=$DATA/experiments/teacher_lora_v93/exact_topk_response6_s20261010_v1/preflight.json
OUT=$DATA/experiments/teacher_lora_v94/common_descent_preflight_s20261011_v1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
test ! -e "$OUT"
```

If the final `test` prints nothing, run this one complete line:

```bash
python -u prepare/preflight_relational_teacher_v94_common_descent.py --failed-v93-report "$FAILED93" --source-dataset-root "$SRC" --dataset-root "$DATA" --index "$DATA/index.json" --v5-dataset-root "$V5DATA" --v5-split "$V5DATA/splits/chair23_bed2_whiteboard12_multistart24_v1.json" --stats-file data/Mean_Std_Cont_HumanML3D_HUMANISE_PROX_contact_cont_joints_0.8_fur.npz --original-checkpoint outputs/CDM-Perceiver-ALL/ckpt/model300000.pt --v5-checkpoint "$V5/fewshot_cdm.pt" --v5-evidence-report "$V5/gate0a_report.json" --preflight-report "$PREFLIGHT" --output-dir "$OUT" --diffusion-steps 500 --seed 20261011 --device cuda:0
```

Then validate independently:

```bash
python prepare/validate_relational_teacher_v94_common_descent.py --report "$OUT/preflight.json"
```

## Interpretation

`COMMON_DESCENT_PASS` means at least one trust-region radius strictly reduced
all six fixed-panel exact-top-k swap losses without reducing any of the six
top-k overlaps or violating prompt, negative, background, or v5 retention.
It authorizes only a new six-update common-descent response experiment.

`COMMON_DESCENT_FAIL` is also useful evidence:

- if `common_direction_exists` fails, the six shared-LoRA gradients have no
  reliable common first-order descent direction and an architecture/parameter
  separation is required;
- if the direction exists but every radius fails, the local linear direction
  does not survive an actual parameter step under the strict retention gates.

Do not run response-120, calibration, rollout, room_0102, room_0201, or paper
test from this package. No checkpoint is saved.

