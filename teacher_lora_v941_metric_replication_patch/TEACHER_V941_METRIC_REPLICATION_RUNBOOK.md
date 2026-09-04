# Teacher-v9.4.1: metric-first noise/timestep replication

Run every command as a complete line. No command ends in a backslash.

## Install

```bash
cd ~/AMDM
conda activate afford
set +e
set +u
set +o pipefail
sha256sum -c teacher_lora_v941_metric_replication_v1.zip.sha256
unzip -q -o teacher_lora_v941_metric_replication_v1.zip
PATCH=teacher_lora_v941_metric_replication_patch
python "$PATCH/prepare/validate_teacher_lora_v941_metric_replication_package.py"
cp "$PATCH"/prepare/*.py prepare/
python prepare/test_relational_teacher_v941_metric_replication_contract.py
```

## Run

```bash
SRC=data/history_affordance_relational_teacher_v7_hd
DATA=data/history_affordance_relational_teacher_v9_all_sittable_v1
V5DATA=data/history_affordance_v1
V5=$V5DATA/experiments/fewshot_cdm_chair23_bed2_whiteboard12_v5r4
PREFLIGHT=$DATA/experiments/teacher_lora_v9/preflight_s20261005_v3/preflight.json
FAILED94=$DATA/experiments/teacher_lora_v94/common_descent_preflight_s20261011_v1/preflight.json
OUT=$DATA/experiments/teacher_lora_v941/metric_replication_s20261012_v1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
if [ -e "$OUT" ]; then echo "[STOP] OUT already exists; use a new versioned OUT"; else echo "[OK] OUT is new"; fi
```

CUDA evaluation, as one complete line:

```bash
python -u prepare/evaluate_relational_teacher_v941_metric_replication.py --failed-v94-report "$FAILED94" --source-dataset-root "$SRC" --dataset-root "$DATA" --index "$DATA/index.json" --v5-dataset-root "$V5DATA" --v5-split "$V5DATA/splits/chair23_bed2_whiteboard12_multistart24_v1.json" --stats-file data/Mean_Std_Cont_HumanML3D_HUMANISE_PROX_contact_cont_joints_0.8_fur.npz --original-checkpoint outputs/CDM-Perceiver-ALL/ckpt/model300000.pt --v5-checkpoint "$V5/fewshot_cdm.pt" --v5-evidence-report "$V5/gate0a_report.json" --preflight-report "$PREFLIGHT" --output-dir "$OUT" --diffusion-steps 500 --seed 20261012 --device cuda:0
```

Deep validation:

```bash
python prepare/validate_relational_teacher_v941_metric_replication.py --summary "$OUT/summary.json"
```

`METRIC_REPLICATION_PASS` authorizes only a six-update common-descent response
gate. It does not authorize a checkpoint, rollout, 120-step overfit,
calibration, room_0102, room_0201, development, or paper-test access.

