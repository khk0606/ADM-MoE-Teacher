# Teacher-v9 all-sittable: fresh LoRA CUDA preflight

This is the first real model-side gate for the new all-sittable target.  It
loads the sealed v5r4 Teacher and installs a fresh zero-output LoRA.  It never
loads v7, v7.1, v8 specialist or late-union candidate state.

The preflight performs zero optimizer updates.  Its purpose is to prevent a
bad loss/data binding from reaching training.

## 1. Verify, install and run CPU contracts

Run from `~/AMDM`:

```bash
cd ~/AMDM
conda activate afford
set -euo pipefail

sha256sum -c teacher_lora_v9_all_sittable_cuda_preflight_v3.zip.sha256
unzip -q -o teacher_lora_v9_all_sittable_cuda_preflight_v3.zip

PATCH=teacher_lora_v9_all_sittable_preflight_patch
python "$PATCH/prepare/validate_teacher_lora_v9_preflight_package.py"

for F in \
  relational_teacher_v9_all_sittable_contract.py \
  relational_teacher_v9_all_sittable_metrics.py \
  validate_relational_teacher_v9_all_sittable_dataset.py \
  fewshot_cdm_common.py \
  fewshot_cdm_lora.py \
  train_fewshot_cdm.py; do
  test -f "prepare/$F" || { echo "[STOP] missing prepare/$F"; exit 1; }
done

cp "$PATCH"/prepare/*.py prepare/
python prepare/test_relational_teacher_v9_lora_preflight_contract.py
```

The CPU contract must print four PASS lines.  In particular it rejects
Bed-only, missing-normal-Chair and missing-High-Chair predictions.

## 2. Revalidate the accepted GT and run CUDA preflight

```bash
SRC=data/history_affordance_relational_teacher_v7_hd
DATA=data/history_affordance_relational_teacher_v9_all_sittable_v1
V5DATA=data/history_affordance_v1
V5=$V5DATA/experiments/fewshot_cdm_chair23_bed2_whiteboard12_v5r4
OUT=$DATA/experiments/teacher_lora_v9/preflight_s20261005_v3

test ! -e "$OUT" || {
  echo "[STOP] $OUT already exists; preserve it and use a new versioned OUT"
  exit 1
}

python prepare/validate_relational_teacher_v9_all_sittable_dataset.py \
  --source-dataset-root "$SRC" \
  --dataset-root "$DATA" \
  --index "$DATA/index.json"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

python -u prepare/preflight_relational_teacher_v9_all_sittable_lora.py \
  --source-dataset-root "$SRC" \
  --dataset-root "$DATA" \
  --index "$DATA/index.json" \
  --v5-dataset-root "$V5DATA" \
  --v5-split "$V5DATA/splits/chair23_bed2_whiteboard12_multistart24_v1.json" \
  --stats-file data/Mean_Std_Cont_HumanML3D_HUMANISE_PROX_contact_cont_joints_0.8_fur.npz \
  --original-checkpoint outputs/CDM-Perceiver-ALL/ckpt/model300000.pt \
  --v5-checkpoint "$V5/fewshot_cdm.pt" \
  --v5-evidence-report "$V5/gate0a_report.json" \
  --output-dir "$OUT" \
  --diffusion-steps 500 \
  --lora-rank 4 \
  --lora-alpha 8 \
  --seed 20261005 \
  --device cuda:0

python prepare/validate_relational_teacher_v9_all_sittable_lora_preflight.py \
  --report "$OUT/preflight.json"
```

If the sealed v5r4 directory is elsewhere, locate it without guessing:

```bash
find data -path '*fewshot_cdm*' -name fewshot_cdm.pt -print
```

Change only `V5` to the directory containing the sealed v5r4 checkpoint and
its `gate0a_report.json`.

## Required meaning of PASS

- `room_0101` and `room_0102` provide the only loaded scene arrays.
- Each scene contributes one GT containing Bed + normal Chair + High Chair.
- Watch/write use byte-identical GT, identical timestep and identical noise.
- The three verified objects receive exactly equal macro weight, independent
  of point count.
- Unverified Chair/Bed points are ignored, not trained as negatives.
- TV, Desk and Whiteboard are the only explicit-negative object roles.
- Bed-only, either missing-Chair, same-object displacement and explicit
  negative-hotspot counterexamples fail the locked policy.
- Fresh zero-init LoRA is bitwise equal to sealed v5r4 on both v9 and v5
  probes; only LoRA parameters receive gradients.
- A hashed `metric_policy.json` is written before any optimizer exists.

A validated `PREFLIGHT_PASS` authorizes only a one-scene overfit smoke.  That
next smoke must demonstrate one generated output containing all three verified
objects simultaneously before response-3 or calibration is allowed.  This
gate does not authorize `room_0201`, long/full training or paper-test access.
