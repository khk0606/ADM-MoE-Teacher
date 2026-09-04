# Teacher-v9.1: corrected one-scene overfit smoke

The validated loss-response grid selected
`support2_rank025_preserve2`.  This gate locks that result and tests whether it
survives a real 500-step reverse-diffusion rollout.

It starts from sealed v5r4 plus fresh zero-init LoRA.  It does not load the
failed v9 overfit state or any response-grid state, and it writes no model
checkpoint.

## 1. Verify, install and test

Run from `~/AMDM`:

```bash
cd ~/AMDM
conda activate afford
set -euo pipefail

sha256sum -c teacher_lora_v91_corrected_overfit_v1.zip.sha256
unzip -q -o teacher_lora_v91_corrected_overfit_v1.zip

PATCH=teacher_lora_v91_corrected_overfit_patch
python "$PATCH/prepare/validate_teacher_lora_v91_corrected_overfit_package.py"

for F in relational_teacher_v9_all_sittable_contract.py relational_teacher_v9_all_sittable_metrics.py relational_teacher_v9_lora_preflight_contract.py relational_teacher_v9_lora_objective.py relational_teacher_v9_lora_runtime.py relational_teacher_v9_overfit_smoke_contract.py run_relational_teacher_v9_one_scene_overfit_smoke.py validate_relational_teacher_v9_one_scene_overfit_smoke.py validate_relational_teacher_v91_loss_response.py validate_relational_teacher_v9_all_sittable_lora_preflight.py fewshot_cdm_common.py fewshot_cdm_lora.py train_fewshot_cdm.py; do
  test -f "prepare/$F" || { echo "[STOP] missing prepare/$F"; exit 1; }
done

cp "$PATCH"/prepare/*.py prepare/
python prepare/test_relational_teacher_v91_corrected_overfit_contract.py
```

The package validator prints two PASS lines.  The CPU contract prints three.

## 2. Revalidate the selected response

```bash
SRC=data/history_affordance_relational_teacher_v7_hd
DATA=data/history_affordance_relational_teacher_v9_all_sittable_v1
V5DATA=data/history_affordance_v1
V5=$V5DATA/experiments/fewshot_cdm_chair23_bed2_whiteboard12_v5r4
PREFLIGHT=$DATA/experiments/teacher_lora_v9/preflight_s20261005_v3/preflight.json
RESPONSE=$DATA/experiments/teacher_lora_v91/loss_response_s20261007_v1/preflight.json
OUT=$DATA/experiments/teacher_lora_v91/corrected_overfit_room0101_s20261008_v1

test -f "$PREFLIGHT" || { echo "[STOP] missing v9 preflight"; exit 1; }
test -f "$RESPONSE" || { echo "[STOP] missing v9.1 loss-response PASS"; exit 1; }
test ! -e "$OUT" || { echo "[STOP] preserve existing OUT and choose a new versioned OUT"; exit 1; }

python prepare/validate_relational_teacher_v9_all_sittable_lora_preflight.py --report "$PREFLIGHT"
python prepare/validate_relational_teacher_v91_loss_response.py --report "$RESPONSE"
```

Both validators must print PASS conclusions and the selected candidate must be
exactly `support2_rank025_preserve2`.

## 3. Run the corrected CUDA smoke

```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

python -u prepare/run_relational_teacher_v91_corrected_one_scene_overfit.py \
  --source-dataset-root "$SRC" \
  --dataset-root "$DATA" \
  --index "$DATA/index.json" \
  --v5-dataset-root "$V5DATA" \
  --v5-split "$V5DATA/splits/chair23_bed2_whiteboard12_multistart24_v1.json" \
  --stats-file data/Mean_Std_Cont_HumanML3D_HUMANISE_PROX_contact_cont_joints_0.8_fur.npz \
  --original-checkpoint outputs/CDM-Perceiver-ALL/ckpt/model300000.pt \
  --v5-checkpoint "$V5/fewshot_cdm.pt" \
  --v5-evidence-report "$V5/gate0a_report.json" \
  --preflight-report "$PREFLIGHT" \
  --loss-response-report "$RESPONSE" \
  --output-dir "$OUT" \
  --diffusion-steps 500 \
  --steps 120 \
  --lr 4e-5 \
  --grad-clip 1 \
  --lora-rank 4 \
  --lora-alpha 8 \
  --seed 20261008 \
  --device cuda:0 \
  --no-progress

python prepare/validate_relational_teacher_v91_corrected_one_scene_overfit.py --summary "$OUT/summary.json"
```

## Locked selection and PASS meaning

The fixed denoising monitor runs only at updates
`0,3,6,12,24,40,60,80,100,120`.  Among eligible positive steps, selection
maximizes the worst Bed/Chair/High-Chair top-k overlap, then the worst recall,
then minimizes the worst active-support MAE, then prefers the earlier step.
This ordering is fixed before CUDA execution.

`CORRECTED_OVERFIT_PASS` requires both watch and write reverse-diffusion maps
to contain Bed, normal Chair and High Chair simultaneously under the original
locked v9 metric policy.  Prompt invariance, explicit negatives and all three
v5 replay probes must remain within their original limits.

Only `room_0101` arrays are read.  `room_0102`, `room_0201` and paper-test
arrays remain unread.  A validated pass authorizes only a fresh response-3
gate; it does not authorize calibration or long/full training.

Paste the monitor lines, both rollout-presence dictionaries, the CUDA
conclusion and validator conclusion.  Do not loosen thresholds on FAIL.
