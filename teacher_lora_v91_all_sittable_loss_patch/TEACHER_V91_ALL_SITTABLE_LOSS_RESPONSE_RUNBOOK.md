# Teacher-v9.1: active-support/ranking loss-response gate

The sealed Teacher-v9 smoke failed for a specific reason: its fixed denoising
panel reduced Bed/Chair/High-Chair active-support error, but Chair hotspot
ordering stayed below the absolute gate and reverse diffusion collapsed to
Bed.  Training longer with the same objective is not authorized.

This gate starts from sealed v5r4 plus a fresh zero-output LoRA for every
candidate.  The failed 120-update output is hash-checked as diagnostic evidence
and no failed state is loaded.

## 1. Verify, install and run the CPU contract

Run from `~/AMDM`:

```bash
cd ~/AMDM
conda activate afford
set -euo pipefail

sha256sum -c teacher_lora_v91_all_sittable_loss_response_v1.zip.sha256
unzip -q -o teacher_lora_v91_all_sittable_loss_response_v1.zip

PATCH=teacher_lora_v91_all_sittable_loss_patch
python "$PATCH/prepare/validate_teacher_lora_v91_loss_response_package.py"

for F in relational_teacher_v9_all_sittable_contract.py relational_teacher_v9_all_sittable_metrics.py relational_teacher_v9_lora_preflight_contract.py relational_teacher_v9_lora_objective.py relational_teacher_v9_lora_runtime.py relational_teacher_v9_overfit_smoke_contract.py run_relational_teacher_v9_one_scene_overfit_smoke.py validate_relational_teacher_v9_one_scene_overfit_smoke.py validate_relational_teacher_v9_all_sittable_lora_preflight.py fewshot_cdm_common.py fewshot_cdm_lora.py train_fewshot_cdm.py; do
  test -f "prepare/$F" || { echo "[STOP] missing prepare/$F"; exit 1; }
done

cp "$PATCH"/prepare/*.py prepare/
python prepare/test_relational_teacher_v91_loss_response_contract.py
```

The package validator must print two PASS lines and the CPU contract must print
three PASS lines.  Stop on any exception.

## 2. Revalidate the sealed evidence

```bash
SRC=data/history_affordance_relational_teacher_v7_hd
DATA=data/history_affordance_relational_teacher_v9_all_sittable_v1
V5DATA=data/history_affordance_v1
V5=$V5DATA/experiments/fewshot_cdm_chair23_bed2_whiteboard12_v5r4
PREFLIGHT=$DATA/experiments/teacher_lora_v9/preflight_s20261005_v3/preflight.json
FAILED=$DATA/experiments/teacher_lora_v9/one_scene_overfit_room0101_s20261006_v1/summary.json
OUT=$DATA/experiments/teacher_lora_v91/loss_response_s20261007_v1

test -f "$PREFLIGHT" || { echo "[STOP] missing validated v9 preflight"; exit 1; }
test -f "$FAILED" || { echo "[STOP] missing validated failed smoke"; exit 1; }
test ! -e "$OUT" || { echo "[STOP] preserve existing OUT and choose a new versioned OUT"; exit 1; }

python prepare/validate_relational_teacher_v9_all_sittable_lora_preflight.py --report "$PREFLIGHT"
python prepare/validate_relational_teacher_v9_one_scene_overfit_smoke.py --summary "$FAILED"
```

Both commands must reproduce their earlier integrity conclusions: preflight
PASS and the scientifically valid smoke FAIL.

## 3. Run the fresh CUDA response grid

```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

python -u prepare/preflight_relational_teacher_v91_loss_response.py \
  --failed-smoke-summary "$FAILED" \
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
  --output-dir "$OUT" \
  --diffusion-steps 500 \
  --seed 20261007 \
  --device cuda:0

python prepare/validate_relational_teacher_v91_loss_response.py --report "$OUT/preflight.json"
```

This is nine optimizer updates total: three candidates times three updates.
All candidates use the same room, prompt pair, timesteps, noise and learning
rate, and each starts from the exact same zero-init LoRA tensor state with a
fresh AdamW optimizer.

## PASS meaning

A candidate is admissible only if all of the following hold simultaneously:

- active-support loss falls for Bed, normal Chair and High Chair separately;
- the within-object hotspot-ranking surrogate falls;
- watch/write invariance stays within +2% of its fresh baseline;
- explicit-negative mean addition is at most 0.002;
- mean sealed-v5 replay degradation stays within +1%.

The grid never evaluates `room_0102`, `room_0201` or paper-test arrays, and it
does not save a model checkpoint.  A validated `LOSS_RESPONSE_PASS` authorizes
only a new corrected one-scene overfit smoke using the selected loss weights.
It does not authorize response-3, calibration, long training, development, or
paper-test access.

Paste all three `[RESPONSE ...]` lines, the CUDA conclusion, and the validator
conclusion.  A FAIL remains useful evidence; do not relax the accepted metric
thresholds or resume the failed 120-update state.
