# Teacher-v9.2: per-object hotspot and background-trust response gate

The corrected v9.1 smoke proved that merely increasing values on each object
does not learn the dense motion-contact shape.  Its three response candidates
were also nearly scalar copies; after global gradient clipping their update
directions were effectively the same.  Continuing to 120 updates then raised
explicit-negative surfaces and damaged prompt/v5 retention.

This gate starts over from sealed v5r4 for each candidate.  It runs six paired
updates on `room_0101` only and requires Bed, normal Chair and High Chair top-k
overlap to improve separately.  It also preserves the frozen Base outside the
three verified objects and drives TV/Desk/Whiteboard toward zero.

It saves maps and measurements only, never a model checkpoint.

## 1. Install without terminal auto-exit

Run each displayed line separately.  Do not enable `set -e`, `set -u`, or
`pipefail`; a scientific FAIL must return you to the prompt instead of closing
the terminal.

```bash
cd ~/AMDM
conda activate afford
set +e
set +u
set +o pipefail
sha256sum -c teacher_lora_v92_hotspot_trust_response_v1.zip.sha256
unzip -q -o teacher_lora_v92_hotspot_trust_response_v1.zip
PATCH=teacher_lora_v92_hotspot_trust_patch
python "$PATCH/prepare/validate_teacher_lora_v92_hotspot_trust_package.py"
cp "$PATCH"/prepare/*.py prepare/
python prepare/test_relational_teacher_v92_hotspot_trust_contract.py
```

Required conclusions are one `PACKAGE_PASS` and three CPU-contract `PASS`
lines.  Stop on an exception, but the shell will remain open.

## 2. Run the fresh CUDA response grid

Set these paths one line at a time:

```bash
SRC=data/history_affordance_relational_teacher_v7_hd
DATA=data/history_affordance_relational_teacher_v9_all_sittable_v1
V5DATA=data/history_affordance_v1
V5=$V5DATA/experiments/fewshot_cdm_chair23_bed2_whiteboard12_v5r4
PREFLIGHT=$DATA/experiments/teacher_lora_v9/preflight_s20261005_v3/preflight.json
RESPONSE=$DATA/experiments/teacher_lora_v91/loss_response_s20261007_v1/preflight.json
FAILED=$DATA/experiments/teacher_lora_v91/corrected_overfit_room0101_s20261008_v1/summary.json
OUT=$DATA/experiments/teacher_lora_v92/hotspot_trust_response6_s20261009_v1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

Check that the new output does not already exist:

```bash
test ! -e "$OUT"
```

Then run this as one line (there is no trailing backslash and therefore no
continuation `>` prompt):

```bash
python -u prepare/preflight_relational_teacher_v92_hotspot_trust.py --failed-corrected-overfit-summary "$FAILED" --selected-response-report "$RESPONSE" --source-dataset-root "$SRC" --dataset-root "$DATA" --index "$DATA/index.json" --v5-dataset-root "$V5DATA" --v5-split "$V5DATA/splits/chair23_bed2_whiteboard12_multistart24_v1.json" --stats-file data/Mean_Std_Cont_HumanML3D_HUMANISE_PROX_contact_cont_joints_0.8_fur.npz --original-checkpoint outputs/CDM-Perceiver-ALL/ckpt/model300000.pt --v5-checkpoint "$V5/fewshot_cdm.pt" --v5-evidence-report "$V5/gate0a_report.json" --preflight-report "$PREFLIGHT" --output-dir "$OUT" --diffusion-steps 500 --seed 20261009 --device cuda:0
```

After it returns to the prompt, validate the saved arrays independently:

```bash
python prepare/validate_relational_teacher_v92_hotspot_trust.py --report "$OUT/preflight.json"
```

## PASS meaning

A validated `HOTSPOT_TRUST_PASS` means at least one genuinely different loss
ratio improved top-k overlap for all of Bed, normal Chair and High Chair on the
fixed paired panel, while retaining recall, active-support MAE, prompt
invariance, v5 replay and explicit-negative bounds.

PASS authorizes only a short actual reverse-diffusion canary of the selected
candidate.  It does not authorize another 120-step overfit, calibration,
`room_0102`, `room_0201`, or paper-test access.

If it prints `HOTSPOT_TRUST_FAIL`, do not change thresholds and do not train
longer.  Paste all three `[HOTSPOT ...]` lines plus the validator conclusion;
those measurements determine whether the remaining problem is hotspot
ordering, preservation, or both.

