# Teacher-v9.3 exact top-k response gate

The sealed v9.2 result showed three distinct tensors but identical discrete
top-k behavior.  Its preservation terms worked; its hotspot target did not
match the evaluation target.  This gate changes only that faulty optimization
unit and performs a short learning-rate response grid.

Run every command as one complete line.  Do not enable `set -e`, `set -u`, or
`pipefail`.

## 1. Verify, install and run CPU contracts

```bash
cd ~/AMDM
conda activate afford
set +e
set +u
set +o pipefail
sha256sum -c teacher_lora_v93_exact_topk_response_v1.zip.sha256
unzip -q -o teacher_lora_v93_exact_topk_response_v1.zip
PATCH=teacher_lora_v93_exact_topk_patch
python "$PATCH/prepare/validate_teacher_lora_v93_exact_topk_package.py"
for F in relational_teacher_v9_lora_objective.py relational_teacher_v91_active_support_objective.py run_relational_teacher_v91_corrected_one_scene_overfit.py relational_teacher_v92_hotspot_trust_contract.py; do test -f "prepare/$F" || echo "[MISSING] prepare/$F"; done
cp "$PATCH"/prepare/*.py prepare/
python prepare/test_relational_teacher_v93_exact_topk_contract.py
```

Required conclusions are one `PACKAGE_PASS` and three v9.3 contract `PASS`
lines.  A missing dependency or exception is a STOP, but the terminal remains
open.

## 2. Run the fresh CUDA grid

```bash
SRC=data/history_affordance_relational_teacher_v7_hd
DATA=data/history_affordance_relational_teacher_v9_all_sittable_v1
V5DATA=data/history_affordance_v1
V5=$V5DATA/experiments/fewshot_cdm_chair23_bed2_whiteboard12_v5r4
PREFLIGHT=$DATA/experiments/teacher_lora_v9/preflight_s20261005_v3/preflight.json
FAILED92=$DATA/experiments/teacher_lora_v92/hotspot_trust_response6_s20261009_v1/preflight.json
OUT=$DATA/experiments/teacher_lora_v93/exact_topk_response6_s20261010_v1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
test ! -e "$OUT"
```

Run the evaluator as this single line:

```bash
python -u prepare/preflight_relational_teacher_v93_exact_topk.py --failed-v92-report "$FAILED92" --source-dataset-root "$SRC" --dataset-root "$DATA" --index "$DATA/index.json" --v5-dataset-root "$V5DATA" --v5-split "$V5DATA/splits/chair23_bed2_whiteboard12_multistart24_v1.json" --stats-file data/Mean_Std_Cont_HumanML3D_HUMANISE_PROX_contact_cont_joints_0.8_fur.npz --original-checkpoint outputs/CDM-Perceiver-ALL/ckpt/model300000.pt --v5-checkpoint "$V5/fewshot_cdm.pt" --v5-evidence-report "$V5/gate0a_report.json" --preflight-report "$PREFLIGHT" --output-dir "$OUT" --diffusion-steps 500 --seed 20261010 --device cuda:0
```

Then independently recompute the saved artifact:

```bash
python prepare/validate_relational_teacher_v93_exact_topk.py --report "$OUT/preflight.json"
```

## Gate meaning

PASS requires, separately for Bed, normal Chair and High Chair:

- mean top-k overlap strictly improves;
- neither watch nor write top-k gets worse;
- recall and active-support MAE remain bounded;
- the exact swap loss decreases.

It also requires Base-background trust, prompt invariance, explicit-negative
suppression and v5 replay retention.  A validated PASS authorizes only a short
actual reverse-diffusion rollout canary of the selected LR.  It does not
authorize 120-step overfit, calibration, `room_0102`, `room_0201`, or paper
test access.

If all candidates fail, paste all three `[EXACT-TOPK ...]` lines and the
validator conclusion.  Do not lower the metric or train longer.
