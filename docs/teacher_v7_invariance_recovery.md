# Teacher-v7 invariance recovery

This is a fresh A/B recovery for the sealed v7 calibration near miss. The
failed artifact is immutable and remains `CALIBRATION_FAIL`. The recovery
starts again from the sealed v5r4 Teacher plus zero-initialized LoRA and never
loads the failed step-12 checkpoint.

The seed, train rows, four-stratum batches, High-Desk coverage, prompt pairs,
v5 replay rows, timesteps, noise, optimizer, learning rate and every other
objective weight are unchanged. The only changed hyperparameter is:

```text
watch/write invariance weight: 0.5 -> 0.75
```

`room_0201` remains metadata-only.

## Contract tests

```bash
cd ~/ADM-MoE-Teacher
conda activate afford

python -m py_compile \
  prepare/recover_relational_teacher_v7_hd_lora_invariance.py \
  prepare/validate_relational_teacher_v7_hd_lora_invariance_recovery.py

python prepare/test_relational_teacher_v7_hd_calibration_contract.py
python prepare/test_relational_teacher_v7_hd_recovery_contract.py
```

## Run the fresh A/B recovery

```bash
DATA=data/history_affordance_relational_teacher_v7_hd
V5DATA=data/history_affordance_v1
V5=$V5DATA/experiments/fewshot_cdm_chair23_bed2_whiteboard12_v5r4
PREFLIGHT=$DATA/experiments/teacher_lora_v7_hd/preflight_s20260908_v1/preflight.json
FAILED=$DATA/experiments/teacher_lora_v7_hd/calibration12_s20260909_v1/summary.json
OUT=$DATA/experiments/teacher_lora_v7_hd/calibration12_invariance075_recovery_s20260909_v1

python -u prepare/recover_relational_teacher_v7_hd_lora_invariance.py \
  --dataset-root "$DATA" \
  --index "$DATA/index.json" \
  --v5-dataset-root "$V5DATA" \
  --v5-split "$V5DATA/splits/chair23_bed2_whiteboard12_multistart24_v1.json" \
  --stats-file data/Mean_Std_Cont_HumanML3D_HUMANISE_PROX_contact_cont_joints_0.8_fur.npz \
  --original-checkpoint outputs/CDM-Perceiver-ALL/ckpt/model300000.pt \
  --v5-checkpoint "$V5/fewshot_cdm.pt" \
  --preflight-report "$PREFLIGHT" \
  --failed-calibration-summary "$FAILED" \
  --output-dir "$OUT" \
  --steps 12 \
  --lr 4e-5 \
  --grad-clip 1 \
  --lora-rank 4 \
  --lora-alpha 8 \
  --seed 20260909 \
  --device cuda:0

python prepare/validate_relational_teacher_v7_hd_lora_invariance_recovery.py \
  --summary "$OUT/summary.json"
```

## Decision rule

Continue only when both commands print `RECOVERY_PASS`. The original 2%
invariance limit is unchanged; the recovery must satisfy it rather than relax
it. High-Desk dense, overall semantic, v5 replay, negative suppression,
preservation, frozen-CDM gradient, and leakage checks also remain unchanged.

A PASS authorizes only the next bounded train-only pilot. It does not authorize
development evaluation, a long/full run, or paper testing.
