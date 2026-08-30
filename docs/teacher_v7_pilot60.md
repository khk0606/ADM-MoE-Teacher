# Teacher-v7: fresh bounded 60-update pilot

The sealed invariance recovery passed with `watch_write_invariance=0.75`.
This gate repeats its first 12 updates exactly, but starts again from the sealed
v5r4 Teacher. It must not load the recovery or failed calibration checkpoint.

## 1. Install and run CPU contract tests

```bash
cd ~/ADM-MoE-Teacher
conda activate afford

python prepare/test_relational_teacher_v7_hd_calibration_contract.py
python prepare/test_relational_teacher_v7_hd_recovery_contract.py
python prepare/test_relational_teacher_v7_hd_pilot_contract.py
```

## 2. Run the fresh CUDA pilot

Use the exact recovery summary shown below. The output directory must not
already exist.

```bash
DATA=data/history_affordance_relational_teacher_v7_hd
V5DATA=data/history_affordance_v1
V5=$V5DATA/experiments/fewshot_cdm_chair23_bed2_whiteboard12_v5r4
RECOVERY=$DATA/experiments/teacher_lora_v7_hd/calibration12_invariance075_recovery_s20260909_v1/summary.json
OUT=$DATA/experiments/teacher_lora_v7_hd/pilot60_invariance075_s20260910_v1

python -u prepare/train_relational_teacher_v7_hd_lora_pilot.py \
  --dataset-root "$DATA" \
  --index "$DATA/index.json" \
  --v5-dataset-root "$V5DATA" \
  --v5-split "$V5DATA/splits/chair23_bed2_whiteboard12_multistart24_v1.json" \
  --stats-file data/Mean_Std_Cont_HumanML3D_HUMANISE_PROX_contact_cont_joints_0.8_fur.npz \
  --original-checkpoint outputs/CDM-Perceiver-ALL/ckpt/model300000.pt \
  --v5-checkpoint "$V5/fewshot_cdm.pt" \
  --recovery-summary "$RECOVERY" \
  --output-dir "$OUT" \
  --steps 60 \
  --lr 4e-5 \
  --grad-clip 1 \
  --lora-rank 4 \
  --lora-alpha 8 \
  --seed 20260909 \
  --device cuda:0

python prepare/validate_relational_teacher_v7_hd_lora_pilot.py \
  --summary "$OUT/summary.json"
```

If the local v5r4 directory differs, locate the sealed checkpoint without
guessing:

```bash
find data -path '*fewshot_cdm_v5r4*/fewshot_cdm.pt' -print
```

Then change only `V5` to the directory containing that file.

## Gate interpretation

`PILOT_PASS` requires at least one of steps `12/24/36/48/60` to satisfy all of:

- semantic loss improves;
- High-Desk dense contact improves;
- watch/write invariance is within +2% of fresh initialization;
- new dense contact and negative suppression are within +2%;
- v5 replay degradation is within +1%;
- preservation is at most `1e-3` and LoRA changed from zero.

At most two checkpoints are shortlisted, ranked by objective proxy, High-Desk
dense loss, semantic loss, then update number. A pass authorizes only paired
train-only reverse-diffusion evaluation of those shortlisted checkpoints. It
does not authorize development payload access, a long/full training run, or a
paper test.
