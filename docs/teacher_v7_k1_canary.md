# Teacher-v7: selected step-12 train-only K=1 canary

The sealed 60-update pilot shortlisted only update 12. This gate compares the
frozen v5r4 Base Teacher and the merged update-12 Teacher with identical initial
and reverse-diffusion noise. It reads only room_0101 and room_0102.

## Contract test

```bash
cd ~/ADM-MoE-Teacher
conda activate afford

python prepare/test_relational_teacher_v7_hd_rollout_contract.py
```

## Run the paired canary

```bash
DATA=data/history_affordance_relational_teacher_v7_hd
V5DATA=data/history_affordance_v1
V5=$V5DATA/experiments/fewshot_cdm_chair23_bed2_whiteboard12_v5r4
PILOT=$DATA/experiments/teacher_lora_v7_hd/pilot60_invariance075_s20260910_v1/summary.json
OUT=$DATA/experiments/teacher_lora_v7_hd/selected_step12_train_canary_k1_s20260911_v1

python -u prepare/evaluate_relational_teacher_v7_hd_lora_rollout_canary.py \
  --dataset-root "$DATA" \
  --index "$DATA/index.json" \
  --v5-dataset-root "$V5DATA" \
  --v5-split "$V5DATA/splits/chair23_bed2_whiteboard12_multistart24_v1.json" \
  --stats-file data/Mean_Std_Cont_HumanML3D_HUMANISE_PROX_contact_cont_joints_0.8_fur.npz \
  --original-checkpoint outputs/CDM-Perceiver-ALL/ckpt/model300000.pt \
  --v5-checkpoint "$V5/fewshot_cdm.pt" \
  --pilot-summary "$PILOT" \
  --output-dir "$OUT" \
  --diffusion-steps 500 \
  --seed 20260911 \
  --device cuda:0 \
  --no-progress

python prepare/validate_relational_teacher_v7_hd_lora_rollout_canary.py \
  --summary "$OUT/summary.json"
```

The evaluator performs 16 paired relational cases: four strata per train room,
each with both watch and write prompts. Four of those cases are High-Desk
Chair cases. It also performs one v5 replay case for chair, bed and whiteboard.
Including the determinism repeat, this is 39 reverse-diffusion draws.

`CANARY_PASS` authorizes only a train-only multi-seed K=3 canary for step 12.
It does not authorize room_0201 access, full training, or paper-test access.
