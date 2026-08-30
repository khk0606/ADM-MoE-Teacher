# Teacher-v7: selected step-12 train-only K=3 canary

This gate reuses the sealed K=1 maps as generation 0 and generates only
generations 1 and 2. Base and step-12 use paired initial/reverse noise for every
case. room_0201 remains unread.

## Contract test

```bash
cd ~/ADM-MoE-Teacher
conda activate afford

python prepare/test_relational_teacher_v7_hd_k3_contract.py
```

## Run K=3

```bash
DATA=data/history_affordance_relational_teacher_v7_hd
K1=$DATA/experiments/teacher_lora_v7_hd/selected_step12_train_canary_k1_s20260911_v1/summary.json
OUT=$DATA/experiments/teacher_lora_v7_hd/selected_step12_train_canary_k3_s20260911_v1

python -u prepare/evaluate_relational_teacher_v7_hd_lora_selected_k3_canary.py \
  --k1-summary "$K1" \
  --output-dir "$OUT" \
  --diffusion-steps 500 \
  --num-generations 3 \
  --seed 20260911 \
  --device cuda:0 \
  --no-progress

python prepare/validate_relational_teacher_v7_hd_lora_selected_k3_canary.py \
  --summary "$OUT/summary.json"
```

The run generates 76 paired evaluation draws plus two determinism repeats,
for 78 new reverse-diffusion draws. `SELECTED_K3_PASS` requires pooled overall
and High-Desk improvement, at least two improving generations for both, bounded
semantic/invariance/v5 behavior, and bounded metrics in both train rooms.

A pass authorizes only the full train-only K=3 evaluation over all 48 motions
from the two training rooms.
It does not authorize room_0201, development evaluation, or paper-test access.
