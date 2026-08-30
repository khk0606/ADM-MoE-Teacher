# Teacher-v7 LoRA CUDA preflight

Run this gate only after the three-scene v7 dense dataset reports
`DENSE_DATASET_PASS` with 72 motions and 144 prompt-expanded rows.

This preflight performs no optimizer update. It loads the sealed v5r4 Teacher,
installs a fresh zero-initialized LoRA, and reads tensors only from the two train
scenes. Each train probe is required to be one of the new `hc_hd` High-Desk
motions targeting `chair_06`; room_0201 remains metadata-only.

```bash
cd ~/ADM-MoE-Teacher
conda activate afford

python prepare/test_relational_teacher_v7_hd_preflight_contract.py

DATA=data/history_affordance_relational_teacher_v7_hd
V5=data/history_affordance_v1/experiments/fewshot_cdm_chair23_bed2_whiteboard12_v5r4
OUT=$DATA/experiments/teacher_lora_v7_hd/preflight_s20260908_v1

python prepare/validate_relational_teacher_v7_hd_dataset.py \
  --dataset-root "$DATA" \
  --index "$DATA/index.json"

python -u prepare/preflight_relational_teacher_v7_hd_lora.py \
  --dataset-root "$DATA" \
  --index "$DATA/index.json" \
  --stats-file data/Mean_Std_Cont_HumanML3D_HUMANISE_PROX_contact_cont_joints_0.8_fur.npz \
  --original-checkpoint outputs/CDM-Perceiver-ALL/ckpt/model300000.pt \
  --v5-checkpoint "$V5/fewshot_cdm.pt" \
  --v5-evidence-report "$V5/gate0a_report.json" \
  --report "$OUT/preflight.json" \
  --diffusion-steps 500 \
  --lora-rank 4 \
  --lora-alpha 8 \
  --seed 20260908 \
  --device cuda:0

python prepare/validate_relational_teacher_v7_hd_lora_preflight.py \
  --report "$OUT/preflight.json"
```

Required result:

- exact 72-motion/144-row v7 index binding;
- fresh LoRA output is bitwise identical to the sealed v5r4 Teacher at zero init;
- exactly 31 LoRA modules and no trainable frozen-CDM parameter;
- the dense loss is evaluated on genuine new High-Desk `hc_hd` train rows;
- dense, semantic, watch/write invariance, preservation, regularizer, and total
  gradients are finite and reach LoRA;
- room_0201 development arrays remain unread;
- only a fresh 12-update train-only calibration is authorized.

Do not load the v6 LoRA checkpoint. Do not start a 60-update, development, or
full run from this preflight alone.
