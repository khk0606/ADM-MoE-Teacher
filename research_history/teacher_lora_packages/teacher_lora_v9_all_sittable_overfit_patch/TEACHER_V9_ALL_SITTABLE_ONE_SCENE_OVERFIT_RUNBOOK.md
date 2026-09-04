# Teacher-v9 all-sittable: one-scene overfit smoke

The validated CUDA preflight authorizes this first optimizer gate.  The smoke
starts again from sealed v5r4 plus a fresh zero-output LoRA.  It does **not**
load any v7/v8 candidate and does not export a checkpoint.

Only `room_0101` arrays are loaded.  `room_0102` remains metadata-only and
`room_0201` plus paper-test data remain unread.  A pass requires both fixed
denoising monitors and real 500-step reverse-diffusion maps to show Bed,
normal Chair and High Chair simultaneously for both prompts.

## 1. Verify, install, and run the CPU contract

Run from `~/AMDM`:

```bash
cd ~/AMDM
conda activate afford
set -euo pipefail

sha256sum -c teacher_lora_v9_all_sittable_one_scene_overfit_v1.zip.sha256
unzip -q -o teacher_lora_v9_all_sittable_one_scene_overfit_v1.zip

PATCH=teacher_lora_v9_all_sittable_overfit_patch
python "$PATCH/prepare/validate_teacher_lora_v9_overfit_smoke_package.py"

for F in \
  relational_teacher_v9_all_sittable_contract.py \
  relational_teacher_v9_all_sittable_metrics.py \
  relational_teacher_v9_lora_preflight_contract.py \
  relational_teacher_v9_lora_objective.py \
  relational_teacher_v9_lora_runtime.py \
  validate_relational_teacher_v9_all_sittable_lora_preflight.py \
  fewshot_cdm_common.py \
  fewshot_cdm_lora.py \
  train_fewshot_cdm.py; do
  test -f "prepare/$F" || { echo "[STOP] missing prepare/$F"; exit 1; }
done

cp "$PATCH"/prepare/*.py prepare/
python prepare/test_relational_teacher_v9_overfit_smoke_contract.py
```

The CPU test must print exactly three PASS conclusions.

## 2. Revalidate the authority and run the CUDA smoke

Use the already validated preflight-v3 report from the successful run:

```bash
SRC=data/history_affordance_relational_teacher_v7_hd
DATA=data/history_affordance_relational_teacher_v9_all_sittable_v1
V5DATA=data/history_affordance_v1
V5=$V5DATA/experiments/fewshot_cdm_chair23_bed2_whiteboard12_v5r4
PREFLIGHT=$DATA/experiments/teacher_lora_v9/preflight_s20261005_v3/preflight.json
OUT=$DATA/experiments/teacher_lora_v9/one_scene_overfit_room0101_s20261006_v1

test -f "$PREFLIGHT" || { echo "[STOP] missing validated preflight"; exit 1; }
test ! -e "$OUT" || {
  echo "[STOP] $OUT already exists; preserve it and use a new versioned OUT"
  exit 1
}

python prepare/validate_relational_teacher_v9_all_sittable_lora_preflight.py \
  --report "$PREFLIGHT"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

python -u prepare/run_relational_teacher_v9_one_scene_overfit_smoke.py \
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
  --steps 120 \
  --lr 1e-4 \
  --grad-clip 1 \
  --lora-rank 4 \
  --lora-alpha 8 \
  --seed 20261006 \
  --device cuda:0 \
  --no-progress

python prepare/validate_relational_teacher_v9_one_scene_overfit_smoke.py \
  --summary "$OUT/summary.json"
```

Do not run the validator if the CUDA command raises an exception; `set -e`
already stops the shell.  If CUDA memory is still occupied by another process,
check it with `nvidia-smi` and stop that process before rerunning with a new
versioned `OUT`.

## PASS interpretation

A validated `ONE_SCENE_OVERFIT_PASS` means:

- exactly 120 updates were applied to a fresh LoRA on `room_0101` only;
- Bed `bed_01`, normal Chair `chair_01`, and High Chair `chair_06` are all
  present in each generated watch/write map;
- both Chair roles improve over the frozen v5r4 Base under the locked policy;
- Bed, explicit negatives, prompt invariance, and the three fixed v5 replay
  probes remain inside their previously sealed limits;
- the saved `one_scene_rollout_maps.npz` was produced by actual 500-step
  reverse diffusion with paired Base/candidate randomness;
- no model checkpoint was written, so this deliberately overfit state cannot
  contaminate the next experiment.

A pass authorizes only a **fresh response-3 hyperparameter gate**, again
starting from sealed v5r4 plus zero-init LoRA.  It does not authorize
calibration, a 60-update/full run, `room_0201`, or paper-test access.

Paste both the CUDA conclusion and validator output.  On FAIL, also paste the
`[OK] rollout presence` and `[OK] failed checks` lines; do not loosen the
already locked thresholds.
