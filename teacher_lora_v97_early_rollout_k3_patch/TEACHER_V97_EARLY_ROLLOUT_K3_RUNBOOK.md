# Teacher-v9.7: fresh early-state actual 500-step K=3 gate

This gate answers one question before any further loss engineering: does the
fresh v9.1 trajectory already contain an early state whose **final generated
affordance maps** contain Bed, normal Chair and High Chair together?

It reproduces updates 1 through 12 exactly, keeps step 3/6/12 LoRA tensors only
in RAM, then performs paired Base/candidate 500-step reverse diffusion at K=3
for `Sit anywhere to watch.` and `Sit anywhere to write.`. No checkpoint is
saved. `room_0102`, `room_0201` and paper-test arrays remain unread.

Every command below is one complete line. None ends in `\`. The commands also
disable inherited fail-fast shell flags so a legitimate FAIL or exception does
not close the interactive terminal.

## 1. Verify, install and run the CPU contract

```bash
cd ~/AMDM
conda activate afford
set +e
set +u
set +o pipefail
sha256sum -c teacher_lora_v97_early_rollout_k3_v1.zip.sha256
unzip -q -o teacher_lora_v97_early_rollout_k3_v1.zip
PATCH=teacher_lora_v97_early_rollout_k3_patch
python "$PATCH/prepare/validate_teacher_lora_v97_early_rollout_k3_package.py"
MISSING=0; for F in fewshot_cdm_common.py fewshot_cdm_lora.py relational_teacher_v9_all_sittable_contract.py relational_teacher_v9_all_sittable_metrics.py relational_teacher_v9_lora_objective.py relational_teacher_v9_lora_preflight_contract.py relational_teacher_v9_lora_runtime.py relational_teacher_v91_active_support_objective.py relational_teacher_v91_corrected_overfit_contract.py relational_teacher_v91_loss_response_contract.py run_relational_teacher_v91_corrected_one_scene_overfit.py train_fewshot_cdm.py; do if [ ! -f "prepare/$F" ]; then echo "[STOP] missing prepare/$F"; MISSING=1; fi; done; if [ "$MISSING" -eq 0 ]; then echo "[OK] dependencies present"; else echo "[STOP] paste the missing-file lines; do not run CUDA"; fi
cp "$PATCH"/prepare/*.py prepare/
python prepare/test_relational_teacher_v97_early_rollout_k3_contract.py
```

Continue only after the package validator prints three PASS lines, the
dependency check prints `[OK] dependencies present`, and the CPU contract
prints three PASS lines.

## 2. Bind the sealed inputs and a new output

```bash
SRC=data/history_affordance_relational_teacher_v7_hd
DATA=data/history_affordance_relational_teacher_v9_all_sittable_v1
V5DATA=data/history_affordance_v1
V5=$V5DATA/experiments/fewshot_cdm_chair23_bed2_whiteboard12_v5r4
PREFLIGHT=$DATA/experiments/teacher_lora_v9/preflight_s20261005_v3/preflight.json
RESPONSE=$DATA/experiments/teacher_lora_v91/loss_response_s20261007_v1/preflight.json
FAILED=$DATA/experiments/teacher_lora_v91/corrected_overfit_room0101_s20261008_v1/summary.json
OUT=$DATA/experiments/teacher_lora_v97/early_rollout_k3_s20261015_v1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
if [ -e "$OUT" ]; then echo "[STOP] OUT already exists; preserve it and use a new versioned OUT"; else echo "[OK] OUT is new"; fi
```

Do not continue unless the last line is `[OK] OUT is new`.

## 3. Revalidate all three authorities

```bash
python prepare/validate_relational_teacher_v9_all_sittable_lora_preflight.py --report "$PREFLIGHT"
python prepare/validate_relational_teacher_v91_loss_response.py --report "$RESPONSE"
python prepare/validate_relational_teacher_v91_corrected_one_scene_overfit.py --summary "$FAILED"
```

The expected conclusions are `PREFLIGHT_PASS`, `LOSS_RESPONSE_PASS`, and the
already sealed `CORRECTED_OVERFIT_FAIL` integrity result. The last FAIL is the
input diagnosis, not a new failure. Its validator must still print all four
PASS evidence lines.

## 4. Run the fresh CUDA K=3 gate

```bash
python -u prepare/evaluate_relational_teacher_v97_early_rollout_k3.py --failed-v91-summary "$FAILED" --source-dataset-root "$SRC" --dataset-root "$DATA" --index "$DATA/index.json" --v5-dataset-root "$V5DATA" --v5-split "$V5DATA/splits/chair23_bed2_whiteboard12_multistart24_v1.json" --stats-file data/Mean_Std_Cont_HumanML3D_HUMANISE_PROX_contact_cont_joints_0.8_fur.npz --original-checkpoint outputs/CDM-Perceiver-ALL/ckpt/model300000.pt --v5-checkpoint "$V5/fewshot_cdm.pt" --v5-evidence-report "$V5/gate0a_report.json" --preflight-report "$PREFLIGHT" --loss-response-report "$RESPONSE" --output-dir "$OUT" --diffusion-steps 500 --steps 12 --lr 4e-5 --grad-clip 1 --lora-rank 4 --lora-alpha 8 --training-seed 20261008 --rollout-seed 20261015 --device cuda:0 --no-progress
```

The run first prints exact `[REPRODUCTION_PASS]` rows for steps 3, 6 and 12.
It then makes 28 reverse-diffusion draws: 6 Base, 18 candidate and 4 exact
determinism repeats. On a 6 GB GPU this is intentionally sequential; it does
not retain GPU rollout batches.

## 5. Deep validation and readable diagnosis

Run both commands whether the result is PASS or FAIL:

```bash
python prepare/validate_relational_teacher_v97_early_rollout_k3.py --summary "$OUT/summary.json"
python prepare/summarize_relational_teacher_v97_early_rollout_k3.py --summary "$OUT/summary.json"
```

`EARLY_ROLLOUT_K3_PASS` means at least one of step 3/6/12 satisfies the locked
continuous three-object K=3 policy. Exact Top-k is printed only as a diagnostic;
it cannot reject a candidate due to a one-point rank swap. Small background
values are not confused with object presence, while major explicit-negative
growth and v5 regression remain independent guards.

A PASS authorizes only an exact fresh rerun that confirms and persists the
selected early state. A legitimate FAIL authorizes only a rollout-state or
on-policy response preflight based on these saved generated maps. Neither
result authorizes `room_0102`, `room_0201`, long training or paper-test access.
