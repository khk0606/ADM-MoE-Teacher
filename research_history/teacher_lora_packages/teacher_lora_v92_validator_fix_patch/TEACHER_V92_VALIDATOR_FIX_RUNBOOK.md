# Teacher-v9.2 validator correction and exact diagnostic

Run every command as one complete line.  No CUDA rerun is needed.

```bash
cd ~/AMDM
conda activate afford
set +e
set +u
set +o pipefail
sha256sum -c teacher_lora_v92_validator_fix_v1.zip.sha256
unzip -q -o teacher_lora_v92_validator_fix_v1.zip
PATCH=teacher_lora_v92_validator_fix_patch
python "$PATCH/prepare/validate_teacher_lora_v92_validator_fix_package.py"
cp "$PATCH"/prepare/*.py prepare/
python prepare/test_relational_teacher_v92_validator_fix.py
DATA=data/history_affordance_relational_teacher_v9_all_sittable_v1
OUT=$DATA/experiments/teacher_lora_v92/hotspot_trust_response6_s20261009_v1
python prepare/validate_relational_teacher_v92_hotspot_trust_v2.py --report "$OUT/preflight.json"
python prepare/summarize_relational_teacher_v92_hotspot_trust.py --report "$OUT/preflight.json"
```

The validator must still conclude `HOTSPOT_TRUST_FAIL`; it now reaches the
conclusion by recomputing the complete saved artifact instead of stopping on a
benign CPU/CUDA `logsumexp` rounding difference.  Paste the full output from
the final summary command before any new optimization run.
