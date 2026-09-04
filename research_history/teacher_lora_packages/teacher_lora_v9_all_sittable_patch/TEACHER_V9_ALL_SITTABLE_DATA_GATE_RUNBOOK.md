# Teacher-v9 all-sittable GT: Ubuntu data and visual gate

This gate fixes the supervision unit before any new LoRA training.  It builds a
new dataset and leaves the sealed Teacher-v7 dataset unchanged.

## 1. Verify and install

Run from `~/AMDM`:

```bash
cd ~/AMDM
conda activate afford
set -e

sha256sum -c teacher_lora_v9_all_sittable_data_gate_v1.zip.sha256
unzip -q -o teacher_lora_v9_all_sittable_data_gate_v1.zip

PATCH=teacher_lora_v9_all_sittable_patch
python "$PATCH/prepare/validate_teacher_lora_v9_all_sittable_package.py"
cp "$PATCH"/prepare/*.py prepare/

python prepare/test_relational_teacher_v9_all_sittable_contract.py
python prepare/test_relational_teacher_v9_all_sittable_roundtrip.py
python prepare/test_relational_teacher_v9_all_sittable_viewer_common.py
```

The package validator and all three test commands must print PASS conclusions.
Stop on any exception.

## 2. Build a fresh scene-level GT dataset

```bash
SRC=data/history_affordance_relational_teacher_v7_hd
OUT=data/history_affordance_relational_teacher_v9_all_sittable_v1
set -e

test ! -e "$OUT" || {
  echo "[STOP] $OUT already exists; do not overwrite it"
  exit 1
}

python prepare/build_relational_teacher_v9_all_sittable_dataset.py \
  --source-dataset-root "$SRC" \
  --source-index "$SRC/index.json" \
  --output-root "$OUT"

python prepare/validate_relational_teacher_v9_all_sittable_dataset.py \
  --source-dataset-root "$SRC" \
  --dataset-root "$OUT" \
  --index "$OUT/index.json"
```

Expected final inventory:

- 2 train scenes and all 48 sealed train dense-contact sources recomputed;
- 12 balanced auxiliary replicas;
- one primary all-sittable GT per scene;
- 4 primary rows: 2 train scenes x 2 object-agnostic prompts;
- watch/write rows within a scene have a byte-identical GT hash;
- each train scene contains one Bed, one normal Chair and one High Chair;
- `room_0201` source metadata is hash-bound while its arrays remain unread;
- `hc_hd` train and `hcw_hdw` development High-Desk sources are disjoint;
- legacy non-High-Desk development evidence is disclosed as scene-held-out,
  not falsely claimed to be source-independent;
- Teacher-v9 LoRA training remains unauthorized;
- only the CUDA preflight becomes authorized.

## 3. Visually inspect the new GT before CUDA work

```bash
python prepare/visualize_relational_teacher_v9_all_sittable_viser.py \
  --source-dataset-root "$SRC" \
  --dataset-root "$OUT" \
  --index "$OUT/index.json" \
  --host 0.0.0.0 \
  --port 8080
```

Open `http://localhost:8080`.  If Ubuntu is remote, forward port 8080 first.

For `room_0101` and `room_0102`, inspect the same camera view and confirm:

1. Bed consensus is on the verified Bed.
2. Normal-Chair consensus is on the verified ordinary Chair.
3. High-Chair consensus is on the verified High-Desk Chair.
4. All-sittable consensus contains all three at the same time.
5. Other unverified Chairs are shown as unknown/ignore, not as negative GT.
6. TV, Desk and Whiteboard points are explicit negatives and have no positive
   object-surface consensus.

The continuous XY heatmap is display-only interpolation.  Numerical gates use
the original 8192 points, so a heatmap cannot make a failed tensor pass.

Stop the viewer with `Ctrl+C` and paste the three PASS logs plus one screenshot
per train scene.  Only after that evidence is accepted should the fresh v5r4
zero-init CUDA preflight/training patch be run.  Do not reuse the v8 specialist
checkpoint or its late-union schedule.

Do not open or visualize `room_0201` in this pre-lock gate.  The later locked
development gate will materialize it with the already sealed aggregation code
and hash, after the candidate checkpoint and inference policy are fixed.
