# Teacher-v9.6: saved one-step affordance-map viewer

This is a read-only visualization gate. It loads the already saved v9.6
`object_balanced_consensus_maps.npz`; it does not load a model, run diffusion,
train, or save a checkpoint.

The six aligned panels are:

1. Input room_0101 RGB point cloud;
2. accepted all-sittable GT (or the selected object's GT);
3. frozen v5r4 Base one-step prediction;
4. selected v9.6 candidate one-step prediction;
5. signed Candidate minus Base difference;
6. absolute Candidate-to-GT error.

The default candidate is the closest v9.6 near miss,
`chair_pair_priority_radius_0p001`. All 16 candidates, four unseen audit
timesteps, both prompts, all six channels, and each verified object can be
selected. When one object is selected, red/green point overlays identify exact
Top-k membership changes.

Important: these are fixed-timestep one-step `x_start` predictions, not final
500-step reverse-diffusion samples and not a trained Teacher checkpoint.

Run from `~/AMDM`. Every command below is deliberately a complete one-line
command so the shell cannot get stuck at a continuation `>` prompt.

```bash
cd ~/AMDM
conda activate afford
set +e
set +u
set +o pipefail
sha256sum -c teacher_lora_v96_affordance_viewer_v1.zip.sha256
unzip -q -o teacher_lora_v96_affordance_viewer_v1.zip
python teacher_lora_v96_affordance_viewer_patch/prepare/validate_teacher_lora_v96_affordance_viewer_package.py
cp teacher_lora_v96_affordance_viewer_patch/prepare/*.py prepare/
python prepare/test_relational_teacher_v96_affordance_viewer_common.py
```

Revalidate the saved artifact, then start the viewer:

```bash
REPORT=data/history_affordance_relational_teacher_v9_all_sittable_v1/experiments/teacher_lora_v96/object_balanced_consensus_s20261014_v1/preflight.json
python prepare/validate_relational_teacher_v96_object_balanced_consensus.py --report "$REPORT"
python prepare/visualize_relational_teacher_v96_object_balanced_viser.py --report "$REPORT" --host 0.0.0.0 --port 8080
```

Open `http://localhost:8080`. If Ubuntu/WSL is remote, forward port 8080.

Inspect in this order:

1. leave the default candidate selected;
2. select `all_verified` and compare GT, Base, and Candidate;
3. select `bed_01`, `chair_01`, and `chair_06` one by one;
4. for each object, inspect all four audit timesteps and watch/write;
5. red Top-k points left the Base set; green points entered the Candidate set;
6. the difference panel is signed and auto-scaled, so always read its printed
   numeric range before judging visual magnitude;
7. stop the viewer with `Ctrl+C`.

This viewer does not change the sealed v9.6 FAIL result and does not authorize
the next training or rollout gate.
