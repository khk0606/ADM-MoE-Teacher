# Teacher-v10 fresh supervised all-sittable capacity run

This package replaces the v9.8.x late rollout-state micro-calibration with
standard START_X diffusion supervision.  Every AdamW update uses the sealed
all-sittable x0 label, q_sample noise over the full timestep curriculum and
equal contribution from both train scenes.  A fresh rank-16 LoRA starts from
sealed v5r4.  The three best train-only fixed-panel states receive actual
two-scene, two-prompt, K=3, 500-step generation.  A checkpoint is written only
when one state passes the actual all-three and v5-retention gates.

No room_0201 array or paper-test payload is read.

`run_training.sh` validates the package, copies the sealed implementation into
`prepare/`, runs the CPU contract, performs the CUDA training, deeply validates
the saved maps and prints a concise diagnosis.  The shell exits on a failed
gate without closing the user's terminal.  It never overwrites an existing
versioned output directory.

After training, `run_viewer.sh` validates the same result and opens a read-only
Viser comparison of GT, frozen v5r4 Base, each shortlisted actual-rollout
candidate, candidate-minus-Base and absolute candidate-to-GT error.  Continuous
XY heatmaps are display-only; acceptance always uses the original 8192 values.
