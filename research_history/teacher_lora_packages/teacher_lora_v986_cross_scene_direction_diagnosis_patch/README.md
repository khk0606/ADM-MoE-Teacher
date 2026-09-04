# Teacher-v9.8.6 cross-scene direction diagnosis

This fail-closed gate binds the exact Teacher-v9.8.5 failure: reconstructed
step 4 improved room_0102 normal Chair and High Chair but slightly worsened
room_0102 Bed recall and active-support MAE.

The gate reconstructs the exact step-4 LoRA state from fresh sealed v5r4,
captures independent train-only t=50 design states for room_0101 and
room_0102, then computes 19 independently normalized gradients:

- six Sit-object tasks and two explicit-negative tasks from room_0101;
- six Sit-object tasks and two explicit-negative tasks from room_0102;
- Chair, Bed, and Whiteboard v5 preservation tasks.

It diagnoses a minimum-norm common direction plus three disclosed Bed-guard
blends and a Bed-only control. No diagnosed direction is applied. The Bed-only
control must fail whenever it does not preserve all other tasks. Ranking only
considers directions whose derivatives are strictly positive for all 19 tasks.

Version 2 accumulates the Gram matrix in float64, audits the raw numerical
asymmetry against a fixed `1e-8` cap, and only then canonicalizes the matrix as
`(G + G.T) / 2`. This prevents harmless float32 matrix-multiplication rounding
from being misclassified as invalid gradient geometry.

See `TEACHER_V986_CROSS_SCENE_DIRECTION_RUNBOOK.md` for the single copy-paste
Ubuntu block.
