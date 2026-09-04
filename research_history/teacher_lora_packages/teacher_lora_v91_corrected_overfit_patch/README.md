# Teacher-v9.1 corrected one-scene overfit smoke

This package consumes the validated Teacher-v9.1 loss-response PASS and starts
again from sealed v5r4 plus a fresh zero-output LoRA.  It trains only
`room_0101` with the selected active-support/ranking objective.

The failed Teacher-v9 120-update state and the short loss-response states are
never loaded.  No model checkpoint is written.  The final gate is an actual
paired 500-step reverse-diffusion rollout requiring Bed, normal Chair and High
Chair simultaneously for both object-agnostic Sit prompts.

Follow `TEACHER_V91_CORRECTED_ONE_SCENE_OVERFIT_RUNBOOK.md` exactly.
