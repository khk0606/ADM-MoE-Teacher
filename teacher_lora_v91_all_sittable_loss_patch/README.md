# Teacher-v9.1 active-support/ranking loss response

This package diagnoses and corrects the failed Teacher-v9 one-scene smoke
without weakening its accepted metrics.  The failed 120-update state is used
only as immutable evidence; it is never loaded as a checkpoint.

The old whole-object MSE was dominated by zero/background channels.  It could
lower average error while leaving the real Chair and High-Chair contact
hotspots below other points.  The corrected objective adds:

- equal-macro loss on GT-active channels for Bed, normal Chair and High Chair;
- within-object ranking of GT hotspot points above hard non-hotspot points;
- explicit-negative, prompt-invariance and sealed-v5 preservation terms.

The CUDA response grid runs three fresh, identical three-update trials from
the sealed v5r4 Teacher and zero-init LoRA.  It saves diagnostics only, not a
model checkpoint.  A validated pass authorizes only a corrected one-scene
overfit smoke.

Follow `TEACHER_V91_ALL_SITTABLE_LOSS_RESPONSE_RUNBOOK.md` exactly.
