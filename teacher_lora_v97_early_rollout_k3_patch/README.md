# Teacher-v9.7 early actual-rollout K=3

This package diagnoses the v9.1 exposure gap at the correct decision unit.
It exactly reproduces the first 12 v9.1 updates from fresh sealed v5r4, retains
steps 3, 6 and 12 only in CPU memory, and evaluates each state with paired
500-step reverse diffusion for both object-agnostic Sit prompts at K=3.

The primary decision is simultaneous Bed, normal-Chair and High-Chair presence
using continuous recall, support MAE and hotspot location. Exact Top-k remains
reported but is not a gate because a one-point membership swap causes a large
discrete jump on the small Chair masks. Explicit-negative leakage and v5 replay
retention remain separate guards.

No model checkpoint is written. Only `room_0101` arrays may be loaded. A PASS
authorizes a locked selected-step confirmation; a legitimate FAIL authorizes
only a rollout-state/on-policy preflight.
