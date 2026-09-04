# Teacher-v9.8.7 actual two-scene radius response

Teacher-v9.8.6 proved that `bed_guard_0p10` is a local common-descent direction
for all 19 source/audit/preservation tasks. This package tests whether that
local result produces real affordance-map improvements after reverse diffusion.

Every radius (`0.0005`, `0.001`, `0.002`, `0.003`) starts from the exact same
freshly reconstructed Teacher-v9.8.4 step-4 LoRA state. The selected v9.8.6
direction is independently recomputed and hash-matched before any candidate
update. Each candidate then runs actual t=50-to-0 generation for room_0101 and
room_0102, K=3, with both watch/write prompts.

The exact Teacher-v9.8.5 scene policy is applied independently to both scenes.
A radius is eligible only if all three roles improve in both scenes while
per-map retention, prompt invariance, explicit negatives, and v5 retention all
pass. Top-k and absolute all-three presence remain diagnostics. No checkpoint
is written.
