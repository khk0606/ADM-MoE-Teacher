# Teacher-v9.8.9 rollout-aligned two-scene radius response

This package binds the sealed Teacher-v9.8.8 PASS and reproduces its selected
`audit_bed_guard_0p10` direction from the exact 51 rollout-state gradients.

Five radii share one exact reconstructed Teacher-v9.8.4 step-4 start. Each
radius is evaluated with actual `t=50 -> 0` generation on both `room_0101` and
`room_0102`, three generations and both Sit prompts. The exact strict v9.8.5
three-role, negative, invariance and v5 gates are applied independently to
both scenes. A radius is eligible only if both scenes pass every check.

The run creates no optimizer and saves no model checkpoint. `room_0201` and
paper-test payloads remain unread. PASS authorizes only a fresh two-scene
rollout-aligned calibration; FAIL authorizes only objective/inference redesign.
