# Teacher-v9.8.10 rollout-aligned recovery bracket

Teacher-v9.8.9 proved that every tested radius passed `room_0101`, while
`room_0102` failed only the Bed recall and Bed MAE strict-improvement checks.
Both Bed metrics recovered monotonically at every increasing radius, but the
largest tested radius stopped before crossing the frozen-Base baseline.

This package binds that exact failure and tests a bounded continuation bracket
`0.004, 0.005, 0.006, 0.007, 0.008`. Each radius starts from the same exact
Teacher-v9.8.4 step-4 state and reuses the exact v9.8.8
`audit_bed_guard_0p10` direction. It runs actual `t=50 -> 0` generation on two
train scenes, K=3 and both prompts. The exact v9.8.5 gates are applied to each
scene independently, and the smallest fully admissible radius is selected.

No optimizer or model checkpoint is created. `room_0201` and paper-test
payloads remain unread. PASS authorizes only fresh two-scene rollout-aligned
calibration. FAIL requires redesigning the cross-scene training objective.
