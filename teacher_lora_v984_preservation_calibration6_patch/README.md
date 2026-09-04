# Teacher-v9.8.4 fresh preservation-aware calibration-6

Teacher-v9.8.3 proved that a second t50 update can improve Bed, normal Chair,
and High Chair simultaneously while reducing all three fixed v5 losses and
explicit-negative means. This package extends only that accepted mechanism.

The run starts from sealed v5r4 plus a fresh zero-output LoRA. Update 1 must
exactly reproduce the selected v9.8.1 six-Sit direction, state, K=3 maps, and
v5 response. Update 2 must exactly reproduce the selected v9.8.3 eleven-task
direction, K=3 maps, and v5 response. Updates 3 through 6 recompute the same
eleven-task common-descent construction on the remaining sealed design states.

After every update, the actual frozen-Base-t499..51 plus LoRA-t50..0 schedule
is evaluated for K=3 and both prompts. A post-first state is admissible only if:

- the complete v9.8.1 response/retention policy still passes against Base;
- all three verified objects improve recall and MAE over the preceding state;
- each of Chair, Bed, and Whiteboard v5 losses is no worse than the preceding
  state;
- every generation's explicit-negative mean stays inside the incremental cap;
- the LoRA state changes along a valid common-descent direction.

Exact Top-k and absolute all-three presence remain disclosed diagnostics. They
are not silently turned into selection gates.

No optimizer is created and no model state is serialized. Only `room_0101`
arrays are loaded. A PASS authorizes only a two-train-scene preservation
calibration preflight, not `room_0102` access by this gate, `room_0201`, a
checkpoint, long training, or paper-test access.

Use the runbook. Its execution is one complete copy-paste block, with no shell
continuation backslashes and no fail-fast shell options.
