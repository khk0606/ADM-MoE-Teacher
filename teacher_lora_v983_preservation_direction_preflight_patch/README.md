# Teacher-v9.8.3 preservation-aware rollout-state direction

Delivery v2 replaces the positional-format candidate log with a checked
f-string. This corrects the v1 `IndexError` after the first candidate without
changing any model, seed, data, metric, direction, radius, or authorization.

Teacher-v9.8.2 established a clean stopping point: update 1 is admissible, but
updates 2--6 progressively damage the fixed v5 replay probe and later increase
explicit-negative means. Prompt invariance improves throughout, so it is not
the cause of that failure.

This package does not relax those gates and does not resume the failed state.
It starts from fresh sealed v5r4, exactly reconstructs the accepted v9.8.1
update-1 state, and constructs a hypothetical second direction from eleven
normalized tasks at a new frozen-Base t50 design state:

- six prompt-by-object Sit losses (Bed, normal Chair, High Chair);
- three fixed v5 replay losses (Chair, Bed, Whiteboard);
- two prompt-specific explicit-negative means (TV, Desk, Whiteboard).

Five small radii are evaluated with the real K=3, two-prompt t50-to-t0
reverse-diffusion continuation. A candidate is admissible only if all three Sit
objects improve beyond update 1, all three v5 cases are no worse than update 1,
each generation's explicit-negative mean stays within the locked increment,
and every original v9.8.1 response/retention gate remains satisfied.

This is a no-optimizer, no-checkpoint preflight. It loads only `room_0101` scene
arrays. A PASS authorizes only a fresh preservation-aware calibration-6 gate;
it does not authorize `room_0102`, `room_0201`, long training, a checkpoint, or
paper-test access.

Use `TEACHER_V983_PRESERVATION_DIRECTION_PREFLIGHT_RUNBOOK.md`. Its execution
section is deliberately one copy-paste block and does not enable shell
fail-fast options, so a legitimate model FAIL cannot close an interactive
terminal.
