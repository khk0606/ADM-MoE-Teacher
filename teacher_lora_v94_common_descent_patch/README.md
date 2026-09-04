# Teacher-v9.4 common-descent preflight

This package binds the genuine Teacher-v9.3 exact-top-k failure and diagnoses
the shared-LoRA gradient geometry of the six fixed train tasks:

- watch/write x Bed;
- watch/write x normal Chair;
- watch/write x High Chair.

It computes a deterministic minimum-norm convex combination of the six
normalized gradients. Four small Euclidean trust-region steps are evaluated.
A candidate is admissible only when every one of the six same-panel swap
losses decreases, no top-k overlap regresses, and all retention gates pass.

This package writes prediction arrays and a JSON report only. It never writes
a model checkpoint and never reads room_0102, room_0201, or paper-test arrays.
A pass authorizes only a separate six-update common-descent response gate.

