# Teacher-v9.8.11 disjoint-seed replication

This package reconstructs the sealed Teacher-v9.8.10 radius-0.006 LoRA state
in memory and evaluates it on a new seed table. It runs actual two-scene K=3
rollouts for both object-agnostic Sit prompts and applies the locked strict
three-role response policy independently to each scene.

It creates no optimizer and serializes no model state. PASS authorizes only a
fresh, short, two-scene rollout-aligned multi-update calibration. FAIL
authorizes only objective redesign. `room_0201` and the paper test remain
unread in both cases.
