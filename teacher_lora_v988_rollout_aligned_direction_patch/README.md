# Teacher-v9.8.8 rollout-aligned cross-scene direction diagnosis

This package binds the sealed Teacher-v9.8.7 failure and recomputes the
direction geometry on the exact K=3, two-prompt, two-scene `t=50` states used
by the failed actual rollouts.

It constructs 51 independently normalized LoRA gradients:

- 36 verified-object tasks: 2 scenes x 3 generations x 2 prompts x 3 roles;
- 12 explicit-negative tasks: 2 scenes x 3 generations x 2 prompts;
- 3 fixed v5 preservation tasks.

The run is diagnostic only. It reconstructs the exact Teacher-v9.8.4 step-4
state in memory, but applies no candidate direction, creates no optimizer and
saves no model checkpoint. `room_0201` and paper-test payloads remain unread.

A PASS authorizes only an actual two-scene K=3 radius response for the selected
rollout-aligned direction. It does not authorize long training or development
evaluation.
