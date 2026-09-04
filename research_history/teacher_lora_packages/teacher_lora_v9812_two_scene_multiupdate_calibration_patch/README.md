# Teacher-v9.8.12 two-scene multi-update calibration

This package starts from the exact Teacher-v9.8.11 radius-0.006 state. It makes
six short, rollout-state-aligned updates using both train scenes. Every update
is audited with disjoint-seed, actual K=3 generation for both prompts and both
scenes.

The one-state shortlist is fail-closed. A state is eligible only if both strict scene
policies pass, all three verified Sit objects appear together in at least two
of three generations for every scene/prompt, each role is preserved relative
to the starting state, all 51 direction tasks are common descent, and v5,
negative and prompt-invariance retention gates pass.

No optimizer or checkpoint is created. PASS authorizes a separate exact-state
reconstruction and checkpoint-export gate; FAIL authorizes objective or model
capacity redesign only.

Use `TEACHER_V9812_TWO_SCENE_MULTIUPDATE_CALIBRATION_RUNBOOK.md`. Its single
shell block does not use continuation backslashes and does not close the
interactive terminal on failure.
