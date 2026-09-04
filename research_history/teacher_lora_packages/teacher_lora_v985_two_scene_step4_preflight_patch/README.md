# Teacher-v9.8.5 two-scene step-4 preflight

Teacher-v9.8.4 established step 4 as the last strictly admissible room_0101
state; steps 5 and 6 regressed Bed recall and MAE. This package does not train
further. It reconstructs the step-4 LoRA state from fresh sealed v5r4 and
tests that exact state on the second train scene, room_0102.

The audit uses three paired reverse-diffusion generations for both object-
agnostic Sit prompts. Bed, normal Chair (`chair_05`) and High Chair
(`chair_06`) are checked independently. Exact Top-k and absolute all-three
presence are reported as diagnostics, while continuous recall, active-support
MAE, centroid, prompt invariance, explicit negatives and v5 retention control
the gate.

No optimizer is created and no model state is saved. room_0201 and the paper
test remain unread. Use the single-block runbook.
