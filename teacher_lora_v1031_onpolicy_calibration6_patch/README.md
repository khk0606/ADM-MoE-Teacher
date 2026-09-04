# Teacher-v10.3.1 on-policy K=3 calibration

This fail-closed calibration consumes only the sealed Teacher-v10.3 PASS and
reconstructs its selected t150/radius-0.004 LoRA state from fresh v5r4. Fresh,
disjoint K=3 design and audit trajectories are created for room_0101 and
room_0102. Six radius-0.001 common-descent updates are recomputed from the
locked 21-task inventory, and each update is measured on actual resumed final
maps for both prompts and all three generations.

Absolute all-three presence remains diagnostic during this short calibration.
Every admissible state must improve the pooled continuous response while
retaining every individual generation, explicit negatives and the v5 fixed
probe. Up to two states are shortlisted and verified in memory. No optimizer
or checkpoint writer exists in this package. A PASS authorizes only extended
on-policy training with actual K=3 monitoring; checkpoint export still requires
a later full-schedule all-three gate. room_0201 and paper-test payloads remain
unread.
