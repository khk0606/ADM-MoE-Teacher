# Teacher-v9.8.1 selected rollout-state response-6

Teacher-v9.8 selected `t50_radius_0p003` using a source-disjoint design
trajectory and one sealed audit generation. This package tests whether that
same parameter direction transfers across all three already sealed v9.7
generation seeds for both object-agnostic Sit prompts.

Generation 0 Base/candidate maps are reused byte-for-byte from v9.7/v9.8.
Generations 1 and 2 regenerate the frozen Base trajectory, prove its final map
equals the sealed v9.7 Base, capture `x_50 + RNG`, and resume the paired final
50 steps with the selected LoRA response. The adapter is deliberately inactive
from t499 through t51 and active only from t50 through t0; this is a response
test, not a final inference policy.

PASS requires High Chair recall and MAE to improve in at least two generations
and in the pooled panel, while the pooled worst-object recall and macro MAE
also improve. Every one of the six maps must retain each object within fixed
tolerances, and prompt invariance, explicit negatives and v5 replay remain
bounded. Absolute all-three presence is reported honestly but is not confused
with this directional-response gate.

No optimizer or checkpoint is created. Only `room_0101` arrays are loaded.
`room_0102`, `room_0201` and paper-test arrays remain unread.
