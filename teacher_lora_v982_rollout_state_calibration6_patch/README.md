# Teacher-v9.8.2 rollout-state calibration-6

This package consumes only the sealed Teacher-v9.8.1 PASS and starts again
from the sealed v5r4 Teacher with a fresh zero-output rank-4 LoRA.

It applies six exact radius-0.003 common-descent updates at diffusion timestep
50. Update 1 exactly reproduces the selected v9.8 response. Updates 2 through
6 use new deterministic Base trajectory states that are disjoint from the K=3
audit seeds. After every update, all three sealed generations and both
object-agnostic Sit prompts are resumed from t50 to t0 and evaluated.

No optimizer state or model checkpoint is written. Room `room_0102`,
development room `room_0201`, long training, and paper-test access remain
unauthorized.
