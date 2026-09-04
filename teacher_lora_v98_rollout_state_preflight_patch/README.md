# Teacher-v9.8 rollout-state response preflight

Teacher-v9.7 proved that one-step improvement does not transfer to the actual
500-step reverse-diffusion output: Bed, normal Chair and High Chair were never
simultaneously present in any of the six prompt/generation panels.

This package therefore measures gradients on states that the frozen v5r4
Teacher actually visits during reverse diffusion. It captures `x_t` and the
exact RNG state at timesteps 400, 200 and 50, constructs a six-task common
descent direction for Bed/normal-Chair/High-Chair across watch/write, and tests
three small trust-region radii at each timestep. Every candidate is judged only
after resuming the original audit trajectory to its final 500-step map.

The new design trajectory is source-disjoint from the audit trajectory. The
audit Base is required to reproduce v9.7 generation 0 bitwise, and every stored
`x_t + RNG` resume must reproduce that same Base final map bitwise. Exact Top-k
is diagnostic only; continuous object recall, active-support MAE, hotspot
location, prompt invariance, explicit negatives and v5 retention form the gate.

This is a response preflight, not training. It creates no optimizer and writes
no model checkpoint. Only `room_0101` arrays may be loaded. `room_0102`,
`room_0201` and paper-test arrays remain unread.
