# Teacher-v10.3 two-scene on-policy response gate

This diagnostic gate binds the failed Teacher-v10.2 result and starts from the
sealed v5r4 model with a fresh zero-output rank-16 LoRA. It does not continue a
failed candidate.

Fresh design and audit reverse-diffusion trajectories are captured for both
train scenes and both object-agnostic prompts at timesteps 350, 150 and 50.
Twenty-one normalized tasks cover twelve scene/prompt/object dense-support
losses, four negative losses, two prompt-invariance losses and three v5 probe
losses. Minimum-norm common-descent directions are tested at four trust-region
radii. Candidate decisions use only disjoint audit trajectories resumed to the
actual final 500-step map.

This is a response preflight, not long training. Absolute all-three presence is
recorded as a diagnostic; the gate requires simultaneous progress without
object regression, negative leakage or more than one-percent v5 drift. No
optimizer or checkpoint writer exists. A PASS authorizes only a fresh
two-scene on-policy multi-update calibration. room_0201 and paper-test payloads
remain unread.
