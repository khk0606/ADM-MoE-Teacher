# Teacher-v10.1 full-field all-sittable supervision

Teacher-v10 learned the verified objects but reduced its positive-only losses by
spreading affordance over broad scene regions.  This package starts again from
sealed v5r4 and a fresh zero-output rank-16 LoRA.  Every non-unknown point is
supervised against the complete all-sittable GT tensor.  Only unverified
sittable object points remain ignored.  Explicit hard-background, worst-object
and step-one v5 preservation losses prevent broad false-positive shortcuts and
catastrophic forgetting.

Both train scenes contribute to every AdamW update over the full timestep
curriculum.  Three v5-aware monitor states receive actual two-scene,
two-prompt, K=3, 500-step generation.  A checkpoint is serialized only after an
actual rollout candidate passes.  Failure maps remain available to the
read-only Viser launcher.  room_0201 and paper-test payloads stay unread.
