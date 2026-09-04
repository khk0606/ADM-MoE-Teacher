# Teacher-v9 all-sittable LoRA CUDA preflight v3

This delivery is the first model-side gate after the accepted Teacher-v9
scene-level GT build and visual audit.

It does not perform an optimizer update and does not save a candidate model.
It proves that a fresh zero-init LoRA on the sealed v5r4 Teacher can receive
separate gradients from Bed, normal Chair and High Chair while preserving the
role policy that unverified Sit objects are unknown/ignored.

Run `TEACHER_V9_ALL_SITTABLE_LORA_PREFLIGHT_RUNBOOK.md` on Ubuntu from
`~/AMDM`.  A validated PASS authorizes only the next one-train-scene overfit
smoke.  It does not authorize response-3, calibration, pilot, development or
paper-test access.

v2 corrects the GT reader contract for `winner_instance_slot`: the sealed
builder stores one winning instance slot per point and contact channel, so its
exact shape is `[8192,6]`.  This is a reader-only correction; GT arrays and
objective semantics are unchanged.

v3 keeps finite-number enforcement for the frozen baseline, but correctly
accepts `+inf` hotspot-centroid distance as the expected fail-closed signal in
Bed-only or missing-Chair counterexamples.  Only the resulting boolean failure
checks are serialized; non-finite numbers never enter the locked policy JSON.
