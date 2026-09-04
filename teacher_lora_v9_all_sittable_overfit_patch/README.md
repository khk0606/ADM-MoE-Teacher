# Teacher-v9 one-scene overfit smoke

This package is the first optimizer-bearing learnability test for the accepted
Teacher-v9 all-sittable target.  Follow
`TEACHER_V9_ALL_SITTABLE_ONE_SCENE_OVERFIT_RUNBOOK.md` exactly.

The run is intentionally disposable: it trains only `room_0101`, validates
actual reverse-diffusion output, writes diagnostic maps and JSON, and forbids
checkpoint export.  Any later response/calibration run must restart from the
sealed v5r4 Teacher.
