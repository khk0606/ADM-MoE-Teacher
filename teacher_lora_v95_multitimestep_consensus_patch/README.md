# Teacher-v9.5 multi-timestep consensus gate

This fail-closed CUDA preflight addresses the timestep-specific regressions
observed in the sealed Teacher-v9.4.1 metric replication.

It computes 24 normalized task gradients from four fixed design panels:
four timesteps x two prompts x three verified sittable objects. A deterministic
minimum-norm convex combination is used to form one common direction. Three
fixed trust-region radii are then evaluated on four different audit panels.

A candidate passes only when Bed, normal Chair and High Chair all improve in
pooled exact top-k overlap, no individual audit case regresses, and prompt,
background, explicit-negative and v5 replay retention gates pass. The gate
loads only `room_0101` arrays and writes no model checkpoint.

See `TEACHER_V95_MULTITIMESTEP_CONSENSUS_RUNBOOK.md` for Ubuntu commands.
