# Teacher-v10.2 dense per-instance support training

This delivery starts from sealed v5r4 and treats the failed Teacher-v10.1
summary as immutable redesign authority. It reuses the audited two-scene
1000-update/K=3 pipeline but replaces the objective.

Each Bed, normal Chair and High Chair instance target contributes its complete
dense positive support with equal macro weight. The union of all three dense
supports is excluded from hard-background mining. Unknown sittable objects
remain ignored, explicit negatives remain supervised, and v5 replay starts at
update one.

The run evaluates three shortlisted in-memory states with actual 500-step K=3
generation on both train scenes and both prompts. A merged LoRA checkpoint is
written only if one state passes the locked actual-rollout gate. room_0201 and
paper-test payloads remain unread.

Run `bash teacher_lora_v102_dense_instance_patch/run_training.sh` from
`~/AMDM`. Afterward, run `bash teacher_lora_v102_dense_instance_patch/run_viewer.sh`
to inspect the saved maps.
