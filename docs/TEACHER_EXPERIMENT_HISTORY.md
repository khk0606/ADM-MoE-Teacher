# Teacher-LoRA experiment history

This document indexes the versioned Teacher source included in this repository.
Each directory under `research_history/teacher_lora_packages/` is an immutable
source delivery containing its run entry point, contract, validator, tests, and
package manifest. The flat `prepare/` directory contains the combined dependency
chain needed by the latest included experiments.

## Milestones

| Version | Purpose | Recorded outcome |
| --- | --- | --- |
| v9 | Build the two-scene all-sittable labels, roles, metrics, and LoRA preflight | Dataset/preflight contracts passed |
| v9.1 | Add active-support and ranking responses; test short one-scene overfit | Loss response passed; rollout smoke gate failed |
| v9.2-v9.6 | Diagnose hotspot trust, exact top-k, common descent, timestep consensus, and object-balanced directions | Diagnostic candidates failed their locked rollout gates |
| v9.7 | Evaluate in-memory steps 3/6/12 with actual 500-step K=3 generation | No early step was admissible; all-three was 0/3 |
| v9.8-v9.8.12 | Move supervision to rollout states, diagnose cross-scene directions, test radius 0.006, replicate on disjoint seeds, and run multi-update calibration | Radius response and replication passed their local gates; multi-update absolute all-three remained 0/3 |
| v10 | Replace response-only updates with direct supervised GT training | High label fit, but absolute K=3 and retention gates failed |
| v10.1 | Apply full-field supervision and explicit hard-background policy | High label fit, but absolute K=3 remained 0/3 |
| v10.2 | Apply equal-macro dense per-instance support supervision | Retained as visual candidate; strict actual-K3 gate failed and no checkpoint was written |
| v10.3 | Measure on-policy response directions from resumed trajectories | Local response-direction gate passed; absolute all-three remained diagnostic and false |
| v10.3.1 | Run six short on-policy calibration updates with K=3 monitoring | Relative calibration shortlisted steps 1 and 2; no final checkpoint was authorized |

## Included source packages

The repository includes every Teacher package from the v9-v10.3.1 chain,
including validator fixes and visualization packages. The final visual candidate
uses these primary packages:

They are grouped under one archive directory so the repository root exposes the
runnable code instead of 34 delivery folders:

```text
research_history/
└── teacher_lora_packages/
    ├── teacher_lora_v9_all_sittable_patch/
    ├── ...
    ├── teacher_lora_v102_dense_instance_patch/
    ├── teacher_lora_v103_onpolicy_response_patch/
    ├── teacher_lora_v1031_onpolicy_calibration6_patch/
    └── teacher_lora_v1031_affordance_viewer_patch/
```

The complete package list is documented in
`research_history/teacher_lora_packages/README.md`.

## Reproduction order

Each runner is fail-closed and binds to the preceding report. Reproduce the
packages in chronological order rather than starting v10.2 against an unrelated
summary. Every package README and shell entry point states its exact input report
and output directory.

For the retained v10.2 result, the final three supervised stages are:

```bash
bash research_history/teacher_lora_packages/teacher_lora_v10_supervised_capacity_patch/run_training.sh
bash research_history/teacher_lora_packages/teacher_lora_v101_fullfield_supervision_patch/run_training.sh
bash research_history/teacher_lora_packages/teacher_lora_v102_dense_instance_patch/run_training.sh
```

The first command also requires the sealed v9.8.12 summary. To reproduce from
the beginning, follow the versioned package READMEs and runbooks in order.

For professor-facing reproduction from a fresh clone, use the portable wrapper
instead of replaying every historical stage:

```bash
bash scripts/teacher_v102/bootstrap.sh
bash scripts/teacher_v102/reproduce.sh
bash scripts/teacher_v102/view.sh
```

The bootstrap only installs and verifies the data/checkpoint Release. Training,
actual K=3 evaluation, and map generation begin only when `reproduce.sh` is run.

## Interpretation rule

A package-level `PASS` means only that package's stated contract passed. It does
not imply that a final Teacher checkpoint exists. Checkpoint authorization is a
separate gate. Teacher-v10.2 and Teacher-v10.3.1 did not authorize checkpoint
export.
