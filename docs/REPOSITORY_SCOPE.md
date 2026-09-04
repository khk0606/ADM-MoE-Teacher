# Repository scope and exclusion policy

This repository is a clean source snapshot, not a backup of the training machine.

## Included

- ADM/CMDM implementation and configuration.
- Base Teacher and LoRA source required by the current Teacher pipeline.
- MoE-IIW source, routing code, evaluators, and contract tests.
- Teacher-v7 source for dataset staging, dense-contact generation, CUDA preflight, invariance recovery, the bounded pilot, K=1/K=3 canaries, and full train-only K=3 evaluation.
- Versioned Teacher-v9 through Teacher-v10.3.1 source deliveries, including
  contracts, objectives, run entry points, validators, tests, summarizers, and
  Viser viewers.
- A path-free public Teacher-v10.2 metric summary and visualization screenshot.
- One Unity C# physical-audit source file.
- Current runbooks and the upstream MIT license.

## Excluded

- `_data_metadata/` and machine-local dataset payloads.
- machine-local `experiments/`, summaries, reports, generated maps, metrics,
  videos, and logs. Small path-free results under `docs/results/` are allowed.
- all `.pt`, `.pth`, `.ckpt`, `.npy`, `.npz`, pickle, and archive files.
- raw/combined motion TXT files and Unity-exported scenes.
- delivery ZIP files, SHA-256 sidecars, and partial transfers. Versioned source
  directories are included, but their transport archives are not.
- the superseded Teacher-v6 training pipeline and failed Teacher-v7 calibration entry point.
- experimental History-Affordance objective-v2/v2.1/v2.2/v2.4 patch trees that are not part of the current Teacher-v7 gate.

Two Teacher-v6 modules remain under `prepare/`: `relational_teacher_v6_contract.py` and `relational_teacher_v6_semantics.py`. They are retained because the current Teacher-v7 code imports their stable prompt, hashing, atomic-write, and semantic-loss helpers. This is dependency reuse, not inclusion of the superseded v6 training pipeline.

## Current claims

Teacher-v10.2 is published as an experimental visual candidate, not as a
validated Teacher checkpoint. It failed the strict actual-K3 selection gate,
`selected_step` remained `None`, and checkpoint export was not authorized.
Teacher-v10.3 and v10.3.1 are included as later diagnostic/calibration history;
they do not change the v10.2 result or establish a final Teacher release.
