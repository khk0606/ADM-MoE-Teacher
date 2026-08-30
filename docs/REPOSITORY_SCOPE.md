# Repository scope and exclusion policy

This repository is a clean source snapshot, not a backup of the training machine.

## Included

- ADM/CMDM implementation and configuration.
- Base Teacher and LoRA source required by the current Teacher pipeline.
- MoE-IIW source, routing code, evaluators, and contract tests.
- Teacher-v7 source for dataset staging, dense-contact generation, CUDA preflight, invariance recovery, the bounded pilot, K=1/K=3 canaries, and full train-only K=3 evaluation.
- One Unity C# physical-audit source file.
- Current runbooks and the upstream MIT license.

## Excluded

- `_data_metadata/` and every dataset array or point-cloud payload.
- `experiments/`, summaries, reports, generated maps, metrics, videos, and logs.
- all `.pt`, `.pth`, `.ckpt`, `.npy`, `.npz`, pickle, and archive files.
- raw/combined motion TXT files and Unity-exported scenes.
- delivery ZIP files, SHA-256 sidecars, partial transfers, and stale patch bundles.
- the superseded Teacher-v6 training pipeline and failed Teacher-v7 calibration entry point.
- experimental History-Affordance objective-v2/v2.1/v2.2/v2.4 patch trees that are not part of the current Teacher-v7 gate.

Two Teacher-v6 modules remain under `prepare/`: `relational_teacher_v6_contract.py` and `relational_teacher_v6_semantics.py`. They are retained because the current Teacher-v7 code imports their stable prompt, hashing, atomic-write, and semantic-loss helpers. This is dependency reuse, not inclusion of the superseded v6 training pipeline.

## Current claims

The source corresponds to a passed train-only Teacher-v7 step-12 K=3 canary. It does not claim a completed full K=3 run, development result, held-out result, or paper-test result. The relation-distance MoE extension is a research direction and must not be described as already validated until its own sealed evaluation passes.
