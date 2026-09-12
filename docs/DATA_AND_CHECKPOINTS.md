# SmallRoom30 data and checkpoints

## Publication status

The repository contains source, architecture, and qualitative screenshots. The project owner reports completed training and Viser review. Final server run payloads have not yet been imported or published. No fresh-clone download/bootstrap is claimed.

## Required saved assets

These are script defaults, not evidence that each directory has been uploaded:

| Path | Role |
| --- | --- |
| `data/small_room30_adm_lora_v1/` | Exact 30-room dataset, index and manifest |
| `outputs/small_room30_student_anywhere01/` | Final Anywhere + Purpose Student |
| `outputs/small_room30_targeted_v5_eval_full01/` | Frozen Teacher 1578 maps and evaluation |
| `outputs/small_room30_student_competition_cpu_eval01/` | Initialization for Anywhere fine-tuning |
| `outputs/small_room30_student_pilot01/` | Baseline bindings and text features |
| Local CLIP ViT-B/32 weights | Verified encoding for the added prompt |

Keep the final Student `summary.json`, `manifest.json`, `comparison.json`, `student_model.json`, `student_weights.npz`, `text_features.npz`, `maps/` and actual logs together. Saved-map viewing requires the matching dataset and code bindings; a checkpoint alone is insufficient.

Teacher retraining additionally needs the LoRA checkpoints, base ADM weights and previous-run dependencies named in its manifests. These paths must be resolved from the actual server run.

## Missing Teacher packaging files

Some legacy Teacher shell scripts reference source seals and audit reports that are absent from the local source snapshot, including targeted_v4/v5 and v5_full SHA256SUMS/dependency lists. They must be recovered and verified from the original server/package before claiming the Teacher training scripts are turnkey. Do not disable validation or regenerate seals merely to bypass mismatches.

## Integrity and privacy

- Preserve raw run JSON and hashes byte-for-byte.
- Record human acceptance in `docs/results/small_room30_anywhere/REVIEW_STATUS.md`, not by changing `approved=false` in viewer inputs.
- Keep large weights, numeric maps and data out of normal Git; publish separately only after checking redistribution rights.
- Remove private paths/identifiers only in a separate public summary, never in sealed raw input files.
- Preserve upstream model, body-model and Unity asset licenses.
- The old Teacher-v10.2 asset release is not a substitute for these SmallRoom30 dependencies.
