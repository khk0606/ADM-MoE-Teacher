# Upstream ADM entry points

Run from the repository root, not from this directory:

```bash
python scripts/adm/train.py [Hydra overrides]
python scripts/adm/test.py [Hydra overrides]
torchrun [distributed options] scripts/adm/train_ddp.py [Hydra overrides]
python scripts/adm/visualize_adm_affordance_viser.py --help
```

Only entry-point paths, root import bootstrapping and Hydra configuration paths changed. Dataset/output paths remain relative to the repository root. The existing task shell scripts have been updated.

For the current accepted Student maps, use `bash scripts/small_room30/student_anywhere.sh view` instead.
