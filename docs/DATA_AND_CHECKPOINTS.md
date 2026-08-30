# External data and checkpoints

Large or restricted assets are intentionally not stored in Git.

## Expected local paths

The current runbooks expect the following categories of files under the repository root:

```text
data/
  history_affordance_relational_teacher_v7_hd/
  history_affordance_v1/
  Mean_Std_Cont_HumanML3D_HUMANISE_PROX_contact_cont_joints_0.8_fur.npz
outputs/
  CDM-Perceiver-ALL/ckpt/model300000.pt
body_models/
  ...
```

The exact relational dataset contains scene point clouds, semantic metadata, dense contact targets, motion bindings, and train/development split information. Raw motion and Unity scene packages must remain outside Git unless their licenses explicitly permit redistribution.

## Integrity

For reproducible experiments, keep a private manifest containing:

- the SHA-256 hash of every checkpoint;
- the SHA-256 hash of every source scene and motion;
- dataset index and split hashes;
- random seed, diffusion steps, and software versions;
- the Git commit used for the run.

Do not commit the private manifest when it contains machine paths, participant information, or restricted asset identifiers.

## Git LFS

This source snapshot does not require Git LFS. If public redistribution of a future checkpoint is authorized, publish it as a versioned release asset or in a model repository with a clear license, rather than committing it to normal Git history.
