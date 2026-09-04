# External data and checkpoints

Large or restricted assets are intentionally not stored in Git.

## Expected local paths

The current runbooks expect the following categories of files under the repository root:

```text
data/
  history_affordance_relational_teacher_v7_hd/
  history_affordance_relational_teacher_v9_all_sittable_v1/
  history_affordance_v1/
  Mean_Std_Cont_HumanML3D_HUMANISE_PROX_contact_cont_joints_0.8_fur.npz
outputs/
  CDM-Perceiver-ALL/ckpt/model300000.pt
body_models/
  ...
```

The exact relational dataset contains scene point clouds, semantic metadata,
dense contact targets, motion bindings, and train/development split information.
The `room_0101`/`room_0102` Unity point clouds and GT motions are author-created
assets and may be distributed in a separately versioned result/data bundle.
Upstream ADM data, pretrained weights, body models, and third-party assets retain
their original licenses and are not redistributed by this repository.

## Teacher-v10.2 artifacts

The v10.2 experiment writes `summary.json` and
`dense_instance_supervision_maps.npz` under its machine-local experiment
directory. The source repository includes only a path-free public metric summary
and a rendered screenshot. The exact numeric map bundle may be published as a
GitHub Release asset if its point-cloud and GT payloads are cleared for release.

Create that portable bundle with
`bash scripts/teacher_v102/export_public_result.sh`. The exporter verifies the
original report/map hash, strips machine paths, copies the exact NPZ, and writes
a portable summary. The companion public Viser verifies the copied hash before
rendering it.

Teacher-v10.2 did not pass checkpoint authorization, so there is no legitimate
Teacher-v10.2 checkpoint to upload. Do not rename an intermediate in-memory state
as a validated checkpoint.

## Integrity

For reproducible experiments, keep a private manifest containing:

- the SHA-256 hash of every checkpoint;
- the SHA-256 hash of every source scene and motion;
- dataset index and split hashes;
- random seed, diffusion steps, and software versions;
- the Git commit used for the run.

Do not commit the private manifest when it contains machine paths, participant information, or restricted asset identifiers.

## Git LFS

This source snapshot does not require Git LFS. If the optional numeric v10.2 map
bundle is published, prefer a versioned GitHub Release asset. If public
redistribution of a future checkpoint is authorized, publish it as a versioned
release asset or in a model repository with a clear license, rather than
committing it to normal Git history.
