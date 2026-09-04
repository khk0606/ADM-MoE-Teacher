# Teacher-v9 all-sittable data gate

This patch replaces the conflicting Teacher-v7/v8 primary supervision with one
scene-level Sit target.  For each scene, the primary target contains all three
physically verified sitting instances at the same time:

- `room_0101`: `bed_01`, `chair_01`, `chair_06`
- `room_0102`: `bed_01`, `chair_05`, `chair_06`
- locked development (materialized later): `room_0201` has `bed_01`,
  `chair_03`, `chair_02`

The source Teacher-v7 dataset is read-only.  The builder creates a new dataset
root and refuses to overwrite an existing output.  The pre-lock data gate
recomputes the two train scenes from all 48 sealed train dense-contact maps.
The independent validator repeats that computation instead of trusting the
generated manifest.  It hash-binds the `room_0201` source record but does not
open any of its arrays; development materialization is deferred until after
checkpoint and inference-policy lock.

The pre-lock metadata audit also proves that both train rooms reuse the sealed
six `hc_hd` recordings and that the locked room uses six disjoint `hcw_hdw`
recordings.  Recording reuse/separation is established by collection, source
group and source sample IDs.  Scene-local motion TXT hashes are still verified
inside each scene but are not used as cross-scene identity: placing one source
recording in two rooms legitimately produces different world-coordinate bytes.
Older non-High-Desk rows are honestly classified as scene-held-out evidence
whose canonical recording families may be shared; they are not misrepresented
as fully source-independent.

Unverified Chair/Bed instances are `unknown/ignore`; they are never silently
converted to negative labels.  TV, Desk and Whiteboard object points remain
explicit semantic negatives.  Relation, distance, target instance and motion
identity never enter the Teacher forward path.

The blocking positive loss is an equal-weight macro over the three verified
object masks.  Environment contact is retained only as auxiliary evidence, so
the much larger floor region cannot swamp either Chair.

This delivery deliberately authorizes only the next CUDA preflight.  It does
not start LoRA training.  The generated all-sittable GT must first pass both
the numerical validator and the visual three-instance audit described in
`TEACHER_V9_ALL_SITTABLE_DATA_GATE_RUNBOOK.md`.
