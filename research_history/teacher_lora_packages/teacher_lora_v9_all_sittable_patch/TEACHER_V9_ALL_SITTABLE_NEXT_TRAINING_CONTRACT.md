# Teacher-v9 all-sittable training contract

## Primary target

For each scene, let `G_bed`, `G_normal_chair`, and `G_high_chair` be the three
physically verified instance consensus maps.  The only primary Teacher target
is:

```text
G_all = max(G_bed, G_normal_chair, G_high_chair)
```

Both object-agnostic prompts use the exact same `G_all` bytes.  Motion IDs,
target IDs, watch/write relations, object distance and selected-instance labels
are supervision metadata only and must not be passed to Teacher forward.

Each six-motion source stratum is first filtered to its bound target object plus
environment points.  Its deterministic consensus is the second-largest value
of six at every point/channel (nearest-rank q75), so one anomalous clip cannot
create a positive region.  High-Desk and legacy motions are separately
aggregated and then max-unioned because they refer to the same physical Chair.

## Role policy

- Positive: only an instance with a sealed, physically passing Sit motion.
- Unknown/ignore: Chair or Bed category without a verified motion.
- Explicit negative: TV, Desk and Whiteboard object points.
- Environment: retained as motion-contact evidence; not treated globally as a
  negative because feet and approach paths legitimately touch it.

The map is an affordance consensus, not one physical trajectory.  No synthetic
distance array may be fabricated for it.

## Next CUDA objective

The next gate must start from the sealed v5r4 Teacher with a fresh, zero-init
LoRA.  It must not load any v7, v7.1, v8 specialist, or late-union checkpoint.
Loss terms and their gradients must be audited independently before training:

```text
L_primary  = equal-weight mean(L_bed, L_normal_chair, L_high_chair)
L_union    = loss(prediction, G_all) on verified sitting-object points only
L_environment = bounded auxiliary contact loss on environment points only
L_negative = addition only on explicit-negative object points
L_prompt   = paired watch/write invariance on the same scene and noise
L_replay   = sealed v5 chair/bed/whiteboard preservation
L_reg      = bounded LoRA regularization
```

The three instance losses have equal weight regardless of point count.  There
must be no `outside = ~high_desk_target` penalty: that old v8 expression
incorrectly suppressed the normal Chair and Bed.

Hyperparameters are selected using only `room_0101` and `room_0102`.  The
`room_0201` arrays remain unread until the checkpoint and inference policy are
locked.

## Mandatory gates after this data delivery

1. CUDA zero-init parity and component-gradient preflight.
2. One-train-scene overfit smoke: a single output must contain all three
   verified instances simultaneously.
3. Fresh response-3.
4. Fresh calibration-12 with checkpoints 3/6/9/12.
5. Fresh bounded pilot-60; save only strictly eligible checkpoints.
6. Paired train K=1 and K=3 with identical Base/candidate noise.
7. Full train-only audit.
8. Locked `room_0201` development K=3; only here may development arrays be
   read.  Paper-test data remains unread.

Every rollout is generated per `scene x prompt x generation`, not once per
source motion.  Every output must pass Bed, normal-Chair and High-Chair metrics
individually.  A Bed-only map must fail even when its pooled MAE improves.

The blocking metrics are per-instance soft recall, active-support MAE, top-k
overlap and hotspot-centroid error, computed on that instance's verified object
points (not on the shared floor/approach path); report macro and worst-instance values.
Bed must be retained, both Chairs must improve, explicit-negative mass must be
bounded, and watch/write plus v5 replay preservation must pass.  Small diffuse
scene-wide values are diagnostic only unless they form a strong hotspot on an
explicit negative object.

This data gate intentionally does not invent numeric rollout cutoffs from
synthetic arrays.  Before the first optimizer update, the CUDA preflight must
measure the sealed v5r4 baseline on a fixed train-only panel, write every
absolute/relative cutoff and tie rule into a hashed policy JSON, and prove with
counterexamples that Bed-only, missing-normal-Chair, missing-High-Chair,
same-object displacement, and explicit-negative hotspot outputs fail.  The
policy hash, panel IDs, seeds, timesteps and noise hashes must then remain
unchanged through response-3, calibration, pilot and rollout selection.
