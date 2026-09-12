# v5: two user-rejected chairs only

Repair only chair100 in0205/sit_anywhere/g0 and0403/sit_anywhere/g2. The user
accepted the other inspected paths. This is not authorization to suppress TV
activation, fit GT extent, change labels, or retrain all scenes.

Start from reviewed v4 P12 saved1570 (SHA256
9950890be6d4c2818394ddeb30b592ec805a6b3f7d30f72e7091a7d1354a0300), fresh AdamW.
The run summary and P12 maps are pinned to the uploaded review. Preserve all old
weights, outputs and code. Different/missing source files cause a stop.

## Changes and limits

- Do not skip0205 when only25% of its chair points activate. Optimize the best50%
  of whole-chair any_joint values toward0.5. A50% coverage floor of0.4 ends this
  bounded optimization; it is a scheduling proxy, NOT visual approval or GT truth.
- Use only the final t0 denoiser after a fresh499-step current-model prefix.
  The prefix is detached: this is NOT differentiation through all500 steps.
  This removes early-time surrogate losses that decreased in v4 without fixing
  0403/g2's final output. Real-model effectiveness remains to be checked on CUDA.
- Positive-only target loss; no GT heat extent/body-channel loss, background loss,
  negative TV loss, inference routing, clipping, smoothing, or map editing.
- Anchor preservation to1570, not1560. Protect all previously visible target
  points up to0.5. Include0403/g1 and0603/g0 in every gradient update, the assigned
  path's other furniture, and3 rotating guards. All16 saved paths are evaluated
  after every proposal; an individual previously present target cannot disappear.
- Shared LoRA can still change unmeasured paths. This16-path canary is NOT a
  full240-case/30-room regression guarantee. Fine-grained intensity/GT differences
  remain visual review evidence, not a claim of exact preservation.
- At most16 proposals, alternating the2 targets, with full/quarter LR rollback.
  Final-output top50% mean must improve on an unfinished assigned target. Both
  rejected attempts stop the pilot and restore adapter AND optimizer. No resume,
  automatic extension, Teacher approval, or downstream inference deployment.

## Run on the existing AMDM CUDA environment

Place the ZIP and checksum in ~/AMDM. From that folder:

```bash
sha256sum -c small_room30_targeted_patch_v5.zip.sha256 &&
unzip -n small_room30_targeted_patch_v5.zip &&
bash scripts/small_room30/run_targeted_v5.sh
```

The command verifies package/runtime dependencies quietly, runs CPU tests,
reproduces1570, checks endpoint gradients, performs disposable rollback and disk
roundtrip tests, then starts bounded research training. Full logs are saved in
outputs/small_room30_targeted_v5_{ready01,run01}.console.log. Outputs must not
already exist. Do not delete or rerun existing experiment directories.

The completed/stopped run creates small_room30_targeted_v5_review01.zip excluding
.pt weights. Keep all weights on the server and send that review ZIP, plus Viser
screenshots of0205/g0 and0403/g2. Also check0403/g1 and0603/g0 stayed usable.

```bash
PORT=8092 bash scripts/small_room30/view_candidates_v5.sh
```

Forward remote8092 to an available local port if needed. Do not kill unrelated
services on a busy port. The viewer reads saved maps only and defaults to the
last saved candidate, with Starting LoRA1570 as reference. It does not train.

This package contains no model weights. Local CPU tests cannot certify CUDA
memory usage, pretrained-model repair, or visual success. Full evaluation and
any later deployment remain separate steps after visual review.
