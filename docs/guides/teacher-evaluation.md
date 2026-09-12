# Saved1578: full30-room evaluation, no training

The user visually accepted0205/g0 and0403/g2 on2026-09-10. Fix the adapter at
v5 P12 saved1578. This package adds evaluation scripts only; it never changes
training code, model weights, labels, or the acceptance fields of prior runs.

Evaluate all30 rooms,80 prompt rows,3 paired generations per row:240 cases.
Original ancestral DDPM500 steps, scene/text conditioning only, no optimizer,
no GT conditioning or map editing. Frozen Base is reused from the validated
original full evaluation, with checksums, seeds, points, GT and supports checked.
Only1578 is regenerated. The16 saved1578 paths must reproduce within1e-5
absolute/relative tolerance; differences stop evaluation instead of silently
substituting maps. All weights are frozen and checked unchanged afterwards.

Comparison: full240 vs original1560, plus16 known paths vs1570. No full1570
evaluation exists, so don't claim full240 preservation relative to1570.
Whole-instance visibility and GT metrics are review diagnostics, not an automatic
instruction to retrain. This development evaluation is not held-out validation
or downstream Teacher deployment approval.

## Server execution

Copy the ZIP and checksum into ~/AMDM, activate the existing afford environment:

```bash
sha256sum -c small_room30_v5_full_evaluation_v1.zip.sha256 &&
unzip -n small_room30_v5_full_evaluation_v1.zip &&
bash scripts/small_room30/evaluate_v5_full.sh
```

The script runs CPU tests, evaluates240 cases, validates outputs and packages
small_room30_targeted_v5_full_review01.zip automatically. The checkpoint files
stay on the server and are not included in the ZIP.

Progress is shown as[PROGRESS] n/240 with approximate remaining minutes based on
newly generated cases.500/500 is one case, not the whole evaluation. Full log:
outputs/small_room30_targeted_v5_eval_full01.console.log.

If interrupted, rerun the SAME evaluation command. This resumes evaluation only,
not training. The manifest must match exactly; cached files and metrics are
verified before reuse. Do not delete output directories or change source files.

```bash
PORT=8092 bash scripts/small_room30/view_v5_full.sh
```

Forward8092 to an available local port as needed. This viewer shows the FULL
saved1578 evaluation (not the16-case training canary). It does not train.

Local CPU tests and evidence audits are not real CUDA generation. This Mac does
not execute the user's AMDM GPU process; run the command on the existing server.
