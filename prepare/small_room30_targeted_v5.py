"""Two visually rejected paths; fixed reviewed1570 initialization, positive-only repair."""
import json
import math
from pathlib import Path
import numpy as np
from small_room30_adm_dataset import sha
from small_room30_targeted_v4 import instance_support, presence, validate_run as validate_v4

TARGETS = ['small_room_0205__sit_anywhere__g0.npz', 'small_room_0403__sit_anywhere__g2.npz']
PASSED = ['small_room_0403__sit_anywhere__g1.npz', 'small_room_0603__sit_anywhere__g0.npz']
SOURCE_SHA = '9950890be6d4c2818394ddeb30b592ec805a6b3f7d30f72e7091a7d1354a0300'
RUN_SHA = 'c694a4372a54e2404da8f1ebdc163900c64e88074b6282ac6d622332a5c44baf'
MAP_SHA = '6e9c727b97606e313bff21f051c3545b270324110a8b9647e4b3b788c679a704'
POLICY = dict(version='small_room30_targeted_v5', max_updates=16, lr=1e-5,
    retry_lr_scales=[1., .25], grad_clip=1., target_paths=TARGETS,
    target_rooms=['small_room_0205', 'small_room_0403'], target_fraction=.5,
    training_margin=.5, completion_floor=.4, min_target_gain=.0001,
    target_weight=1., preservation_weight=1.,
    initialization='reviewed v4 saved1570; fresh AdamW, no old optimizer resume',
    loss='final t0 current-policy prediction; top50% target hinge, positive preservation',
    gradient_scope='499-step detached current-policy prefix; gradient through final denoiser only',
    preservation_anchor='reviewed1570, 16 saved paths only',
    diagnostics_only=['GT extent', 'MAE', 'background', 'non-target activation'],
    visual_approval_required=True, teacher_checkpoint_authorized=False)
FILES = ['prepare/'+n+'.py' for n in ('small_room30_targeted_v5',
    'small_room30_targeted_v5_runtime', 'run_small_room30_targeted_v5',
    'test_small_room30_targeted_v5', 'view_small_room30_candidates_v5',
    'review_small_room30_targeted_v5')]


def target_stats(raw, mask):
    if raw.shape != (8192, 6) or not np.isfinite(raw).all() or mask.shape != (8192,) or mask.dtype != bool:
        raise ValueError('Invalid target map/mask')
    values = np.sort(raw.max(axis=1)[mask].astype(np.float64))
    if len(values) < 16: raise ValueError('Tiny target')
    k = int(math.ceil(len(values)*POLICY['target_fraction'])); top = values[-k:]
    return dict(top_mean=float(top.mean()), coverage_floor=float(top.min()),
        hit_fraction=float((values >= .3).mean()),
        hinge=float(np.maximum(POLICY['training_margin']-top, 0.).__pow__(2).mean()),
        optimization_complete=bool(top.min() >= POLICY['completion_floor']))


def source_contract(data, source_path, evaluation_path, diagnosis_path, repo):
    if sha(source_path) != RUN_SHA: raise ValueError('Need the reviewed v4 run01 summary')
    run, original, old = validate_v4(data, source_path, repo)
    if run['last_checkpoint']['sha256'] != SOURCE_SHA or run['completed_step'] != 1570:
        raise ValueError('Need reviewed saved1570, not1560 or an unsaved candidate')
    if sha(evaluation_path) != old['evaluation_sha256']:
        raise ValueError('Original full evaluation differs')
    if Path(diagnosis_path).resolve() != Path(run['diagnosis_summary']).resolve():
        raise ValueError('Diagnosis provenance differs')
    receipt = [a for t in run['transactions'] for a in t['attempts'] if a['accepted']][-1]
    ref = Path(source_path).resolve().parent/'proposal_12_try_1'/'summary.json'
    if sha(ref) != MAP_SHA or receipt['canary_sha256'] != MAP_SHA:
        raise ValueError('1570 reference maps differ')
    maps = json.loads(ref.read_text())
    if maps['adapter_digest'] != run['last_checkpoint']['adapter_digest']:
        raise ValueError('1570 map/checkpoint identity differs')
    cases = [r['case'] for r in old['failures']+old['guards']]
    if [r['case'] for r in maps['rows']] != cases or len(cases) != 16:
        raise ValueError('1570 reference panel differs')
    for row in maps['rows']:
        c = row['case']; raw = reference(ref, c)
        masks, names, _ = instance_support(data, c['index'])
        p = presence(raw, masks, names)
        if p != row['after_presence']: raise ValueError('1570 presence receipt differs')
    bound = dict(old, policy=POLICY, source_checkpoint=run['last_checkpoint'],
        original_source_checkpoint=old['source_checkpoint'], start_binding=old,
        start_run_sha256=RUN_SHA, reference_summary=str(ref), reference_sha256=MAP_SHA,
        code_sha256={n:sha(repo/n) for n in FILES})
    return run, maps, bound, []


def reference(summary, case):
    path = Path(summary); report = json.loads(path.read_text())
    row = next(r for r in report['rows'] if r['case'] == case)
    name = case['filename']
    if Path(name).name != name: raise ValueError('Unsafe case name')
    file = path.parent/name
    if sha(file) != row['sha256']: raise ValueError('1570 reference map hash differs')
    with np.load(file, allow_pickle=False) as z: raw = z['adapted_raw'].copy()
    if raw.shape != (8192,6) or not np.isfinite(raw).all(): raise ValueError('Invalid1570 map')
    return raw


def plan(data, bound, seed):
    cases = {r['case']['filename']:r['case'] for r in bound['failures']+bound['guards']}
    guards = [r['case'] for r in bound['guards']]
    if len(guards) != 12: raise ValueError('Need12 guards')
    for name in TARGETS:
        c = cases[name]
        if c['scene_id'] not in POLICY['target_rooms'] or c['action'] != 'sit':
            raise ValueError('Outside two-path scope')
    # Protect both recently accepted paths at every update; rotate all12 older guards.
    return [dict(case=cases[TARGETS[n%2]], timesteps=[0],
        guards=[cases[k] for k in PASSED]+[guards[(n+a*4)%12] for a in range(3)])
        for n in range(POLICY['max_updates'])]


def acceptance(candidate, previous, assigned):
    if assigned['filename'] not in TARGETS: raise ValueError('Outside two-path repair')
    rows, prior = candidate['rows'], previous['rows']
    if len(rows) != 16 or [r['case'] for r in rows] != [r['case'] for r in prior]:
        raise ValueError('Changed canary panel')
    reasons = []
    for r, old in zip(rows, prior):
        groups = [x['instances'] for x in (r['before_presence'], old['after_presence'], r['after_presence'])]
        if not all([p['id'] for p in g] == [p['id'] for p in groups[0]] for g in groups):
            raise ValueError('Changed target identities')
        for initial, before, after in zip(*groups):
            if (initial['present'] or before['present']) and not after['present']:
                reasons.append('lost_target:'+r['case']['filename']+':'+after['id'])
    r = next(r for r in rows if r['case']==assigned)
    old = next(r for r in prior if r['case']==assigned)
    gain = r['repair']['top_mean']-old['repair']['top_mean']
    if not np.isfinite(gain): raise ValueError('Nonfinite gain')
    # Old25% presence does NOT exempt0205 from actual endpoint progress.
    if not r['repair']['optimization_complete'] and gain < POLICY['min_target_gain']:
        reasons.append('no_assigned_final_output_progress')
    return dict(accepted=not reasons, reasons=reasons, assigned_target_top_mean_gain=gain,
        visual_approval_required=True, teacher_checkpoint_authorized=False)


def check_ready(report, bound):
    keys = ['source_reproduced','missing_target_gradients','all_80_forward_finite',
        'disposable_update','optimizer_adapter_rollback','disk_roundtrip','frozen_unchanged']
    if (report.get('status') != 'TARGETED_V5_RUNTIME_READY' or report.get('binding') != bound
            or report.get('teacher_checkpoint_authorized') is not False
            or not all(report.get('checks',{}).get(k) is True for k in keys)):
        raise ValueError('Missing/stale v5 readiness')
