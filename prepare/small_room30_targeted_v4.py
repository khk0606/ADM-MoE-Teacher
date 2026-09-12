"""Presence-only repair of four audited Sit paths in exactly three rooms.

Whole target-instance labels are supervision/evaluation only. No GT heat extent,
channel support, background fit, or scene-id routing enters model inference.
"""
import json
import math
from pathlib import Path
import numpy as np
from small_room30_adm_dataset import inside, sha
from small_room30_targeted_v2 import source_contract as previous_contract, FAILURES, SOURCE_SHA

POLICY = dict(version='small_room30_targeted_v4', max_updates=12, lr=1e-5,
    retry_lr_scales=[1., .25], grad_clip=1., activation_threshold=.3,
    minimum_hit_fraction=.25, training_margin=.5, min_target_gain=.0001,
    target_rooms=['small_room_0205', 'small_room_0403', 'small_room_0603'],
    target_paths=FAILURES, target_weight=1., preservation_weight=1.,
    initialization='original1560, fresh AdamW; never resume rejected attempts',
    loss='missing target instance top25% any_joint hinge; presence-only preservation',
    acceptance='no previously present target lost; assigned missing target improves',
    diagnostics_only=['GT extent', 'MAE', 'background', 'surface mean/hit drop'],
    visual_approval_required=True, teacher_checkpoint_authorized=False)
FILES = ['prepare/small_room30_targeted_v4.py', 'prepare/small_room30_targeted_v4_runtime.py',
    'prepare/run_small_room30_targeted_v4.py', 'prepare/evaluate_small_room30_targeted_v4.py',
    'prepare/test_small_room30_targeted_v4.py', 'prepare/view_small_room30_candidates_v4.py',
    'prepare/review_small_room30_targeted_v4.py']


def instance_support(data, index):
    row = data.samples[index]
    source = inside(data.root, data.index['source_contact_index'])
    motions = {m['id']: m for m in json.loads(source.read_text())['motions']}
    folder = source.parent / 'scenes' / row['scene_id']
    meta = json.loads((folder / 'raw_scene_manifest.json').read_text())
    lookup = {m['stable_instance_id']: m['instance_id'] for m in meta['instances']}
    if len(lookup) != len(meta['instances']) or len(set(lookup.values())) != len(lookup):
        raise ValueError('Ambiguous furniture labels')
    with np.load(folder / 'supervision_only.npz', allow_pickle=False) as z:
        labels = z['instance_ids'].copy()
    if labels.shape != (8192,): raise ValueError('Wrong instance point order/shape')
    masks, names, targets = [], [], []
    for mid in row['source_motion_ids']:
        motion = motions[mid]
        if motion['scene_id'] != row['scene_id'] or motion['action'] != row['action']:
            raise ValueError('Wrong target action/scene')
        target = motion['target_instance_id']; mask = labels == lookup[target]
        if mask.sum() < 16: raise ValueError('Missing furniture geometry')
        masks.append(mask); names.append(mid); targets.append(target)
    if not masks or len(set(targets)) != len(targets): raise ValueError('Ambiguous targets')
    return np.stack(masks), names, targets


def presence(raw, masks, names):
    if raw.shape != (8192, 6) or not np.isfinite(raw).all(): raise ValueError('Invalid raw map')
    if masks.shape != (len(names), 8192) or masks.dtype != bool: raise ValueError('Invalid masks')
    instances = []
    for mask, name in zip(masks, names):
        values = raw.max(axis=1)[mask].astype(np.float64)
        if len(values) < 16: raise ValueError('Tiny/empty target')
        count = int((values >= POLICY['activation_threshold']).sum())
        k = int(math.ceil(len(values) * POLICY['minimum_hit_fraction']))
        instances.append(dict(id=name, points=len(values), active_points=count,
            hit_fraction=count / len(values), present=count >= k,
            top_mean=float(np.sort(values)[-k:].mean())))
    return dict(instances=instances, all_furniture_present=all(r['present'] for r in instances))


def source_contract(data, source_path, evaluation_path, diagnosis_path, repo):
    source, report, old, trace = previous_contract(data, source_path, evaluation_path, diagnosis_path, repo)
    from small_room30_targeted_runtime import reference
    panel = []
    for item in old['failures'] + old['guards']:
        case = item['case']; masks, names, _ = instance_support(data, case['index'])
        panel.append(dict(case=case, presence=presence(reference(evaluation_path, case), masks, names)))
    for item in panel[:4]:
        missing = [r['id'] for r in item['presence']['instances'] if not r['present']]
        if missing != [item['case']['scene_id'] + '__100__sit']:
            raise ValueError('Whole-instance audit no longer matches missing chair100')
    bound = dict(old, policy=POLICY, presence_panel=panel,
        preserved_v2_code_sha256=old['code_sha256'], code_sha256={n:sha(repo/n) for n in FILES})
    return source, report, bound, trace


def plan(data, bound, seed):
    transitions = bound['transitions']
    if [r['case']['filename'] for r in transitions] != FAILURES: raise ValueError('Changed target paths')
    for row in transitions:
        c = row['case']
        if c['scene_id'] not in POLICY['target_rooms'] or c['action'] != 'sit': raise ValueError('Outside repair scope')
    # All 12 guard paths get a free-prefix preservation loss once per cycle.
    guards = [r['case'] for r in bound['guards']]
    if len(guards) != 12: raise ValueError('Need 12 preservation paths')
    return [dict(case=transitions[n%4]['case'], timesteps=transitions[n%4]['timesteps'],
        guards=[guards[(a*4+n)%12] for a in range(3)]) for n in range(POLICY['max_updates'])]


def acceptance(candidate, previous, assigned):
    rows, oldrows = candidate['rows'], previous['rows']
    if len(rows) != 16 or [r['case'] for r in rows] != [r['case'] for r in oldrows]:
        raise ValueError('Changed canary panel')
    if assigned['filename'] not in FAILURES: raise ValueError('Outside repair scope')
    reasons = []
    for row, old in zip(rows, oldrows):
        before = row['before_presence']['instances']; after = row['after_presence']['instances']
        prior = old['after_presence']['instances']
        if [r['id'] for r in before] != [r['id'] for r in after] or [r['id'] for r in prior] != [r['id'] for r in after]:
            raise ValueError('Target identity changed')
        for b, p, a in zip(before, prior, after):
            if (b['present'] or p['present']) and not a['present']:
                reasons.append('lost_target:' + row['case']['filename'] + ':' + a['id'])
    row = next(r for r in rows if r['case'] == assigned)
    old = next(r for r in oldrows if r['case'] == assigned)
    mid = assigned['scene_id'] + '__100__sit'
    new = next(r for r in row['after_presence']['instances'] if r['id'] == mid)
    prior = next(r for r in old['after_presence']['instances'] if r['id'] == mid)
    gain = new['top_mean'] - prior['top_mean']
    if not np.isfinite(gain): raise ValueError('Nonfinite target gain')
    if not new['present'] and gain < POLICY['min_target_gain']:
        reasons.append('no_assigned_target_progress')
    return dict(accepted=not reasons, reasons=reasons, assigned_target_top_mean_gain=gain,
        visual_approval_required=True, teacher_checkpoint_authorized=False)


def check_ready(report, bound):
    keys = ['source_reproduced', 'missing_target_gradients', 'all_80_forward_finite',
        'disposable_update', 'optimizer_adapter_rollback', 'disk_roundtrip', 'frozen_unchanged']
    if (report.get('status') != 'TARGETED_V4_RUNTIME_READY' or report.get('binding') != bound
            or report.get('teacher_checkpoint_authorized') is not False
            or not all(report.get('checks', {}).get(k) is True for k in keys)):
        raise ValueError('Missing/stale v4 readiness')


def validate_run(data, path, repo):
    report = json.loads(Path(path).read_text())
    source, _, bound, _ = source_contract(data, Path(report['source_summary']),
        Path(report['evaluation_summary']), Path(report['diagnosis_summary']), repo)
    cp = report.get('last_checkpoint')
    if (report.get('status') not in ('TARGETED_V4_STOPPED_REJECTED', 'TARGETED_V4_FINISHED_REVIEW_REQUIRED')
            or report.get('binding') != bound or report.get('teacher_checkpoint_authorized') is not False
            or not cp or not 1561 <= cp['step'] <= 1572 or report['completed_step'] != cp['step']
            or sha(cp['path']) != cp['sha256']): raise ValueError('No validated v4 saved checkpoint')
    return report, source, bound
