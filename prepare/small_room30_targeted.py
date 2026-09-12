"""Bounded 1560 repair contract; development evidence is never Teacher approval."""
import json
from pathlib import Path

import numpy as np

from small_room30_adm_dataset import sha
from small_room30_onpolicy import validate_run as validate_parent
from small_room30_evaluation_common import validate_report, panel, load_case, conditioning_arrays, supervision
from small_room30_coverage import surface_support, surface_metrics
from small_room30_training_contract import schedule

SOURCE_SHA = 'e2715f22b1d12f8b1c4168f2485e4e44c61ae00197196bb043e1e6f8261bc3e6'
FAILURES = ['small_room_0205__sit_anywhere__g0.npz',
    'small_room_0403__sit_anywhere__g1.npz', 'small_room_0403__sit_anywhere__g2.npz',
    'small_room_0603__sit_anywhere__g0.npz']
POLICY = dict(version='small_room30_targeted_v1', max_updates=12, lr=1e-5,
    trace_timesteps=[499,498,495,490,475,450,400,300,200,100,50,0],
    selected_timesteps=3, checkpoint_interval=4, grad_clip=1.,
    free_guard_weight=1., replay_weight=1.,
    guard_mae_increase_limit=.03, guard_background_increase_limit=.05,
    refresh='fresh current-adapter prefix on every training path',
    initialization='exact step1560; fresh AdamW; no optimizer resume',
    inputs='XYZ RGB text only; labels used only in loss',
    acceptance='target furniture presence; MAE/background are separate regression safety warnings',
    teacher_checkpoint_authorized=False)
FILES = ['prepare/small_room30_targeted.py', 'prepare/small_room30_targeted_runtime.py',
    'prepare/run_small_room30_targeted.py', 'prepare/evaluate_small_room30_targeted.py',
    'prepare/test_small_room30_targeted.py', 'prepare/summarize_small_room30_targeted.py']


def score(data, case, raw):
    gt = data.sample(case['index'])['target']
    masks, names, _ = surface_support(data, case['index'])
    result = surface_metrics(raw, gt, masks, names)
    result['mae'] = float(np.abs(raw.astype(np.float64)-gt).mean())
    return result


def source_contract(data, source_path, evaluation_path, repo):
    source, _, parent = validate_parent(data, source_path, repo)
    if source['completed_step'] != 1560 or source['last_checkpoint']['sha256'] != SOURCE_SHA:
        raise ValueError('Repair requires the audited step1560, not another source')
    report, manifest = validate_report(evaluation_path)
    if (manifest['mode'] != 'full' or manifest['cases'] != panel(data, 'full')
            or manifest['checkpoint_sha256'] != SOURCE_SHA or manifest['training_binding'] != parent
            or manifest['training_summary_sha256'] != sha(source_path)):
        raise ValueError('Need exact step1560 full K=3 source evaluation')
    scored = []
    for row in report['rows']:
        case = row['case']
        arrays = load_case(Path(evaluation_path).parent/'cases'/case['filename'])
        points, _ = conditioning_arrays(data, case['index'])
        gt, masks, ids = supervision(data, case['index'])
        if not all(np.array_equal(a,b) for a,b in [(arrays['points'],points), (arrays['gt'],gt),
                (arrays['active_masks'],masks), (arrays['motion_ids'],np.asarray(ids))]):
            raise ValueError('Source geometry/GT/point order differs: '+case['filename'])
        scored.append(dict(case=case, score=score(data,case,arrays['adapted_raw'])))
    failures = [r for name in FAILURES for r in scored if r['case']['filename'] == name]
    if len(failures) != 4 or any(r['score']['all_furniture_present'] for r in failures):
        raise ValueError('The four diagnosed source failures do not reproduce in saved arrays')
    # Four distinct successful rooms per action. Highest MAE first stresses weaker successes.
    guards = []
    for action in ('sit','lie','write_board'):
        rooms = set()
        candidates = sorted([r for r in scored if r['case']['action'] == action
            and r['score']['all_furniture_present']], key=lambda r:(-r['score']['mae'],r['case']['filename']))
        for row in candidates:
            if row['case']['scene_id'] in rooms: continue
            rooms.add(row['case']['scene_id']); guards.append(row)
            if len(rooms) == 4: break
        if len(rooms) != 4: raise ValueError('Insufficient distinct successful guard rooms: '+action)
    bound = dict(policy=POLICY, original_binding=parent['original_binding'], parent_binding=parent,
        source_summary_sha256=sha(source_path), source_checkpoint=source['last_checkpoint'],
        evaluation_sha256=sha(evaluation_path), evaluation_manifest_sha256=report['manifest_sha256'],
        failures=failures, guards=guards, code_sha256={n:sha(repo/n) for n in FILES})
    return source, report, bound


def plan(data, bound, trace, count, seed):
    if count not in (4,8,12): raise ValueError('Pilot allows only 4, 8, or 12 updates')
    if [r['case']['filename'] for r in trace] != FAILURES: raise ValueError('Trace order changed')
    replay = schedule(data, count*3, seed+630001)
    guards = bound['guards']
    result=[]
    for n in range(count):
        row=trace[n%4]; times=row['selected_timesteps']
        if len(times)!=3 or len(set(times))!=3 or any(t not in POLICY['trace_timesteps'] for t in times):
            raise ValueError('Invalid diagnosed timesteps')
        # Sit/Lie/Write guard cycle, each of four rooms used once over 12 updates.
        guard=guards[(n%3)*4+n//3]
        result.append(dict(case=row['case'], timestep=times[n//4], guard=guard['case'],
            replay=replay[3*n:3*n+3]))
    return result


def check_ready(ready, bound):
    if (ready.get('status')!='TARGETED_RUNTIME_READY' or ready.get('binding')!=bound
            or ready.get('teacher_checkpoint_authorized') is not False
            or not all(ready.get('checks',{}).get(k) is True for k in
                ['exact_paths_reproduced','corrective_gradients','all_80_forward_finite',
                 'disposable_update','disk_roundtrip','frozen_unchanged'])):
        raise ValueError('Missing/stale targeted readiness')
    if len(ready.get('trace',[])) != 4: raise ValueError('Missing trace')


def validate_run(data,path,repo):
    report=json.loads(Path(path).read_text())
    source, _, bound=source_contract(data,Path(report['source_summary']),Path(report['evaluation_summary']),repo)
    cp=report['last_checkpoint']
    if (report.get('status') not in ('TARGETED_PILOT_FINISHED_NOT_APPROVED','TARGETED_STOPPED_REGRESSION')
            or report.get('binding')!=bound or report.get('teacher_checkpoint_authorized') is not False
            or report['completed_step']!=cp['step'] or cp['step'] not in (1564,1568,1572)
            or sha(cp['path'])!=cp['sha256']):
        raise ValueError('Changed/incomplete targeted checkpoint')
    return report,source,bound
