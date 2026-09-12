"""Transition repair experiment, from unchanged1560; never auto-approved."""
import json
from pathlib import Path
from small_room30_adm_dataset import sha
from small_room30_targeted import source_contract as old_contract, check_ready as check_old_ready, FAILURES, SOURCE_SHA
from small_room30_training_contract import schedule

DIAGNOSIS_SHA='140a713fddbbf13e90b01321b70d06df755843f217a546074fc962504a2c8a8e'
POLICY=dict(version='small_room30_targeted_v2',max_updates=4,lr=1e-5,
    retry_lr_scales=[1.,.25],grad_clip=1.,target_weight=1.,replay_weight=1.,
    guard_surface_weight=10.,guard_map_weight=.1,guard_loss_tolerance=.02,
    guard_mean_drop_limit=.02,guard_hit_drop_limit=.05,min_target_mean_gain=.0001,
    timesteps='last observed present-to-missing pair, plus t0; NOT maximum loss only',
    guards='weakest Sit, Lie and Write guard in every update; all12 checked after every proposal',
    rollback='adapter AND AdamW state; no rejected update retained',
    initialization='original1560, fresh optimizer; no1564 warm start',
    teacher_checkpoint_authorized=False)
FILES=['prepare/small_room30_targeted_v2.py','prepare/small_room30_targeted_v2_runtime.py',
    'prepare/run_small_room30_targeted_v2.py','prepare/evaluate_small_room30_targeted_v2.py',
    'prepare/test_small_room30_targeted_v2.py']


def transition(row):
    records=row['records']
    if len(records)<2 or any(records[i]['timestep']<=records[i+1]['timestep'] for i in range(len(records)-1)):
        raise ValueError('Trace must follow decreasing sampler timesteps')
    candidates=[i for i in range(len(records)-1) if records[i]['score']['all_furniture_present']
        and all(not r['score']['all_furniture_present'] for r in records[i+1:])]
    if not candidates: raise ValueError('No observed stable present-to-missing transition')
    i=candidates[-1];ts=[records[i]['timestep'],records[i+1]['timestep'],0]
    if len(set(ts))!=3:raise ValueError('Need distinct before/after/final probes')
    return dict(case=row['case'],timesteps=ts,observed_bracket=ts[:2])


def source_contract(data,source_path,evaluation_path,diagnosis_path,repo):
    source,report,old=old_contract(data,source_path,evaluation_path,repo)
    if sha(diagnosis_path)!=DIAGNOSIS_SHA:raise ValueError('Need the audited original targeted_ready01 summary')
    diagnosis=json.loads(Path(diagnosis_path).read_text());check_old_ready(diagnosis,old)
    trace=diagnosis['trace']
    if [r['case']['filename'] for r in trace]!=FAILURES:raise ValueError('Changed diagnostic panel')
    transitions=[transition(r) for r in trace]
    critical=[]
    for action in ('sit','lie','write_board'):
        candidates=[g for g in old['guards'] if g['case']['action']==action]
        critical.append(min(candidates,key=lambda g:(min(i['hit_fraction'] for i in g['score']['instances']),g['case']['filename']))['case'])
    if critical[-1]['filename']!='small_room_0404__go_write_board__g1.npz':
        raise ValueError('Expected exact weak Write0404/g1 guard')
    bound=dict(old,policy=POLICY,diagnosis_sha256=sha(diagnosis_path),transitions=transitions,
        critical_guards=critical,code_sha256={n:sha(repo/n) for n in FILES})
    return source,report,bound,trace


def plan(data,bound,seed):
    replay=schedule(data,32,seed+730001)
    return [dict(case=t['case'],timesteps=t['timesteps'],guards=bound['critical_guards'],
        replay=replay[8*n:8*n+8]) for n,t in enumerate(bound['transitions'])]


def check_ready(report,bound):
    keys=['source_reproduced','transition_gradients','all_80_forward_finite',
        'disposable_update','optimizer_adapter_rollback','disk_roundtrip','frozen_unchanged']
    if (report.get('status')!='TARGETED_V2_RUNTIME_READY' or report.get('binding')!=bound
            or report.get('teacher_checkpoint_authorized') is not False
            or not all(report.get('checks',{}).get(k) is True for k in keys)):
        raise ValueError('Missing/stale v2 readiness')


def validate_run(data,path,repo):
    report=json.loads(Path(path).read_text())
    source,_,bound,_=source_contract(data,Path(report['source_summary']),Path(report['evaluation_summary']),
        Path(report['diagnosis_summary']),repo)
    cp=report.get('last_checkpoint')
    if (report.get('status') not in ('TARGETED_V2_FINISHED_REVIEW_REQUIRED','TARGETED_V2_STOPPED_REJECTED')
            or report.get('binding')!=bound or report.get('teacher_checkpoint_authorized') is not False
            or not cp or not 1561<=cp['step']<=1564 or report['completed_step']!=cp['step']
            or sha(cp['path'])!=cp['sha256']):
        raise ValueError('No validated accepted v2 checkpoint; keep original1560')
    return report,source,bound
