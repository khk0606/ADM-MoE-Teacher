"""Separate, bounded research experiment. No automatic Teacher approval."""
import json
from pathlib import Path
import random

from small_room30_adm_dataset import sha
from small_room30_coverage import validate_experiment

POLICY = dict(version='small_room30_onpolicy_v1', lr=0.00002, max_updates=60,
    onpolicy_weight=1.0, replay_weight=1.0, grad_clip=1.0,
    early_timesteps=list(range(499, 489, -1)), later_timesteps=[450, 300, 100, 0],
    early_fraction='4 of every 5 updates', refresh='current adapter every update; no cached training states',
    gradient='detached free x_t; one differentiable clean prediction; no full-trajectory backprop',
    inputs='XYZ RGB text only; GT and instance labels only in loss',
    initialization='exact coverage step 1500; fresh AdamW', teacher_checkpoint_authorized=False)
FILES = ['prepare/small_room30_onpolicy.py', 'prepare/small_room30_onpolicy_runtime.py',
    'prepare/run_small_room30_onpolicy.py', 'prepare/evaluate_small_room30_onpolicy.py',
    'prepare/test_small_room30_onpolicy.py']


def timestep_plan(count, seed):
    if not 1 <= count <= POLICY['max_updates']:
        raise ValueError('This pilot permits 1..60 updates only')
    rng = random.Random(seed + 86117)
    early, late, result = [], [], []
    for i in range(count):
        pool = late if i % 5 == 4 else early
        if not pool:
            pool.extend(POLICY['later_timesteps'] if i % 5 == 4 else POLICY['early_timesteps'])
            rng.shuffle(pool)
        result.append(pool.pop())
    return result


def source_contract(data, source_path, diagnostic_path, repo):
    source, original, parent = validate_experiment(data, source_path, repo)
    if source['completed_step'] != 1500:
        raise ValueError('This pilot is bound to the diagnosed step 1500 source')
    diagnostic_path = Path(diagnostic_path)
    diag = json.loads(diagnostic_path.read_text())
    manifest_path = diagnostic_path.parent / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if (diag.get('status') != 'DIAGNOSTIC_COMPLETE' or not diag.get('model_state_unchanged')
            or diag.get('training_started') is not False or len(diag.get('rows', [])) != 6
            or sha(manifest_path) != diag['manifest_sha256']
            or manifest['checkpoint_sha256'] != source['last_checkpoint']['sha256']
            or manifest['binding'] != parent
            or manifest['training_summary_sha256'] != sha(source_path)):
        raise ValueError('Diagnostic/source provenance mismatch')
    scenes = set()
    for row in diag['rows']:
        case = row['case']
        if (case['scene_id'] in scenes or not case['sample_id'].endswith('__sit_anywhere')
                or case['generation'] != 0 or not row['metrics']['replay_matches_saved']
                or row['metrics']['replay_max_abs_error'] != 0):
            raise ValueError('Need six unique, exact-replayed Sit diagnostic cases')
        scenes.add(case['scene_id'])
        path = diagnostic_path.parent / 'cases' / case['filename']
        if path.resolve().parent != (diagnostic_path.parent / 'cases').resolve() or sha(path) != row['sha256']:
            raise ValueError('Diagnostic array changed/unsafe')
    bound = dict(policy=POLICY, original_binding=original['binding'], parent_binding=parent,
        source_summary_sha256=sha(source_path), source_checkpoint=source['last_checkpoint'],
        diagnostic_summary_sha256=sha(diagnostic_path),
        code_sha256={name: sha(repo/name) for name in FILES})
    return source, diag, bound


def check_ready(ready, bound):
    checks = ['source_restored', 'diagnostic_t490_reproduced', 'failed_target_coverage_gradients',
              'all_80_forward_finite', 'updates_finite_nonzero', 'frozen_unchanged', 'disk_roundtrip']
    if (ready.get('status') != 'ONPOLICY_RUNTIME_READY' or ready.get('binding') != bound
            or ready.get('teacher_checkpoint_authorized') is not False
            or not all(ready.get('checks', {}).get(k) is True for k in checks)):
        raise ValueError('Missing/stale on-policy readiness; run smoke first')


def validate_run(data, path, repo):
    report = json.loads(Path(path).read_text())
    source, _, bound = source_contract(data, Path(report['source_summary']), Path(report['diagnostic_summary']), repo)
    cp = report['last_checkpoint']
    if (report.get('status') != 'ONPOLICY_PILOT_FINISHED_NOT_EVALUATED'
            or report.get('binding') != bound or report.get('teacher_checkpoint_authorized') is not False
            or report['completed_step'] != cp['step'] or sha(cp['path']) != cp['sha256']):
        raise ValueError('Changed/incomplete pilot checkpoint')
    return report, source, bound
