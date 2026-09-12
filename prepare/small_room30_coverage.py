"""Additive, development-only furniture coverage experiment; no original file edits.

Instance labels are used ONLY for supervision/evaluation, never model inputs.
This is an experimental objective, not a demonstrated cure for rollout omissions.
"""
import json
from pathlib import Path
import numpy as np
from small_room30_adm_dataset import inside, sha
from small_room30_training_contract import binding, SCHEMA

POLICY = dict(version='small_room30_surface_coverage_v1', lr=0.00005,
    activation_threshold=0.30, minimum_surface_hit_fraction=0.25,
    min_core_points=16, training_top_fraction=0.50, training_activation_margin=0.50,
    coverage_weight=2.0, background_weight=1.0, background_margin=0.15,
    max_background_point_hit_fraction=0.05, background_gt_threshold=0.05,
    scope='sit furniture presence only; GT extent/MAE are secondary diagnostics',
    optimizer='fresh AdamW, no source optimizer moments',
    noise_and_schedule='unchanged original GT-noised uniform-timestep training',
    teacher_checkpoint_authorized=False)
FILES = ['prepare/small_room30_coverage.py', 'prepare/small_room30_coverage_runtime.py',
         'prepare/run_small_room30_coverage.py', 'prepare/evaluate_small_room30_coverage.py',
         'prepare/score_small_room30_coverage.py']


def surface_support(data, index):
    """Return source-motion active channels restricted to the TARGET furniture.

Unlike the old full spatial support this excludes halos on the floor/TV/desk.
At least 16 furniture points must exist: no silently empty/fallback masks.
"""
    row = data.samples[index]
    source = inside(data.root, data.index['source_contact_index'])
    motions = {m['id']: m for m in json.loads(source.read_text())['motions']}
    sd = source.parent / 'scenes' / row['scene_id']
    meta = json.loads((sd / 'raw_scene_manifest.json').read_text())
    instances = meta['instances']
    lookup = {m['stable_instance_id']: m['instance_id'] for m in instances}
    if len(lookup) != len(instances) or len(set(lookup.values())) != len(lookup):
        raise ValueError('Ambiguous instance mapping')
    with np.load(sd / 'supervision_only.npz', allow_pickle=False) as z:
        labels = z['instance_ids'].copy()
    if labels.shape != (8192,): raise ValueError('Bad instance label shape')
    masks, names, targets = [], [], []
    for mid in row['source_motion_ids']:
        m = motions[mid]
        if m['scene_id'] != row['scene_id'] or m['action'] != row['action']:
            raise ValueError('Cross-scene/action motion')
        target = m['target_instance_id']
        with np.load(inside(source.parent, m['file']), allow_pickle=False) as z:
            active = z['affordance'] >= .30
        mask = active & (labels == lookup[target])[:, None]
        if int(mask.any(axis=1).sum()) < POLICY['min_core_points']:
            raise ValueError('Insufficient target surface support: ' + mid)
        masks.append(mask); names.append(mid); targets.append(target)
    if not masks or len(set(targets)) != len(targets):
        raise ValueError('Missing/duplicated target furniture')
    return np.stack(masks), names, targets


def surface_metrics(raw, gt, masks, names):
    """Presence, NOT GT-extent fit. Values and rules fixed before extra training."""
    if raw.shape != gt.shape or raw.shape != (8192, 6) or not np.isfinite(raw).all():
        raise ValueError('Invalid raw map')
    if masks.shape != (len(names), 8192, 6) or not names:
        raise ValueError('Invalid surface masks')
    instances = []
    for mask, name in zip(masks.astype(bool), names):
        core = mask.any(axis=1)
        if core.sum() < POLICY['min_core_points']: raise ValueError('Empty/tiny furniture support')
        scores = np.where(mask, raw, -np.inf).max(axis=1)[core]
        hit = float((scores >= POLICY['activation_threshold']).mean())
        instances.append(dict(id=name, core_points=int(core.sum()), hit_fraction=hit,
            present=hit >= POLICY['minimum_surface_hit_fraction']))
    bg = gt.max(axis=1) <= POLICY['background_gt_threshold']
    bg_hit = float((raw.max(axis=1)[bg] >= POLICY['activation_threshold']).mean()) if bg.any() else None
    guard = bg_hit is not None and bg_hit <= POLICY['max_background_point_hit_fraction']
    return dict(instances=instances, all_furniture_present=all(x['present'] for x in instances),
        background_point_hit_fraction=bg_hit, background_guard=guard,
        coverage_pass=all(x['present'] for x in instances) and guard)


def audit_data(data):
    from validate_small_room30_adm_export import validate
    validation = validate(data.root)
    rows = []
    for i, row in enumerate(data.samples):
        if row['action'] != 'sit': continue
        masks, names, targets = surface_support(data, i)
        if len(targets) != 2: raise ValueError('Expected two Sit furniture candidates')
        gt = data.sample(i)['target']
        for mask in masks:
            if np.any(gt[mask] < .30 - 1e-6): raise ValueError('Surface absent from union target')
        rows.append(dict(sample_id=row['id'], targets=targets, motion_ids=names,
            surface_point_counts=[int(m.any(axis=1).sum()) for m in masks]))
    if len(rows) != 65: raise ValueError('Incomplete Sit audit')
    return dict(status='UNION_AND_TWO_FURNITURE_SURFACE_AUDIT_PASS', validation=validation,
        sit_prompt_count=len(rows), rows=rows, policy=POLICY,
        real_model_tested=False, teacher_checkpoint_authorized=False)


def original_source(data, summary_path, repo):
    """Fail closed against the source run; never relabel a changed original binding."""
    source = json.loads(Path(summary_path).read_text())
    if (source.get('schema') != SCHEMA or source.get('status') != 'RESEARCH_TRAINING_FINISHED_NOT_EVALUATED'
        or source.get('teacher_checkpoint_authorized') is not False or source.get('coverage_experiment')):
        raise ValueError('Expected ORIGINAL completed research training, not smoke/coverage')
    current = binding(data, source['binding']['inputs'], repo)
    if current != source['binding']: raise ValueError('Original source code/data/base changed')
    cp = source['last_checkpoint']
    if cp['step'] != source['completed_step'] or sha(cp['path']) != cp['sha256']:
        raise ValueError('Original checkpoint hash/step mismatch')
    return source


def experiment_binding(source, summary_path, repo):
    return dict(original_binding=source['binding'], policy=POLICY,
        source_summary_sha256=sha(summary_path), source_checkpoint=source['last_checkpoint'],
        code_sha256={name: sha(repo / name) for name in FILES})


def validate_checkpoint(saved, source):
    if (saved.get('schema') != SCHEMA or saved.get('purpose') != 'RESEARCH_TRAINING'
        or saved.get('binding') != source['binding'] or saved.get('seed') != source['seed']
        or saved.get('step') != source['completed_step']
        or saved.get('teacher_checkpoint_authorized') is not False):
        raise ValueError('Original checkpoint contract mismatch')


def validate_experiment(data, report_path, repo):
    report = json.loads(Path(report_path).read_text())
    if (report.get('schema') != SCHEMA or report.get('status') != 'RESEARCH_TRAINING_FINISHED_NOT_EVALUATED'
        or report.get('coverage_experiment') != POLICY['version']
        or report.get('teacher_checkpoint_authorized') is not False):
        raise ValueError('Not completed coverage research training')
    source_path = Path(report['source_summary'])
    source = original_source(data, source_path, repo)
    current = experiment_binding(source, source_path, repo)
    if report['binding'] != current: raise ValueError('Coverage experiment binding changed')
    cp = report['last_checkpoint']
    if cp['step'] != report['completed_step'] or sha(cp['path']) != cp['sha256']:
        raise ValueError('Coverage checkpoint changed')
    return report, source, current
