"""NumPy-only definitions for saved FREE-NOISE rollout auditing (no training)."""
import hashlib
import json
from pathlib import Path
import numpy as np
from small_room30_adm_dataset import inside, sha

SCHEMA = 'small_room30_free_rollout_evaluation_v1'
EVAL_FILES = ['prepare/small_room30_evaluation_common.py', 'prepare/evaluate_small_room30_teacher.py']


def panel(data, mode):
    if mode not in ('preview', 'full'):
        raise ValueError('Unknown panel mode')
    scenes = sorted(data.scenes)
    if mode == 'preview':
        # First layout per composition, selected BEFORE looking at predictions.
        scenes = sorted({min(s['id'] for s in data.scenes.values() if s['composition_group'] == g)
                         for g in {s['composition_group'] for s in data.scenes.values()}})
    indices = [i for i, r in enumerate(data.samples) if r['scene_id'] in scenes]
    return [dict(index=i, sample_id=data.samples[i]['id'], scene_id=data.samples[i]['scene_id'],
                 action=data.samples[i]['action'], generation=g,
                 filename='%s__g%d.npz' % (data.samples[i]['id'], g))
            for i in indices for g in range(1 if mode == 'preview' else 3)]


def seeds(case, seed=20260908):
    # Same scene/action/generation => same noise even across purpose prompt aliases.
    key = '%s|%s|%s|%s' % (seed, case['scene_id'], case['action'], case['generation'])
    raw = hashlib.sha256(key.encode()).digest()
    return [int.from_bytes(raw[:8], 'little') % (2**31-1),
            int.from_bytes(raw[8:16], 'little') % (2**31-1)]


def conditioning_arrays(data, index):
    """Read scene and text ONLY; no GT target or motion read in this function."""
    row = data.samples[index]
    with np.load(inside(data.root, data.scenes[row['scene_id']]['points_file']), allow_pickle=False) as z:
        points = z['points'].astype(np.float32)
    if points.shape != (8192, 6) or not np.isfinite(points).all():
        raise ValueError('Bad scene points')
    return points, dict(c_pc_xyz=np.ascontiguousarray(points[None, :, :3]),
                        c_pc_feat=np.ascontiguousarray(points[None, :, 3:] / 255.), c_text=[row['text']])


def supervision(data, index):
    """Used AFTER generation, to compute metrics, never fed to the model."""
    row = data.samples[index]
    gt = data.sample(index)['target']
    src = inside(data.root, data.index['source_contact_index'])
    records = {r['id']: r for r in json.loads(src.read_text())['motions']}
    masks = []
    for mid in row['source_motion_ids']:
        with np.load(inside(src.parent, records[mid]['file']), allow_pickle=False) as z:
            masks.append(z['affordance'] >= .30)
    return gt, np.stack(masks), row['source_motion_ids']


def metrics(raw, gt, masks, names):
    raw, gt = np.asarray(raw), np.asarray(gt)
    if raw.shape != gt.shape or raw.ndim != 2 or raw.shape[1] != 6:
        raise ValueError('Expected matching N x 6 maps')
    if not np.isfinite(raw).all() or not np.isfinite(gt).all():
        raise ValueError('Nonfinite generated/GT map')
    if np.any((gt < 0) | (gt > 1)) or masks.shape != (len(names),) + gt.shape:
        raise ValueError('Invalid GT/support')
    clipped = np.clip(raw, 0, 1)
    error = np.abs(raw - gt)
    bg = gt <= .05
    k = max(1, int(np.ceil(len(gt) * .01)))
    # Global union top-1% POINT overlap, not an impossible per-object global ranking.
    true_top = np.argsort(-gt.max(axis=1), kind='stable')[:k]
    pred_score = raw.max(axis=1)
    pred_top = np.argsort(-pred_score, kind='stable')[:k]
    overlap = float(np.intersect1d(true_top, pred_top).size / k) if np.ptp(pred_score) > 1e-12 else 0.
    instances = []
    for name, mask in zip(names, masks):
        mask = mask.astype(bool)
        if not mask.any():
            raise ValueError('Empty instance active support: ' + name)
        if np.any(gt[mask] < .3 - 1e-6):
            raise ValueError('Instance support is inconsistent with union GT: ' + name)
        instances.append(dict(id=name, active_mae_raw=float(error[mask].mean()),
            active_hit_rate_at_0_3=float((raw[mask] >= .3).mean()),
            soft_recall=float((np.minimum(clipped[mask], gt[mask]) / gt[mask]).mean())))
    return dict(mae_raw=float(error.mean()), mae_clipped=float(np.abs(clipped-gt).mean()),
                channel_mae_raw=error.mean(axis=0).tolist(),
                union_top1pct_point_overlap=overlap,
                background_positive_mean=float(np.maximum(raw[bg], 0).mean()) if bg.any() else None,
                background_positive_max=float(np.maximum(raw[bg], 0).max()) if bg.any() else None,
                out_of_range_fraction=float(((raw < 0) | (raw > 1)).mean()),
                raw_min=float(raw.min()), raw_max=float(raw.max()), instances=instances)


def case_metrics(arrays):
    names = arrays['motion_ids'].tolist()
    result = {key: metrics(arrays[key], arrays['gt'], arrays['active_masks'], names)
              for key in ('base_raw', 'adapted_raw')}
    result['zero_baseline'] = metrics(np.zeros_like(arrays['gt']), arrays['gt'], arrays['active_masks'], names)
    result['adapted_minus_base_mae_raw'] = result['adapted_raw']['mae_raw'] - result['base_raw']['mae_raw']
    result['adapted_beats_zero_mae_raw'] = result['adapted_raw']['mae_raw'] < result['zero_baseline']['mae_raw']
    return result


def load_case(path):
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    expected = {'points', 'gt', 'base_raw', 'adapted_raw', 'active_masks', 'motion_ids'}
    if set(arrays) != expected or arrays['points'].shape != (8192, 6):
        raise ValueError('Unexpected saved case arrays')
    if not np.isfinite(arrays['points']).all() or arrays['gt'].shape != (8192, 6):
        raise ValueError('Invalid saved geometry/GT')
    case_metrics(arrays)
    return arrays


def aggregate_rows(rows):
    result = {}
    for action in ('sit', 'lie', 'write_board'):
        groups = {}
        for row in rows:
            if row['case']['action'] == action:
                groups.setdefault(row['case']['scene_id'], []).append(row['metrics'])
        if not groups: raise ValueError('Missing action in evaluation: ' + action)
        result[action] = dict(scenes=len(groups))
        for branch in ('base_raw', 'adapted_raw', 'zero_baseline'):
            result[action][branch + '_mae'] = float(np.mean([
                np.mean([r[branch]['mae_raw'] for r in values]) for values in groups.values()]))
    return result


def prompt_consistency(rows, case_root):
    from itertools import combinations
    groups = {}
    for row in rows:
        c = row['case']
        groups.setdefault((c['scene_id'], c['action'], c['generation']), []).append(row)
    records = []
    for key, group in sorted(groups.items()):
        for left, right in combinations(group, 2):
            a = load_case(case_root / left['case']['filename'])
            b = load_case(case_root / right['case']['filename'])
            if not np.array_equal(a['gt'], b['gt']) or left['seeds'] != right['seeds']:
                raise ValueError('Prompt invariance pair GT/seeds differ')
            active = a['gt'] >= .3
            values = dict(scene_id=key[0], action=key[1], generation=key[2],
                          prompts=[left['case']['sample_id'], right['case']['sample_id']])
            for branch in ('base_raw', 'adapted_raw'):
                values[branch + '_active_mae_between_prompts'] = float(np.abs(a[branch]-b[branch])[active].mean())
            records.append(values)
    return records


def validate_report(summary):
    summary = Path(summary).resolve()
    report = json.loads(summary.read_text())
    if report.get('schema') != SCHEMA or report.get('status') != 'EVALUATION_COMPLETE_NOT_TEACHER_APPROVAL':
        raise ValueError('Not a completed free-rollout evaluation')
    if report.get('teacher_checkpoint_authorized') is not False:
        raise ValueError('Unexpected Teacher promotion')
    manifest_path = summary.parent / 'manifest.json'
    if sha(manifest_path) != report['manifest_sha256']:
        raise ValueError('Evaluation manifest changed')
    manifest = json.loads(manifest_path.read_text())
    cases = manifest['cases']
    if manifest.get('mode') not in ('preview', 'full'):
        raise ValueError('Unknown evaluation scope')
    expected = 16 if manifest['mode'] == 'preview' else 240
    if len(cases) != expected or len({r['filename'] for r in cases}) != expected:
        raise ValueError('Incomplete or duplicated expected evaluation panel')
    if [r['case'] for r in report['rows']] != manifest['cases']:
        raise ValueError('Incomplete or reordered panel')
    for row in report['rows']:
        path = inside(summary.parent, 'cases/' + row['case']['filename'])
        if sha(path) != row['sha256']:
            raise ValueError('Saved map hash mismatch: ' + path.name)
        actual = case_metrics(load_case(path))
        if actual != row['metrics']:
            raise ValueError('Saved metrics do not match arrays')
        if row['seeds'] != seeds(row['case']):
            raise ValueError('Case seeds changed')
    if report.get('total_cases') != len(cases) or report.get('scope') != manifest['scope']:
        raise ValueError('Report scope/count changed')
    if report.get('by_action') != aggregate_rows(report['rows']):
        raise ValueError('Aggregate metrics changed')
    if report.get('prompt_consistency') != prompt_consistency(report['rows'], summary.parent/'cases'):
        raise ValueError('Prompt consistency metrics changed')
    return report, manifest
