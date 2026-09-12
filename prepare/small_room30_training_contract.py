"""CPU-only, versioned preparation contract. Never certifies Teacher quality."""
import json
import random
from pathlib import Path

from small_room30_adm_dataset import sha, inside

SCHEMA = 'small_room30_training_preparation_v1'
POLICY = dict(diffusion_steps=500, rank=16, alpha=16., lr=0.0001,
              grad_clip=1., active_threshold=0.30, background_threshold=0.05,
              loss_weights=dict(normalized_mse=1., instance_active=2., background=1., range=0.1),
              initialization='original ADM + frozen v5 base + fresh zero-output LoRA',
              teacher_checkpoint_authorized=False,
              split='development only; no held-out/generalization claim',
              sampling='one scene/action group per action per update; rotate prompt variants',
              forward_keys=['c_pc_xyz', 'c_pc_feat', 'c_text'])


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def groups_for(data):
    groups = {}
    for i, row in enumerate(data.samples):
        groups.setdefault((row['scene_id'], row['action']), []).append(i)
    if len(groups) != 45:
        raise ValueError('Expected 45 distinct scene/action targets')
    return groups


def schedule(data, steps, seed):
    """Balance three actions without counting prompt aliases as new GT maps."""
    if steps < 1:
        raise ValueError('steps must be positive')
    groups = groups_for(data)
    rng = random.Random(seed)
    pools = {a: sorted(k for k in groups if k[1] == a)
             for a in ('sit', 'lie', 'write_board')}
    orders = {a: [] for a in pools}
    visits = {k: 0 for k in groups}
    result = []
    for _ in range(steps):
        update = []
        for action in pools:
            if not orders[action]:
                orders[action] = pools[action].copy()
                rng.shuffle(orders[action])
            key = orders[action].pop()
            rows = groups[key]
            update.append(rows[visits[key] % len(rows)])
            visits[key] += 1
        result.append(update)
    return result


def resolve_inputs(summary, original=None, base=None, stats=None):
    """Read provenance only; do NOT load the research-only v111 adapter."""
    values = dict(original_checkpoint=original, v5_checkpoint=base, stats_file=stats)
    source = None
    if summary:
        source = json.loads(Path(summary).read_text())
    for name, raw in values.items():
        if raw is None:
            if source is None:
                raise ValueError('Supply --source-summary or all three checkpoint/stats paths')
            raw = source['paths'][name]
        path = Path(raw).resolve()
        if not path.is_file():
            raise FileNotFoundError(str(path))
        digest = sha(path)
        if source and name in source.get('paths', {}) and path == Path(source['paths'][name]).resolve():
            expected = source.get('path_sha256', {}).get(name)
            if expected is None or expected != digest:
                raise ValueError('Source checkpoint/stat hash mismatch: ' + name)
        values[name] = dict(path=str(path), sha256=digest)
    return values


def binding(data, inputs, repo):
    """Bind report to data, inputs, executable code and model/diffusion config."""
    for info in inputs.values():
        if sha(info['path']) != info['sha256']:
            raise ValueError('Bound input changed: ' + info['path'])
    names = ['prepare/small_room30_adm_dataset.py', 'prepare/validate_small_room30_adm_export.py',
             'prepare/small_room30_checkpoint_io.py',
             'prepare/small_room30_training_contract.py', 'prepare/small_room30_training_runtime.py',
             'prepare/run_small_room30_teacher.py', 'prepare/fewshot_cdm_lora.py',
             'prepare/relational_teacher_v9_lora_runtime.py']
    for directory in ('models', 'diffusion', 'configs', 'utils'):
        names.extend(str(p.relative_to(repo)) for p in (repo / directory).rglob('*')
                     if p.is_file() and p.suffix in ('.py', '.yaml', '.yml'))
    return dict(schema=SCHEMA, policy=POLICY, inputs=inputs,
                dataset_index_sha256=sha(data.root / 'index.json'),
                dataset_checksums_sha256=sha(data.root / 'SHA256SUMS.txt'),
                code_sha256={n: sha(inside(repo, n)) for n in sorted(set(names))})


def require_readiness(report, current):
    if report.get('status') != 'RUNTIME_READINESS_PASS' or report.get('binding') != current:
        raise ValueError('Missing/stale runtime readiness; rerun smoke with these exact inputs/code')
    checks = ('all_80_forward_finite', 'zero_lora_equivalence', 'nonzero_finite_gradients',
              'lora_updated', 'frozen_state_unchanged', 'adapter_reload_exact', 'output_reload_close')
    if not all(report.get('checks', {}).get(k) is True for k in checks):
        raise ValueError('Incomplete runtime readiness checks')
    if report.get('teacher_checkpoint_authorized') is not False:
        raise ValueError('Readiness must not authorize a Teacher checkpoint')
