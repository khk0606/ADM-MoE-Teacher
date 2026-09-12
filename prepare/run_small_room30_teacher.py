#!/usr/bin/env python3
"""Plan -> disposable runtime smoke -> explicitly authorized research training.

This runner never publishes or promotes a Teacher checkpoint. Performance testing
must use free-noise reverse diffusion, not its GT-noised readiness forwards.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from small_room30_adm_dataset import SmallRoom30ADM, sha, inside
from small_room30_training_contract import (SCHEMA, POLICY, binding, groups_for,
    resolve_inputs, schedule, write_json, require_readiness)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['plan', 'smoke', 'train'], default='plan')
    p.add_argument('--dataset', type=Path, default=Path('data/small_room30_adm_lora_v1'))
    p.add_argument('--source-summary', type=Path)
    p.add_argument('--original-checkpoint', type=Path)
    p.add_argument('--base-checkpoint', type=Path)
    p.add_argument('--stats-file', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=20260908)
    p.add_argument('--readiness', type=Path)
    p.add_argument('--steps', type=int)
    p.add_argument('--resume', type=Path)
    p.add_argument('--allow-research-training', action='store_true')
    a = p.parse_args()
    if a.mode == 'train' and (not a.allow_research_training or not a.readiness or not a.steps or a.steps < 1):
        p.error('train requires --allow-research-training --readiness FILE --steps POSITIVE_INTEGER')
    if a.mode != 'train' and (a.steps is not None or a.resume or a.allow_research_training):
        p.error('Training flags cannot be used for plan/smoke')
    return a


def plan(data, inputs, current, seed):
    from validate_small_room30_adm_export import validate
    validation = validate(data.root, inputs['stats_file']['path'], allow_research=True)
    # Check ALL support files before consuming GPU time; do not load relations as features.
    import numpy as np
    source_file = inside(data.root, data.index['source_contact_index'])
    motions = {m['id']: m for m in json.loads(source_file.read_text())['motions']}
    for mid, motion in motions.items():
        with np.load(inside(source_file.parent, motion['file']), allow_pickle=False) as z:
            if not np.any(z['affordance'] >= POLICY['active_threshold']):
                raise ValueError('Empty action support: ' + mid)
    return dict(schema=SCHEMA, status='DATA_AND_TRAINING_PLAN_PASS', binding=current,
                validation=validation, scene_action_groups=len(groups_for(data)),
                first_3_updates=[[data.samples[i]['id'] for i in ids] for ids in schedule(data, 3, seed)],
                seed=seed, teacher_checkpoint_authorized=False,
                performance_tested=False, training_started=False)


def save_state(path, runtime, optimizer, current, step, seed, purpose):
    import torch
    from small_room30_checkpoint_io import load_checkpoint
    from small_room30_training_runtime import adapter_state, restore_adapter, state_digest
    state = adapter_state(runtime.model)
    checkpoint = dict(schema=SCHEMA, purpose=purpose, teacher_checkpoint_authorized=False,
                      binding=current, step=step, seed=seed, adapter=state,
                      optimizer=optimizer.state_dict(), base_digest=runtime.base_digest,
                      encoder_weight=runtime.encoder_weight)
    temp = path.with_suffix('.tmp')
    torch.save(checkpoint, temp)
    saved = load_checkpoint(temp)
    if state_digest(saved['adapter']) != state_digest(state):
        raise RuntimeError('Serialized adapter tensor mismatch')
    # Poison first, then reload: verifies that the restore actually takes effect.
    from fewshot_cdm_lora import lora_named_parameters
    with torch.no_grad():
        for p in lora_named_parameters(runtime.model).values():
            p.zero_()
    restore_adapter(runtime.model, saved['adapter'])
    if state_digest(adapter_state(runtime.model)) != state_digest(state):
        raise RuntimeError('Adapter restore mismatch')
    temp.replace(path)
    return sha(path)


def smoke(data, inputs, current, args):
    import torch
    from small_room30_training_runtime import Runtime, frozen_digest
    runtime = Runtime(data, inputs, args.device, args.seed)
    representatives = [next(i for i, r in enumerate(data.samples) if r['action'] == a)
                       for a in ('sit', 'lie', 'write_board')]
    # Real model zero-LoRA equivalence, for all action types and noise levels.
    before = {}
    with torch.no_grad():
        for i in representatives:
            for t in (0, 250, 499):
                before[i, t] = runtime.predict(i, t, args.seed + i + t)[0].cpu()
    runtime.install()
    checks = {}
    with torch.no_grad():
        for (i, t), value in before.items():
            actual = runtime.predict(i, t, args.seed + i + t)[0].cpu()
            if not torch.equal(value, actual):
                raise RuntimeError('Fresh zero-output LoRA changed the frozen Base forward')
        checks['zero_lora_equivalence'] = True
        for i in range(len(data)):
            prediction, target = runtime.predict(i, (0, 250, 499)[i % 3], args.seed + 1000 + i)
            loss, _ = runtime.objective(prediction, target, i)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite objective for ' + data.samples[i]['id'])
            print(f'[FORWARD {i+1}/80] {data.samples[i]["id"]} finite loss={float(loss):.6f}', flush=True)
        checks['all_80_forward_finite'] = True
    from fewshot_cdm_lora import lora_named_parameters
    optimizer = torch.optim.AdamW(list(lora_named_parameters(runtime.model).values()), lr=POLICY['lr'], weight_decay=0.)
    logs = []
    for step, indices in enumerate(schedule(data, 3, args.seed), 1):
        row = runtime.update(optimizer, indices, step)
        logs.append(row)
        print(f'[SMOKE {step}/3] grad={row["gradient_norm"]:.6f}; temporary adapter only', flush=True)
    checks['nonzero_finite_gradients'] = checks['lora_updated'] = True
    if frozen_digest(runtime.model) != runtime.frozen_after_install:
        raise RuntimeError('Frozen base/encoder parameters or buffers changed')
    checks['frozen_state_unchanged'] = True
    with torch.no_grad():
        reference = runtime.predict(representatives[0], 250, args.seed + 999)[0].cpu()
    cp = args.output_dir / 'SMOKE_ONLY_adapter.pt'
    digest = save_state(cp, runtime, optimizer, current, 3, args.seed, 'DISPOSABLE_RUNTIME_SMOKE')
    with torch.no_grad():
        reloaded = runtime.predict(representatives[0], 250, args.seed + 999)[0].cpu()
    if not torch.allclose(reference, reloaded, atol=1e-6, rtol=1e-6):
        raise RuntimeError('Output differs after adapter disk roundtrip')
    checks['adapter_reload_exact'] = checks['output_reload_close'] = True
    return dict(schema=SCHEMA, status='RUNTIME_READINESS_PASS', binding=current, checks=checks,
                seed=args.seed, base_digest=runtime.base_digest, encoder_weight=runtime.encoder_weight,
                checkpoint_loads=runtime.loads, torch_version=str(torch.__version__),
                cuda_version=torch.version.cuda, device=torch.cuda.get_device_name(args.device),
                diagnostic_updates=3, update_log=logs, smoke_checkpoint_sha256=digest,
                teacher_checkpoint_authorized=False, performance_tested=False,
                training_started=False, explanation='Only disposable smoke updates; no main training or quality approval')


def train(data, inputs, current, args):
    import torch
    from small_room30_checkpoint_io import load_checkpoint
    from small_room30_training_runtime import Runtime, frozen_digest, restore_adapter
    from fewshot_cdm_lora import lora_named_parameters
    readiness = json.loads(args.readiness.read_text())
    require_readiness(readiness, current)
    if readiness['seed'] != args.seed:
        raise ValueError('Use the same seed as the runtime readiness report')
    if (readiness.get('torch_version') != str(torch.__version__)
            or readiness.get('cuda_version') != torch.version.cuda):
        raise ValueError('Torch/CUDA version changed; rerun runtime readiness')
    runtime = Runtime(data, inputs, args.device, args.seed)
    if runtime.base_digest != readiness['base_digest'] or runtime.encoder_weight != readiness['encoder_weight']:
        raise ValueError('Actual base/encoder state differs from readiness')
    runtime.install()  # Always fresh; the smoke adapter is NEVER the warm start.
    optimizer = torch.optim.AdamW(list(lora_named_parameters(runtime.model).values()), lr=POLICY['lr'], weight_decay=0.)
    start = 0
    if args.resume:
        saved = load_checkpoint(args.resume)
        if (saved.get('schema') != SCHEMA or saved.get('purpose') != 'RESEARCH_TRAINING'
                or saved.get('binding') != current or saved.get('seed') != args.seed
                or saved.get('base_digest') != runtime.base_digest
                or saved.get('teacher_checkpoint_authorized') is not False):
            raise ValueError('Resume checkpoint contract mismatch (smoke checkpoints prohibited)')
        restore_adapter(runtime.model, saved['adapter'])
        optimizer.load_state_dict(saved['optimizer'])
        start = int(saved['step'])
        if not 0 < start < args.steps:
            raise ValueError('--steps must exceed the resumed step')
    updates = schedule(data, args.steps, args.seed)
    latest = None
    with (args.output_dir / 'train.jsonl').open('x') as log:
        for step in range(start + 1, args.steps + 1):
            row = runtime.update(optimizer, updates[step - 1], step)
            log.write(json.dumps(row, allow_nan=False) + '\n'); log.flush()
            print(f'[TRAIN {step}/{args.steps}] grad={row["gradient_norm"]:.6f} losses={row["loss"]}', flush=True)
            if step % 100 == 0 or step == args.steps:
                if frozen_digest(runtime.model) != runtime.frozen_after_install:
                    raise RuntimeError('Frozen base/encoders changed during training')
                path = args.output_dir / f'research_adapter_step_{step:06d}.pt'
                digest = save_state(path, runtime, optimizer, current, step, args.seed, 'RESEARCH_TRAINING')
                latest = dict(path=str(path.resolve()), sha256=digest, step=step)
    return dict(schema=SCHEMA, status='RESEARCH_TRAINING_FINISHED_NOT_EVALUATED', binding=current,
                seed=args.seed, last_checkpoint=latest, training_started=True,
                completed_step=args.steps, teacher_checkpoint_authorized=False, performance_tested=False,
                note='GT-noised losses are not free-generation quality. No approved Teacher was produced.')


def main():
    args = parse_args()
    if Path.cwd().resolve() != REPO:
        raise ValueError('Run from the AMDM repository root (pretrained encoder paths are relative)')
    args.output_dir.mkdir(parents=True, exist_ok=False)  # Never overwrite existing runs.
    try:
        data = SmallRoom30ADM(args.dataset)
        inputs = resolve_inputs(args.source_summary, args.original_checkpoint, args.base_checkpoint, args.stats_file)
        current = binding(data, inputs, REPO)
        print('[PREPARE] verifying all source/union/support data and pretrained statistics', flush=True)
        report = plan(data, inputs, current, args.seed)
        write_json(args.output_dir / 'plan.json', report)
        if args.mode == 'smoke':
            report = smoke(data, inputs, current, args)
        elif args.mode == 'train':
            report = train(data, inputs, current, args)
        write_json(args.output_dir / 'summary.json', report)
        print(f'[PASS] {report["status"]}', flush=True)
        print(f'[REPORT] {args.output_dir / "summary.json"}', flush=True)
        print('[NOTICE] Teacher quality has NOT been evaluated or authorized.', flush=True)
    except Exception as exc:
        write_json(args.output_dir / 'failure.json', dict(status='FAILED', mode=args.mode,
                   error_type=type(exc).__name__, error=str(exc), teacher_checkpoint_authorized=False))
        raise


if __name__ == '__main__':
    main()
