#!/usr/bin/env python3
"""Targeted pilot evaluator; unchanged sampler and Viser array format."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from small_room30_adm_dataset import SmallRoom30ADM, sha
from small_room30_training_contract import binding, write_json, SCHEMA as TRAIN_SCHEMA
from small_room30_evaluation_common import (SCHEMA, EVAL_FILES, panel, seeds, conditioning_arrays,
    supervision, case_metrics, load_case, validate_report, aggregate_rows, prompt_consistency)


def rollout(runtime, kwargs_np, initial_seed, reverse_seed, label):
    import torch
    from relational_teacher_v9_lora_runtime import deterministic_noise
    kwargs = {k: v if k == 'c_text' else torch.from_numpy(v).to(runtime.device)
              for k, v in kwargs_np.items()}
    if set(kwargs) != {'c_pc_xyz', 'c_pc_feat', 'c_text'}:
        raise ValueError('Only scene and text are permitted during inference')
    device = torch.device(runtime.device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    noise = deterministic_noise(torch.Size((1, 8192, 6)), initial_seed, runtime.device)
    runtime.model.eval()
    # Reset reverse-process RNG as well as initial noise for BOTH branches.
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        torch.manual_seed(reverse_seed)
        if device.type == 'cuda': torch.cuda.manual_seed(reverse_seed)
        count = 0
        for count, out in enumerate(runtime.diffusion.p_sample_loop_progressive(
                runtime.model, (1, 8192, 6), noise=noise, clip_denoised=False,
                model_kwargs=kwargs, device=runtime.device, progress=False), 1):
            if count % 100 == 0:
                print('[ROLLOUT %s] %d/500' % (label, count), flush=True)
        if count != 500 or out['sample'].shape != (1, 8192, 6):
            raise ValueError('Incomplete/wrong-shape reverse diffusion')
        raw = out['sample'][0].detach().cpu().numpy() * runtime.std + runtime.mean
    if not np.isfinite(raw).all(): raise FloatingPointError('Nonfinite generated affordance')
    return raw.astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-summary', type=Path, default=Path('outputs/small_room30_targeted_run01/summary.json'))
    p.add_argument('--dataset', type=Path, default=Path('data/small_room30_adm_lora_v1'))
    p.add_argument('--mode', choices=['preview', 'full'], default='preview')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if Path.cwd().resolve() != REPO: raise ValueError('Run from the AMDM repository root')
    data = SmallRoom30ADM(args.dataset)
    from small_room30_targeted import validate_run, FILES
    training, source, current = validate_run(data, args.training_summary, REPO)
    cp_info = training['last_checkpoint']
    cp_path = Path(cp_info['path'])
    import torch
    from small_room30_checkpoint_io import load_checkpoint
    saved = load_checkpoint(cp_path)
    if (saved.get('schema') != TRAIN_SCHEMA or saved.get('purpose') != 'RESEARCH_TARGETED_PILOT'
            or saved.get('binding') != current or saved.get('seed') != training['seed']
            or saved.get('step') != cp_info['step'] or saved.get('step') != training['completed_step']
            or saved.get('teacher_checkpoint_authorized') is not False):
        raise ValueError('Adapter contract mismatch')
    cases = panel(data, args.mode)
    manifest = dict(schema=SCHEMA, mode=args.mode, cases=cases,
        training_summary_sha256=sha(args.training_summary), checkpoint_sha256=sha(cp_path),
        training_binding=current, checkpoint_step=saved['step'], seed=20260908,
        evaluation_code_sha256={n: sha(REPO / n) for n in EVAL_FILES + FILES},
        torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
        device=torch.cuda.get_device_name(args.device),
        diffusion_steps=500, generation='free Gaussian noise; ancestral DDPM; no GT conditioning',
        pairing='same initial and reverse RNG seeds for Base/adapter and same-action prompt aliases',
        scope='DEVELOPMENT ONLY: 6 representative rooms, K=1' if args.mode == 'preview' else 'DEVELOPMENT ONLY: all 30 rooms, K=3',
        teacher_checkpoint_authorized=False)
    if args.output_dir.exists():
        if not args.resume: raise ValueError('Output exists; use --resume for this exact evaluation')
        if json.loads((args.output_dir / 'manifest.json').read_text()) != manifest:
            raise ValueError('Resume evaluation manifest differs')
        if (args.output_dir / 'summary.json').exists():
            validate_report(args.output_dir / 'summary.json')
            print('[COMPLETE] Existing evaluation verified; no new inference or training.', flush=True)
            return
    else:
        args.output_dir.mkdir(parents=True)
        (args.output_dir / 'cases').mkdir()
        write_json(args.output_dir / 'manifest.json', manifest)
    try:
        from small_room30_training_runtime import Runtime, restore_adapter, frozen_digest, state_digest
        from fewshot_cdm_lora import set_lora_enabled, lora_named_parameters
        runtime = Runtime(data, current['original_binding']['inputs'], args.device, training['seed'])
        if runtime.base_digest != saved['base_digest'] or runtime.encoder_weight != saved['encoder_weight']:
            raise ValueError('Actual base/encoder tensors differ from saved training checkpoint')
        runtime.install()
        restore_adapter(runtime.model, saved['adapter'])
        if state_digest(lora_named_parameters(runtime.model)) != state_digest(saved['adapter']):
            raise ValueError('Loaded adapter mismatch')
        for parameter in runtime.model.parameters(): parameter.requires_grad_(False)
        frozen_before = frozen_digest(runtime.model)
        adapter_before = state_digest(lora_named_parameters(runtime.model))
        rows = []
        print('[EVAL] %d cases x 2 models x 500 steps; no optimizer or training' % len(cases), flush=True)
        for num, case in enumerate(cases, 1):
            path = args.output_dir / 'cases' / case['filename']
            receipt_path = path.with_suffix('.json')
            if receipt_path.exists():
                row = json.loads(receipt_path.read_text())
                if row['case'] != case or sha(path) != row['sha256'] or row['metrics'] != case_metrics(load_case(path)):
                    raise ValueError('Cached case changed: ' + path.name)
                rows.append(row)
                print('[CACHE %d/%d] %s' % (num, len(cases), case['filename']), flush=True)
                continue
            points, kw = conditioning_arrays(data, case['index'])
            initial, reverse = seeds(case)
            maps = {}
            for name, enabled in (('base_raw', False), ('adapted_raw', True)):
                set_lora_enabled(runtime.model, enabled)
                maps[name] = rollout(runtime, kw, initial, reverse, '%d/%d %s %s' % (num, len(cases), case['sample_id'], name))
            gt, masks, ids = supervision(data, case['index'])
            arrays = dict(points=points, gt=gt, active_masks=masks, motion_ids=np.asarray(ids), **maps)
            scores = case_metrics(arrays)
            temp = path.with_suffix('.npz.tmp')
            with temp.open('wb') as f: np.savez_compressed(f, **arrays)
            if case_metrics(load_case(temp)) != scores: raise ValueError('Saved arrays changed metrics')
            temp.replace(path)
            row = dict(case=case, seeds=[initial, reverse], sha256=sha(path), metrics=scores)
            write_json(receipt_path, row)
            rows.append(row)
            print('[CASE %d/%d] Base MAE %.6f -> adapted %.6f; zero %.6f' % (
                num, len(cases), scores['base_raw']['mae_raw'], scores['adapted_raw']['mae_raw'], scores['zero_baseline']['mae_raw']), flush=True)
        if frozen_digest(runtime.model) != frozen_before or state_digest(lora_named_parameters(runtime.model)) != adapter_before:
            raise ValueError('Model state mutated during evaluation')
        if any(p.grad is not None for p in runtime.model.parameters()): raise ValueError('Unexpected evaluation gradients')
        report = dict(schema=SCHEMA, status='EVALUATION_COMPLETE_NOT_TEACHER_APPROVAL',
            manifest_sha256=sha(args.output_dir / 'manifest.json'), scope=manifest['scope'],
            rows=rows, by_action=aggregate_rows(rows),
            prompt_consistency=prompt_consistency(rows, args.output_dir/'cases'),
            worst_cases_by_adapted_mae=[r['case']['filename'] for r in sorted(rows, key=lambda r: r['metrics']['adapted_raw']['mae_raw'], reverse=True)[:10]],
            improved_case_count=sum(r['metrics']['adapted_minus_base_mae_raw'] < 0 for r in rows),
            total_cases=len(rows), model_state_unchanged=True, optimizer_created=False,
            teacher_checkpoint_authorized=False, note='Not held-out; no Teacher promotion from these diagnostics')
        write_json(args.output_dir / 'summary.json', report)
        validate_report(args.output_dir / 'summary.json')
        print('[EVALUATION COMPLETE] %s/summary.json' % args.output_dir, flush=True)
        print(json.dumps(report['by_action'], indent=2), flush=True)
        print('[NOTICE] Saved free-rollout evidence, NOT Teacher approval.', flush=True)
    except Exception as exc:
        write_json(args.output_dir / 'failure.json', dict(status='FAILED', error=str(exc), teacher_checkpoint_authorized=False))
        raise


if __name__ == '__main__': main()
