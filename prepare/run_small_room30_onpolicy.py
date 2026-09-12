#!/usr/bin/env python3
"""Verify real failed-state gradients, then separately run a 60-update pilot."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from small_room30_adm_dataset import SmallRoom30ADM, sha
from small_room30_training_contract import SCHEMA, schedule, write_json
from small_room30_onpolicy import POLICY, source_contract, timestep_plan, check_ready


def load_runtime(data, source, bound, device):
    from small_room30_checkpoint_io import load_checkpoint
    from small_room30_onpolicy_runtime import OnPolicyRuntime
    from small_room30_training_runtime import restore_adapter, state_digest, adapter_state
    saved = load_checkpoint(source['last_checkpoint']['path'])
    if (saved.get('schema') != SCHEMA or saved.get('purpose') != 'RESEARCH_COVERAGE_TRAINING'
            or saved.get('binding') != bound['parent_binding'] or saved.get('step') != 1500
            or saved.get('seed') != source['seed'] or saved.get('teacher_checkpoint_authorized') is not False):
        raise ValueError('Source adapter contract mismatch')
    runtime = OnPolicyRuntime(data, bound['original_binding']['inputs'], device, source['seed'])
    if runtime.base_digest != saved['base_digest'] or runtime.encoder_weight != saved['encoder_weight']:
        raise ValueError('Frozen base/encoders differ from source')
    runtime.install()
    restore_adapter(runtime.model, saved['adapter'])
    if state_digest(adapter_state(runtime.model)) != state_digest(saved['adapter']):
        raise ValueError('Source restore differs')
    return runtime


def diagnostic_gradient_check(runtime, diagnostic, diagnostic_path):
    import torch
    from fewshot_cdm_lora import lora_named_parameters
    from small_room30_coverage import surface_support
    from small_room30_coverage_runtime import coverage_terms
    from small_room30_training_runtime import state_digest, adapter_state, frozen_digest
    before = (state_digest(adapter_state(runtime.model)), frozen_digest(runtime.model))
    params = list(lora_named_parameters(runtime.model).values())
    results = []
    for row in diagnostic['rows']:
        case = row['case']
        with np.load(diagnostic_path.parent/'cases'/case['filename'], allow_pickle=False) as z:
            reference = z['free_clean_t490'].copy()
        prediction, target = runtime.free_prediction(case['index'], 490, *row['seeds'])
        physical = prediction * runtime.std_t + runtime.mean_t
        actual = physical[0].detach().cpu().numpy()
        if not np.allclose(actual, reference, atol=1e-5, rtol=1e-5):
            raise ValueError('t490 differs from the diagnosed rollout: ' + case['scene_id'])
        masks, names, _ = surface_support(runtime.data, case['index'])
        extra = coverage_terms(physical, target * runtime.std_t + runtime.mean_t,
                              torch.as_tensor(masks, device=runtime.device, dtype=torch.bool))
        correction = extra['surface_mean'] + extra['surface_worst']
        expected_failed = not next(s for s in row['metrics']['free_clean_estimates']
                                   if s['timestep'] == 490)['surface']['all_furniture_present']
        gradient_norm = None
        if expected_failed:
            gradients = torch.autograd.grad(correction, params, allow_unused=True)
            used = [g for g in gradients if g is not None]
            if not used or not all(torch.isfinite(g).all() for g in used):
                raise ValueError('Missing/nonfinite corrective gradient')
            gradient_norm = sum(float(g.detach().double().square().sum()) for g in used) ** .5
            if float(correction.detach()) <= 0 or gradient_norm <= 0:
                raise ValueError('Failed furniture has no corrective loss/gradient')
        results.append(dict(scene=case['scene_id'], timestep=490, max_abs_error=float(np.abs(actual-reference).max()),
            failed_in_reference=expected_failed, surface_loss=float(correction.detach()),
            corrective_gradient_norm=gradient_norm))
        print('[SOURCE CHECK] ' + json.dumps(results[-1]), flush=True)
    if sum(r['failed_in_reference'] for r in results) != 4:
        raise ValueError('Expected the four diagnosed failures, not another source result')
    if before != (state_digest(adapter_state(runtime.model)), frozen_digest(runtime.model)):
        raise ValueError('Diagnostic check mutated model')
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['smoke', 'train'], required=True)
    p.add_argument('--dataset', type=Path, default=Path('data/small_room30_adm_lora_v1'))
    p.add_argument('--source-summary', type=Path, default=Path('outputs/small_room30_coverage_run01/summary.json'))
    p.add_argument('--diagnostic-summary', type=Path, default=Path('outputs/small_room30_generation_diagnostic01/summary.json'))
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--readiness', type=Path)
    p.add_argument('--updates', type=int, default=60)
    p.add_argument('--allow-research-training', action='store_true')
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    if a.mode == 'train' and (not a.readiness or not a.allow_research_training or not 1 <= a.updates <= 60):
        p.error('Train requires readiness, research flag, and 1..60 updates')
    if a.mode == 'smoke' and (a.readiness or a.allow_research_training): p.error('Training flags in smoke')
    if Path.cwd().resolve() != REPO: raise ValueError('Run from AMDM root')
    a.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        data = SmallRoom30ADM(a.dataset)
        source, diag, bound = source_contract(data, a.source_summary, a.diagnostic_summary, REPO)
        common = dict(schema=SCHEMA, binding=bound, seed=source['seed'],
            source_summary=str(a.source_summary.resolve()), diagnostic_summary=str(a.diagnostic_summary.resolve()),
            teacher_checkpoint_authorized=False)
        import torch
        from fewshot_cdm_lora import lora_named_parameters
        from small_room30_training_runtime import frozen_digest
        from run_small_room30_teacher import save_state
        if a.mode == 'train':
            ready = json.loads(a.readiness.read_text())
            check_ready(ready, bound)
            if ready['torch_version'] != str(torch.__version__) or ready['cuda_version'] != torch.version.cuda:
                raise ValueError('Runtime changed after readiness')
        runtime = load_runtime(data, source, bound, a.device)
        count = 3 if a.mode == 'smoke' else a.updates
        if a.mode == 'smoke':
            results = diagnostic_gradient_check(runtime, diag, a.diagnostic_summary)
            write_json(a.output_dir/'source_gradient_check.json', results)
            with torch.no_grad():
                for i in range(len(data)):
                    pred, target = runtime.predict(i, (0, 250, 499)[i % 3], runtime.seed + 70000 + i)
                    loss, _ = runtime.objective(pred, target, i)
                    if not torch.isfinite(loss): raise ValueError('Nonfinite replay forward')
                    print('[REPLAY CHECK %d/80] finite' % (i+1), flush=True)
        optimizer = torch.optim.AdamW(list(lora_named_parameters(runtime.model).values()), lr=POLICY['lr'], weight_decay=0.)
        # New 60-update schedule covers all 30 Sit groups twice, not only the six pictured rooms.
        updates = schedule(data, count, runtime.seed + 300001)
        timesteps = [490, 499, 0] if a.mode == 'smoke' else timestep_plan(count, runtime.seed)
        write_json(a.output_dir/'plan.json', dict(timesteps=timesteps,
            samples=[[data.samples[i]['id'] for i in group] for group in updates], policy=POLICY))
        checkpoints = []
        with (a.output_dir/'train.jsonl').open('x') as log:
            for number, (indices, t) in enumerate(zip(updates, timesteps), 1):
                print('[ONPOLICY %d/%d] collecting fresh x_%d; %d prefix steps' % (number,count,t,499-t), flush=True)
                row = runtime.update_onpolicy(optimizer, indices, 1500+number, t)
                log.write(json.dumps(row, allow_nan=False)+'\n'); log.flush()
                print('[UPDATE] '+json.dumps(row), flush=True)
                if number == count or number % 20 == 0:
                    if frozen_digest(runtime.model) != runtime.frozen_after_install: raise ValueError('Frozen state changed')
                    with torch.no_grad(): before = runtime.predict(0,499,runtime.seed+991)[0].cpu()
                    name = 'SMOKE_ONLY.pt' if a.mode == 'smoke' else 'onpolicy_adapter_step_%06d.pt' % (1500+number)
                    path = a.output_dir/name
                    purpose = 'DISPOSABLE_ONPOLICY_SMOKE' if a.mode == 'smoke' else 'RESEARCH_ONPOLICY_PILOT'
                    digest = save_state(path,runtime,optimizer,bound,1500+number,runtime.seed,purpose)
                    with torch.no_grad(): after = runtime.predict(0,499,runtime.seed+991)[0].cpu()
                    if not torch.equal(before,after): raise ValueError('Checkpoint roundtrip changes prediction')
                    checkpoints.append(dict(path=str(path.resolve()), sha256=digest, step=1500+number))
        if sha(source['last_checkpoint']['path']) != source['last_checkpoint']['sha256']:
            raise ValueError('Source checkpoint changed')
        if a.mode == 'smoke':
            result = dict(common,status='ONPOLICY_RUNTIME_READY',main_training_started=False,
                checks={k:True for k in ['source_restored','diagnostic_t490_reproduced','failed_target_coverage_gradients',
                    'all_80_forward_finite','updates_finite_nonzero','frozen_unchanged','disk_roundtrip']},
                torch_version=str(torch.__version__),cuda_version=torch.version.cuda,smoke_checkpoint=checkpoints[-1])
        else:
            result = dict(common,status='ONPOLICY_PILOT_FINISHED_NOT_EVALUATED',completed_step=1500+count,
                extra_steps=count,last_checkpoint=checkpoints[-1],checkpoints=checkpoints,training_started=True,
                readiness_sha256=sha(a.readiness),source_checkpoint_preserved=True,performance_tested=False)
        write_json(a.output_dir/'summary.json', result)
        print('[%s] %s/summary.json' % (result['status'],a.output_dir),flush=True)
        print('[NOTICE] Execution success is NOT furniture coverage success.',flush=True)
    except Exception as exc:
        write_json(a.output_dir/'failure.json',dict(status='FAILED',error=str(exc),teacher_checkpoint_authorized=False))
        raise


if __name__ == '__main__': main()
