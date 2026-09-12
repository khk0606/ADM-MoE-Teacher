#!/usr/bin/env python3
"""Audit -> disposable warm-start smoke -> 300 extra updates, in separate directories."""
import argparse
import json
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from small_room30_adm_dataset import SmallRoom30ADM, sha
from small_room30_training_contract import write_json, schedule, SCHEMA
from small_room30_coverage import (POLICY, audit_data, original_source, experiment_binding,
                                   validate_checkpoint)


def load_runtime(data, source, device):
    from small_room30_checkpoint_io import load_checkpoint
    from small_room30_training_runtime import restore_adapter, adapter_state, state_digest
    from small_room30_coverage_runtime import CoverageRuntime
    saved = load_checkpoint(source['last_checkpoint']['path'])
    validate_checkpoint(saved, source)
    runtime = CoverageRuntime(data, source['binding']['inputs'], device, source['seed'])
    if runtime.base_digest != saved['base_digest'] or runtime.encoder_weight != saved['encoder_weight']:
        raise ValueError('Actual frozen base/encoder differs from source checkpoint')
    runtime.install()
    restore_adapter(runtime.model, saved['adapter'])
    if state_digest(adapter_state(runtime.model)) != state_digest(saved['adapter']):
        raise ValueError('Warm-start adapter differs')
    print('[WARM START] exact source adapter restored; original optimizer NOT restored', flush=True)
    return runtime


def check_ready(ready, current):
    checks = ('warm_start_verified', 'all_80_forward_finite', 'nonzero_updates',
              'frozen_unchanged', 'roundtrip_verified')
    if (ready.get('status') != 'COVERAGE_RUNTIME_READY' or ready.get('binding') != current
        or ready.get('teacher_checkpoint_authorized') is not False
        or not all(ready.get('checks', {}).get(k) is True for k in checks)):
        raise ValueError('Missing/stale coverage readiness; run prepare_coverage.sh first')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['audit', 'smoke', 'train'], default='audit')
    p.add_argument('--dataset', type=Path, default=Path('data/small_room30_adm_lora_v1'))
    p.add_argument('--source-summary', type=Path, default=Path('outputs/small_room30_research_run01/summary.json'))
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--readiness', type=Path)
    p.add_argument('--extra-steps', type=int, default=300)
    p.add_argument('--allow-research-training', action='store_true')
    a = p.parse_args()
    if a.mode == 'train' and (not a.readiness or not a.allow_research_training or not 1 <= a.extra_steps <= 300):
        p.error('train needs readiness, explicit research flag and 1..300 extra steps')
    if a.mode != 'train' and (a.allow_research_training or a.readiness):
        p.error('Training flags are invalid in audit/smoke')
    if Path.cwd().resolve() != REPO: raise ValueError('Run from AMDM root')
    a.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        data = SmallRoom30ADM(a.dataset)
        source = original_source(data, a.source_summary, REPO)
        current = experiment_binding(source, a.source_summary, REPO)
        print('[AUDIT] validating source hashes, all union targets and furniture surfaces', flush=True)
        audit = audit_data(data)
        write_json(a.output_dir / 'data_audit.json', audit)
        common = dict(schema=SCHEMA, binding=current, seed=source['seed'],
            source_summary=str(a.source_summary.resolve()), coverage_experiment=POLICY['version'],
            teacher_checkpoint_authorized=False)
        if a.mode == 'audit':
            write_json(a.output_dir / 'summary.json', dict(common, status=audit['status']))
            print('[AUDIT PASS] 65 Sit prompts each contain BOTH target furniture surfaces', flush=True)
            return
        import torch
        from fewshot_cdm_lora import lora_named_parameters
        from small_room30_training_runtime import frozen_digest
        from run_small_room30_teacher import save_state
        ready = None
        if a.mode == 'train':
            ready = json.loads(a.readiness.read_text())
            check_ready(ready, current)
            if ready['torch_version'] != str(torch.__version__) or ready['cuda_version'] != torch.version.cuda:
                raise ValueError('Runtime version changed after coverage smoke')
        runtime = load_runtime(data, source, a.device)
        optimizer = torch.optim.AdamW(list(lora_named_parameters(runtime.model).values()),
                                     lr=POLICY['lr'], weight_decay=0.)
        start = int(source['completed_step'])
        if a.mode == 'smoke':
            with torch.no_grad():
                for i in range(len(data)):
                    pred, target = runtime.predict(i, (0, 250, 499)[i % 3], source['seed'] + 70000 + i)
                    loss, parts = runtime.objective(pred, target, i)
                    if not torch.isfinite(loss): raise ValueError('Nonfinite coverage forward')
                    print('[COVERAGE FORWARD %d/80] loss=%.6f' % (i + 1, float(loss)), flush=True)
            count = 3
        else:
            count = a.extra_steps
        updates = schedule(data, start + count, source['seed'])
        latest = None
        with (a.output_dir / 'train.jsonl').open('x') as log:
            for step in range(start + 1, start + count + 1):
                row = runtime.update(optimizer, updates[step - 1], step)
                log.write(json.dumps(row, allow_nan=False) + '\n'); log.flush()
                print('[COVERAGE %d/%d] %s' % (step - start, count, json.dumps(row)), flush=True)
                if step == start + count or (a.mode == 'train' and (step - start) % 100 == 0):
                    if frozen_digest(runtime.model) != runtime.frozen_after_install:
                        raise ValueError('Frozen parameters/buffers changed')
                    with torch.no_grad():
                        before = runtime.predict(0, 499, source['seed'] + 991)[0].cpu()
                    purpose = 'DISPOSABLE_COVERAGE_SMOKE' if a.mode == 'smoke' else 'RESEARCH_COVERAGE_TRAINING'
                    path = a.output_dir / ('SMOKE_ONLY.pt' if a.mode == 'smoke' else 'coverage_adapter_step_%06d.pt' % step)
                    digest = save_state(path, runtime, optimizer, current, step, source['seed'], purpose)
                    with torch.no_grad():
                        after = runtime.predict(0, 499, source['seed'] + 991)[0].cpu()
                    if not torch.allclose(before, after, atol=1e-6, rtol=1e-6):
                        raise ValueError('Prediction differs after disk reload')
                    latest = dict(path=str(path.resolve()), sha256=digest, step=step)
        if a.mode == 'smoke':
            report = dict(common, status='COVERAGE_RUNTIME_READY',
                checks=dict(warm_start_verified=True, all_80_forward_finite=True, nonzero_updates=True,
                            frozen_unchanged=True, roundtrip_verified=True),
                torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                main_training_started=False, smoke_checkpoint=latest)
        else:
            report = dict(common, status='RESEARCH_TRAINING_FINISHED_NOT_EVALUATED',
                completed_step=start + count, extra_steps=count, last_checkpoint=latest,
                readiness_sha256=sha(a.readiness), training_started=True, performance_tested=False,
                optimizer_reset=True, source_checkpoint_preserved=True)
        if sha(source['last_checkpoint']['path']) != source['last_checkpoint']['sha256']:
            raise ValueError('Source checkpoint changed during experiment')
        write_json(a.output_dir / 'summary.json', report)
        print('[%s] %s/summary.json' % (report['status'], a.output_dir), flush=True)
        print('[NOTICE] No furniture coverage/performance pass from training losses.', flush=True)
    except Exception as exc:
        write_json(a.output_dir / 'failure.json', dict(status='FAILED', error=str(exc), teacher_checkpoint_authorized=False))
        raise


if __name__ == '__main__': main()
