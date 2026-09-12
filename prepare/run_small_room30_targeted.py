#!/usr/bin/env python3
"""Preflight exact1560 failures, then a separate bounded repair pilot."""
import argparse
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from small_room30_adm_dataset import SmallRoom30ADM,sha
from small_room30_training_contract import SCHEMA,write_json
from small_room30_targeted import POLICY,source_contract,check_ready,plan


def load_runtime(data,source,bound,device):
    from small_room30_checkpoint_io import load_checkpoint
    from small_room30_targeted_runtime import TargetedRuntime
    from small_room30_training_runtime import restore_adapter,adapter_state,state_digest
    saved=load_checkpoint(source['last_checkpoint']['path'])
    if (saved.get('schema')!=SCHEMA or saved.get('purpose')!='RESEARCH_ONPOLICY_PILOT'
            or saved.get('binding')!=bound['parent_binding'] or saved.get('step')!=1560
            or saved.get('seed')!=source['seed'] or saved.get('teacher_checkpoint_authorized') is not False):
        raise ValueError('1560 checkpoint metadata mismatch')
    runtime=TargetedRuntime(data,bound['original_binding']['inputs'],device,source['seed'])
    if runtime.base_digest!=saved['base_digest'] or runtime.encoder_weight!=saved['encoder_weight']:
        raise ValueError('Frozen base/encoder changed')
    runtime.install();restore_adapter(runtime.model,saved['adapter'])
    if state_digest(adapter_state(runtime.model))!=state_digest(saved['adapter']):
        raise ValueError('Source restore differs')
    return runtime


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['smoke','train'],required=True)
    p.add_argument('--dataset',type=Path,default=Path('data/small_room30_adm_lora_v1'))
    p.add_argument('--source-summary',type=Path,default=Path('outputs/small_room30_onpolicy_run01/summary.json'))
    p.add_argument('--evaluation-summary',type=Path,default=Path('outputs/small_room30_onpolicy_eval_full01/summary.json'))
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--readiness',type=Path)
    p.add_argument('--updates',type=int,choices=[4,8,12],default=12)
    p.add_argument('--allow-research-training',action='store_true')
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    if a.mode=='train' and (not a.readiness or not a.allow_research_training): p.error('Need readiness and research flag')
    if a.mode=='smoke' and (a.readiness or a.allow_research_training): p.error('Training flags in smoke')
    if Path.cwd().resolve()!=REPO: raise ValueError('Run from AMDM root')
    a.output_dir.mkdir(parents=True,exist_ok=False)
    try:
        data=SmallRoom30ADM(a.dataset)
        source,_,bound=source_contract(data,a.source_summary,a.evaluation_summary,REPO)
        common=dict(schema=SCHEMA,binding=bound,seed=source['seed'],source_summary=str(a.source_summary.resolve()),
            evaluation_summary=str(a.evaluation_summary.resolve()),teacher_checkpoint_authorized=False)
        import torch
        from fewshot_cdm_lora import lora_named_parameters
        from small_room30_training_runtime import frozen_digest,adapter_state,state_digest
        from small_room30_targeted_runtime import trace_case,reference,canary
        from run_small_room30_teacher import save_state
        if a.mode=='train':
            ready=json.loads(a.readiness.read_text());check_ready(ready,bound)
            if ready['torch_version']!=str(torch.__version__) or ready['cuda_version']!=torch.version.cuda:
                raise ValueError('Runtime changed after smoke')
        runtime=load_runtime(data,source,bound,a.device)
        original_digest=state_digest(adapter_state(runtime.model))
        if a.mode=='smoke':
            trace=[trace_case(runtime,r['case'],reference(a.evaluation_summary,r['case'])) for r in bound['failures']]
            write_json(a.output_dir/'trace.json',trace)
            canary(runtime,bound,a.evaluation_summary,a.output_dir/'source_canary',reproduce=True)
            with torch.no_grad():
                for index in range(len(data)):
                    pred,target=runtime.predict(index,(0,250,499)[index%3],runtime.seed+81000+index)
                    loss,_=runtime.objective(pred,target,index)
                    if not torch.isfinite(loss): raise ValueError('Nonfinite replay forward')
                    print('[REPLAY CHECK] %d/%d'%(index+1,len(data)),flush=True)
            if state_digest(adapter_state(runtime.model))!=original_digest: raise ValueError('Preflight mutated adapter')
        else: trace=ready['trace']
        entries=plan(data,bound,trace,a.updates,runtime.seed)
        write_json(a.output_dir/'plan.json',dict(entries=entries,policy=POLICY))
        optimizer=torch.optim.AdamW(list(lora_named_parameters(runtime.model).values()),lr=POLICY['lr'],weight_decay=0.)
        checkpoints=[];checks=[]
        # Smoke is disposable. Train ALWAYS reloads untouched1560 and fresh optimizer.
        work=entries[:2] if a.mode=='smoke' else entries
        with (a.output_dir/'train.jsonl').open('x') as log:
            for n,entry in enumerate(work,1):
                print('[TARGETED %d/%d] %s t=%d'%(n,len(work),entry['case']['filename'],entry['timestep']),flush=True)
                row=runtime.update_targeted(optimizer,entry,1560+n,a.evaluation_summary)
                log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
                print('[UPDATE] '+json.dumps(row),flush=True)
                if n==len(work) or n%4==0:
                    if frozen_digest(runtime.model)!=runtime.frozen_after_install: raise ValueError('Frozen weights changed')
                    with torch.no_grad(): before=runtime.predict(0,499,runtime.seed+991)[0].cpu()
                    name='SMOKE_ONLY.pt' if a.mode=='smoke' else 'targeted_adapter_step_%06d.pt'%(1560+n)
                    purpose='DISPOSABLE_TARGETED_SMOKE' if a.mode=='smoke' else 'RESEARCH_TARGETED_PILOT'
                    cp=a.output_dir/name;digest=save_state(cp,runtime,optimizer,bound,1560+n,runtime.seed,purpose)
                    with torch.no_grad(): after=runtime.predict(0,499,runtime.seed+991)[0].cpu()
                    if not torch.equal(before,after): raise ValueError('Disk roundtrip changed prediction')
                    checkpoints.append(dict(path=str(cp.resolve()),sha256=digest,step=1560+n))
                    if a.mode=='train':
                        # No optimizer in canary; report-only custom panel, not a forged full summary.
                        optimizer.zero_grad(set_to_none=True)
                        result=canary(runtime,bound,a.evaluation_summary,a.output_dir/('canary_step_%06d'%(1560+n)))
                        checks.append(dict(step=1560+n,decision=result['decision']))
                        if result['decision']['stop_for_regression'] or result['decision']['all_four_repaired']: break
        if sha(source['last_checkpoint']['path'])!=source['last_checkpoint']['sha256']:
            raise ValueError('Source checkpoint changed')
        if a.mode=='smoke':
            report=dict(common,status='TARGETED_RUNTIME_READY',trace=trace,main_training_started=False,
                checks={k:True for k in ['exact_paths_reproduced','corrective_gradients','all_80_forward_finite',
                    'disposable_update','disk_roundtrip','frozen_unchanged']},smoke_checkpoint=checkpoints[-1],
                torch_version=str(torch.__version__),cuda_version=torch.version.cuda)
        else:
            stopped=checks[-1]['decision']['stop_for_regression']
            report=dict(common,status='TARGETED_STOPPED_REGRESSION' if stopped else 'TARGETED_PILOT_FINISHED_NOT_APPROVED',
                completed_step=checkpoints[-1]['step'],last_checkpoint=checkpoints[-1],checkpoints=checkpoints,
                canary_checks=checks,extra_steps=checkpoints[-1]['step']-1560,readiness_sha256=sha(a.readiness),
                source_checkpoint_preserved=True,full_evaluation_completed=False,
                recommended_next='Keep1560 and inspect regression' if stopped else 'Run full paired evaluation and visual review')
        write_json(a.output_dir/'summary.json',report)
        print('[%s] %s/summary.json'%(report['status'],a.output_dir),flush=True)
        print('[NOTICE] Original1560 preserved. No Teacher approval or automatic further training.',flush=True)
    except Exception as exc:
        write_json(a.output_dir/'failure.json',dict(status='FAILED',error=str(exc),teacher_checkpoint_authorized=False))
        raise


if __name__=='__main__': main()
