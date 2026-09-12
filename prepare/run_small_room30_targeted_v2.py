#!/usr/bin/env python3
"""Readiness then bounded transition repair with real-rollout accept/rollback."""
import argparse
import copy
import json
from pathlib import Path
import sys
import numpy as np
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from small_room30_adm_dataset import SmallRoom30ADM,sha
from small_room30_training_contract import SCHEMA,write_json
from small_room30_targeted_v2 import POLICY,source_contract,check_ready,plan


def load_runtime(data,source,bound,device):
    from small_room30_checkpoint_io import load_checkpoint
    from small_room30_targeted_v2_runtime import RepairRuntime
    from small_room30_training_runtime import restore_adapter,adapter_state,state_digest
    saved=load_checkpoint(source['last_checkpoint']['path'])
    if (saved.get('schema')!=SCHEMA or saved.get('purpose')!='RESEARCH_ONPOLICY_PILOT'
            or saved.get('binding')!=bound['parent_binding'] or saved.get('step')!=1560
            or saved.get('seed')!=source['seed'] or saved.get('teacher_checkpoint_authorized') is not False):
        raise ValueError('Original1560 checkpoint required')
    rt=RepairRuntime(data,bound['original_binding']['inputs'],device,source['seed'])
    if rt.base_digest!=saved['base_digest'] or rt.encoder_weight!=saved['encoder_weight']:raise ValueError('Base/encoder changed')
    rt.install();restore_adapter(rt.model,saved['adapter'])
    if state_digest(adapter_state(rt.model))!=state_digest(saved['adapter']):raise ValueError('Restore differs')
    return rt


def check_transitions(rt,bound,trace):
    import torch
    from fewshot_cdm_lora import lora_named_parameters
    from small_room30_evaluation_common import seeds
    from small_room30_coverage import surface_support
    from small_room30_coverage_runtime import coverage_terms
    from small_room30_targeted import score
    results=[]
    for item,old in zip(bound['transitions'],trace):
        c=item['case']
        for t in item['timesteps']:
            pred,target=rt.free_prediction(c['index'],t,*seeds(c))
            physical=pred*rt.std_t+rt.mean_t
            actual=score(rt.data,c,(pred[0].detach().cpu().numpy()*rt.std+rt.mean).astype(np.float32))
            expected=next(r for r in old['records'] if r['timestep']==t)['score']
            if actual['instances']!=expected['instances'] or abs(actual['mae']-expected['mae'])>1e-5:
                raise ValueError('Source transition does not reproduce: '+c['filename']+' t='+str(t))
            masks,_,_=surface_support(rt.data,c['index'])
            parts=coverage_terms(physical,target*rt.std_t+rt.mean_t,torch.as_tensor(masks,device=rt.device,dtype=torch.bool))
            loss=parts['surface_mean']+parts['surface_worst']
            grads=[g for g in torch.autograd.grad(loss,list(lora_named_parameters(rt.model).values()),allow_unused=True) if g is not None]
            norm=sum(float(g.detach().double().square().sum()) for g in grads)**.5
            if not grads or not all(torch.isfinite(g).all() for g in grads) or norm<=0:raise ValueError('Missing corrective gradient')
            results.append(dict(case=c,timestep=t,surface_loss=float(loss.detach()),gradient_norm=norm))
            print('[TRANSITION] '+json.dumps(results[-1]),flush=True)
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['smoke','train'],required=True)
    p.add_argument('--dataset',type=Path,default=Path('data/small_room30_adm_lora_v1'))
    p.add_argument('--source-summary',type=Path,default=Path('outputs/small_room30_onpolicy_run01/summary.json'))
    p.add_argument('--evaluation-summary',type=Path,default=Path('outputs/small_room30_onpolicy_eval_full01/summary.json'))
    p.add_argument('--diagnosis-summary',type=Path,default=Path('outputs/small_room30_targeted_ready01/summary.json'))
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--readiness',type=Path)
    p.add_argument('--allow-research-training',action='store_true')
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    if a.mode=='train' and (not a.readiness or not a.allow_research_training):p.error('Need v2 readiness and research flag')
    if a.mode=='smoke' and (a.readiness or a.allow_research_training):p.error('Training flags in smoke')
    if Path.cwd().resolve()!=REPO:raise ValueError('Run from AMDM root')
    a.output_dir.mkdir(parents=True,exist_ok=False)
    try:
        import torch
        from fewshot_cdm_lora import lora_named_parameters
        from small_room30_training_runtime import adapter_state,restore_adapter,state_digest,frozen_digest
        from small_room30_targeted_v2_runtime import measured_canary,baseline,acceptance,transactional_step,tree_equal
        from run_small_room30_teacher import save_state
        data=SmallRoom30ADM(a.dataset)
        source,_,bound,trace=source_contract(data,a.source_summary,a.evaluation_summary,a.diagnosis_summary,REPO)
        common=dict(schema=SCHEMA,binding=bound,seed=source['seed'],source_summary=str(a.source_summary.resolve()),
            evaluation_summary=str(a.evaluation_summary.resolve()),diagnosis_summary=str(a.diagnosis_summary.resolve()),
            teacher_checkpoint_authorized=False)
        if a.mode=='train':
            ready=json.loads(a.readiness.read_text());check_ready(ready,bound)
            if ready['torch_version']!=str(torch.__version__) or ready['cuda_version']!=torch.version.cuda:raise ValueError('Runtime changed after smoke')
        rt=load_runtime(data,source,bound,a.device)
        optimizer=torch.optim.AdamW(list(lora_named_parameters(rt.model).values()),lr=POLICY['lr'],weight_decay=0.)
        entries=plan(data,bound,rt.seed);write_json(a.output_dir/'plan.json',dict(entries=entries,policy=POLICY))
        def save(name,step,purpose):
            if frozen_digest(rt.model)!=rt.frozen_after_install:raise ValueError('Frozen state changed')
            with torch.no_grad():before=rt.predict(0,499,rt.seed+991)[0].cpu()
            path=a.output_dir/name;digest=save_state(path,rt,optimizer,bound,step,rt.seed,purpose)
            with torch.no_grad():after=rt.predict(0,499,rt.seed+991)[0].cpu()
            if not torch.equal(before,after):raise ValueError('Disk roundtrip changed predictions')
            return dict(path=str(path.resolve()),sha256=digest,step=step)
        if a.mode=='smoke':
            original=adapter_state(rt.model);moments=copy.deepcopy(optimizer.state_dict())
            probes=check_transitions(rt,bound,trace);write_json(a.output_dir/'transition_check.json',probes)
            measured_canary(rt,bound,a.evaluation_summary,a.output_dir/'source_canary',reproduce=True)
            with torch.no_grad():
                for index in range(len(data)):
                    pred,target=rt.predict(index,(0,250,499)[index%3],rt.seed+82000+index)
                    loss,_=rt.objective(pred,target,index)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite replay')
            if state_digest(adapter_state(rt.model))!=state_digest(original):raise ValueError('Preflight mutated adapter')
            row=rt.gradients(optimizer,entries[0],1561,a.evaluation_summary)
            write_json(a.output_dir/'disposable_gradient.json',row)
            # Force two rejections to exercise real adapter AND nonempty AdamW rollback.
            def reject(number,scale):
                cp=save('SMOKE_ATTEMPT_%d.pt'%number,1561,'DISPOSABLE_TARGETED_V2_SMOKE')
                return dict(accepted=False,forced_smoke_rejection=True,checkpoint=cp,lr_scale=scale)
            result=transactional_step(rt,optimizer,reject)
            if result['accepted'] or not tree_equal(optimizer.state_dict(),moments) or state_digest(adapter_state(rt.model))!=state_digest(original):
                raise ValueError('Smoke rollback mismatch')
            write_json(a.output_dir/'rollback_check.json',result)
            report=dict(common,status='TARGETED_V2_RUNTIME_READY',main_training_started=False,
                checks={k:True for k in ['source_reproduced','transition_gradients','all_80_forward_finite',
                    'disposable_update','optimizer_adapter_rollback','disk_roundtrip','frozen_unchanged']},
                torch_version=str(torch.__version__),cuda_version=torch.version.cuda)
        else:
            previous=baseline(data,bound,a.evaluation_summary);checkpoints=[];decisions=[];stopped=False
            with (a.output_dir/'train.jsonl').open('x') as log:
                for n,entry in enumerate(entries,1):
                    print('[V2 %d/4] transition=%s; all3 critical guards protected'%(n,entry['timesteps']),flush=True)
                    row=rt.gradients(optimizer,entry,1560+n,a.evaluation_summary)
                    def evaluate(number,scale):
                        out=a.output_dir/('proposal_%02d_try_%d'%(n,number))
                        result=measured_canary(rt,bound,a.evaluation_summary,out)
                        decision=acceptance(result,previous,entry['case'])
                        decision.update(proposal=n,attempt=number,lr_scale=scale,canary_summary=str((out/'summary.json').resolve()),
                            canary_sha256=sha(out/'summary.json'),candidate_adapter_digest=result['adapter_digest'])
                        write_json(out/'acceptance.json',decision)
                        print('[ACCEPTANCE] '+json.dumps(decision),flush=True)
                        return decision
                    decision=transactional_step(rt,optimizer,evaluate);decisions.append(decision)
                    log.write(json.dumps(dict(row,transaction=decision),allow_nan=False)+'\n');log.flush()
                    if not decision['accepted']:
                        stopped=True;break
                    chosen=decision['attempts'][-1]
                    previous=json.loads(Path(chosen['canary_summary']).read_text())
                    cp=save('targeted_v2_adapter_step_%06d.pt'%(1560+n),1560+n,'RESEARCH_TARGETED_V2_PILOT')
                    cp['adapter_digest']=state_digest(adapter_state(rt.model))
                    if cp['adapter_digest']!=chosen['candidate_adapter_digest']:raise ValueError('Saved weights differ from evaluated candidate')
                    checkpoints.append(cp)
                    if previous['decision']['all_four_repaired']:break
            report=dict(common,status='TARGETED_V2_STOPPED_REJECTED' if stopped else 'TARGETED_V2_FINISHED_REVIEW_REQUIRED',
                completed_step=checkpoints[-1]['step'] if checkpoints else 1560,last_checkpoint=checkpoints[-1] if checkpoints else None,
                accepted_updates=len(checkpoints),checkpoints=checkpoints,transactions=decisions,readiness_sha256=sha(a.readiness),
                source_checkpoint_preserved=True,rejected_updates_retained=False,full_evaluation_completed=False)
        if frozen_digest(rt.model)!=rt.frozen_after_install:raise ValueError('Frozen weights changed')
        if sha(source['last_checkpoint']['path'])!=source['last_checkpoint']['sha256']:raise ValueError('Source checkpoint changed')
        write_json(a.output_dir/'summary.json',report)
        print('[%s] %s/summary.json'%(report['status'],a.output_dir),flush=True)
        print('[STOP] No automatic extension or Teacher approval. Send summary and proposal receipts for review.',flush=True)
    except Exception as exc:
        write_json(a.output_dir/'failure.json',dict(status='FAILED',error=str(exc),teacher_checkpoint_authorized=False))
        raise


if __name__=='__main__':main()
