"""Exact-seed diagnostic, fresh-policy repair and explicit regression canaries."""
from pathlib import Path
import numpy as np
import torch

from small_room30_targeted import POLICY, score
from small_room30_onpolicy_runtime import OnPolicyRuntime, clean_prediction
from small_room30_evaluation_common import conditioning_arrays, seeds, load_case
from small_room30_training_runtime import adapter_state, state_digest, frozen_digest, assert_frozen
from small_room30_coverage import surface_support
from small_room30_coverage_runtime import coverage_terms
from fewshot_cdm_lora import lora_named_parameters
from relational_teacher_v9_lora_runtime import deterministic_noise
from evaluate_small_room30_onpolicy import rollout


def reference(evaluation_path,case):
    return load_case(Path(evaluation_path).parent/'cases'/case['filename'])['adapted_raw']


def trace_case(runtime,case,expected):
    """One original rollout; store x_t BEFORE t for a few clean-prediction probes."""
    _, values=conditioning_arrays(runtime.data,case['index'])
    kwargs={k:v if k=='c_text' else torch.from_numpy(v).to(runtime.device) for k,v in values.items()}
    initial,reverse=seeds(case)
    state=deterministic_noise((1,8192,6),initial,runtime.device)
    device=torch.device(runtime.device)
    devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
    states={}; records=[]; runtime.model.eval()
    with torch.random.fork_rng(devices=devices),torch.no_grad():
        torch.manual_seed(reverse)
        if device.type=='cuda': torch.cuda.manual_seed(reverse)
        count=0
        for count,out in enumerate(runtime.diffusion.p_sample_loop_progressive(runtime.model,state.shape,
                noise=state,clip_denoised=False,model_kwargs=kwargs,device=runtime.device,progress=False),1):
            t=500-count
            if t in POLICY['trace_timesteps']:
                states[t]=state.detach().cpu().clone()
                pred=out['pred_xstart']
                target=(torch.from_numpy(runtime.data.sample(case['index'])['target'][None]).to(runtime.device)-runtime.mean_t)/runtime.std_t
                masks,_,_=surface_support(runtime.data,case['index'])
                parts=coverage_terms(pred*runtime.std_t+runtime.mean_t,target*runtime.std_t+runtime.mean_t,
                    torch.as_tensor(masks,device=runtime.device,dtype=torch.bool))
                raw=(pred[0].cpu().numpy()*runtime.std+runtime.mean).astype(np.float32)
                records.append(dict(timestep=t,surface_loss=float(parts['surface_mean']+parts['surface_worst']),
                    score=score(runtime.data,case,raw)))
            state=out['sample']
            if count%100==0: print('[TRACE %s] %d/500'%(case['filename'],count),flush=True)
        if count!=500: raise ValueError('Incomplete trace')
        raw=(state[0].cpu().numpy()*runtime.std+runtime.mean).astype(np.float32)
    if not np.allclose(raw,expected,atol=1e-5,rtol=1e-5):
        raise ValueError('1560 exact-seed path does not reproduce saved final: '+case['filename'])
    selected=sorted(records,key=lambda r:(-r['surface_loss'],-r['timestep']))[:3]
    params=list(lora_named_parameters(runtime.model).values())
    for row in selected:
        t=row['timestep']; pred=clean_prediction(runtime,states[t].to(runtime.device),kwargs,t)
        parts=coverage_terms(pred*runtime.std_t+runtime.mean_t,target*runtime.std_t+runtime.mean_t,
            torch.as_tensor(masks,device=runtime.device,dtype=torch.bool))
        loss=parts['surface_mean']+parts['surface_worst']
        gradients=[g for g in torch.autograd.grad(loss,params,allow_unused=True) if g is not None]
        norm=sum(float(g.detach().double().square().sum()) for g in gradients)**.5
        if not gradients or not all(torch.isfinite(g).all() for g in gradients) or norm<=0 or float(loss.detach())<=0:
            raise ValueError('Missing finite nonzero coverage correction at selected timestep')
        row['corrective_gradient_norm']=norm
    return dict(case=case,seeds=[initial,reverse],records=records,
        selected_timesteps=[r['timestep'] for r in selected],
        replay_max_abs_error=float(np.abs(raw-expected).max()))


def regression_decision(rows):
    guards=[r for r in rows if r['role']=='guard']
    lost=[r['case']['filename'] for r in guards if not r['after']['all_furniture_present']]
    warnings=[]
    for action in ('sit','lie','write_board'):
        subset=[r for r in guards if r['case']['action']==action]
        if len(subset)!=4: raise ValueError('Canary needs four guards per action')
        mae=float(np.mean([r['after']['mae']-r['before']['mae'] for r in subset]))
        bg=float(np.mean([(r['after']['background_point_hit_fraction'] or 0.)-
            (r['before']['background_point_hit_fraction'] or 0.) for r in subset]))
        if mae>POLICY['guard_mae_increase_limit'] or bg>POLICY['guard_background_increase_limit']:
            warnings.append(dict(action=action,mean_mae_increase=mae,mean_background_increase=bg))
    repaired=[r['case']['filename'] for r in rows if r['role']=='failure' and r['after']['all_furniture_present']]
    return dict(stop_for_regression=bool(lost or warnings),lost_guard_cases=lost,
        regression_warnings=warnings,repaired_failures=repaired,
        all_four_repaired=len(repaired)==4,teacher_checkpoint_authorized=False)


def canary(runtime,bound,evaluation_path,output_dir=None,reproduce=False):
    from small_room30_training_contract import write_json
    before=(state_digest(adapter_state(runtime.model)),frozen_digest(runtime.model))
    rows=[]
    if output_dir is not None: Path(output_dir).mkdir(parents=True,exist_ok=False)
    for role in ('failure','guard'):
        for item in bound['failures' if role=='failure' else 'guards']:
            case=item['case']; _,values=conditioning_arrays(runtime.data,case['index'])
            raw=rollout(runtime,values,*seeds(case),label='CANARY '+case['filename'])
            expected=reference(evaluation_path,case)
            if reproduce and not np.allclose(raw,expected,atol=1e-5,rtol=1e-5):
                raise ValueError('Source canary failed exact-seed reproduction: '+case['filename'])
            row=dict(case=case,role=role,before=item['score'],after=score(runtime.data,case,raw))
            if output_dir is not None:
                path=Path(output_dir)/case['filename']
                with path.open('xb') as f: np.savez_compressed(f,source_raw=expected,adapted_raw=raw)
                from small_room30_adm_dataset import sha
                row['sha256']=sha(path)
            rows.append(row)
            print('[CANARY] %s present=%s'%(case['filename'],row['after']['all_furniture_present']),flush=True)
    if before!=(state_digest(adapter_state(runtime.model)),frozen_digest(runtime.model)):
        raise ValueError('Canary mutated model')
    result=dict(scope='Custom 4 failed paths + 12 guards; NOT the full 240-case evaluation',
        rows=rows,decision=regression_decision(rows),teacher_checkpoint_authorized=False)
    if output_dir is not None: write_json(Path(output_dir)/'summary.json',result)
    return result


class TargetedRuntime(OnPolicyRuntime):
    def update_targeted(self,optimizer,entry,step,evaluation_path):
        optimizer.zero_grad(set_to_none=True)
        case=entry['case']; pred,target=self.free_prediction(case['index'],entry['timestep'],*seeds(case))
        loss,parts=self.objective(pred,target,case['index'])
        if not torch.isfinite(loss): raise ValueError('Nonfinite target loss')
        loss.backward(); target_loss=float(loss.detach())
        del pred,target,loss,parts
        # Preserve source1560 final behavior on a separate successful free path, not GT-noised input.
        guard=entry['guard']; pred,_=self.free_prediction(guard['index'],0,*seeds(guard))
        old=torch.from_numpy(reference(evaluation_path,guard)[None]).to(self.device)
        loss=(pred*self.std_t+self.mean_t-old).square().mean()
        if not torch.isfinite(loss): raise ValueError('Nonfinite free guard loss')
        (POLICY['free_guard_weight']*loss).backward(); guard_loss=float(loss.detach())
        del pred,loss,old
        replay=[]
        if len(entry['replay'])!=3 or any([self.data.samples[i]['action'] for i in g]!=['sit','lie','write_board'] for g in entry['replay']):
            raise ValueError('Need three balanced replay groups')
        for group in entry['replay']:
            for index in group:
                slot=len(replay); t=int(np.random.default_rng(self.seed+step*1009+slot).integers(0,500))
                pred,target=self.predict(index,t,self.seed+step*10007+slot)
                loss,_=self.objective(pred,target,index)
                if not torch.isfinite(loss): raise ValueError('Nonfinite replay loss')
                (POLICY['replay_weight']*loss/9.).backward()
                replay.append(dict(index=index,timestep=t,loss=float(loss.detach())))
        named=lora_named_parameters(self.model); grads=[p.grad for p in named.values() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads): raise ValueError('Missing/nonfinite gradients')
        norm=torch.nn.utils.clip_grad_norm_(list(named.values()),POLICY['grad_clip'])
        if not torch.isfinite(norm) or float(norm)<=0: raise ValueError('Nonzero finite gradient required')
        assert_frozen(self.model); before=state_digest(named);optimizer.step()
        if before==state_digest(named) or not all(torch.isfinite(p).all() for p in named.values()):
            raise ValueError('Missing finite update')
        return dict(step=step,case=case,timestep=entry['timestep'],seeds=seeds(case),
            target_loss=target_loss,guard=guard,free_guard_loss=guard_loss,replay=replay,
            gradient_norm=float(norm),fresh_current_policy=True)
