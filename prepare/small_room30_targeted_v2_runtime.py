"""Fresh transition-state gradients and transactional candidate checks."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
from small_room30_targeted_v2 import POLICY
from small_room30_onpolicy_runtime import OnPolicyRuntime
from small_room30_targeted_runtime import reference,canary
from small_room30_training_runtime import adapter_state,restore_adapter,state_digest,frozen_digest,assert_frozen
from small_room30_coverage import surface_support
from small_room30_evaluation_common import seeds
from fewshot_cdm_lora import lora_named_parameters
from small_room30_training_contract import write_json


def surface_means(raw,masks,names):
    return {n:float(np.where(m,raw,-np.inf).max(axis=1)[m.any(axis=1)].astype(np.float64).mean()) for m,n in zip(masks,names)}


def preserve_loss(physical,old,masks):
    if physical.shape!=old.shape or physical.ndim!=3 or physical.shape[0]!=1:raise ValueError('Bad guard shapes')
    values=[]
    for mask in masks:
        if mask.dtype!=torch.bool or mask.shape!=physical.shape[1:] or not mask.any():raise ValueError('Bad guard surface')
        core=mask.any(dim=1)
        now=physical[0].masked_fill(~mask,-torch.inf).max(dim=1).values[core]
        ref=old[0].masked_fill(~mask,-torch.inf).max(dim=1).values[core]
        values.append(torch.relu(ref-now-POLICY['guard_loss_tolerance']).square().mean())
    if not values:raise ValueError('Missing guard furniture')
    per_object=torch.stack(values)
    surface=per_object.mean()+per_object.max()
    full=(physical-old).square().mean()
    return POLICY['guard_surface_weight']*surface+POLICY['guard_map_weight']*full,dict(surface=surface,full_map=full)


def backward_term(loss,weight,params):
    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
    weighted=loss*weight
    grads=[g for g in torch.autograd.grad(weighted,params,retain_graph=True,allow_unused=True) if g is not None]
    if not grads or not all(torch.isfinite(g).all() for g in grads):raise ValueError('Missing/nonfinite term gradient')
    norm=sum(float(g.detach().double().square().sum()) for g in grads)**.5
    weighted.backward()
    return dict(loss=float(loss.detach()),weight=weight,weighted_gradient_norm=norm)


class RepairRuntime(OnPolicyRuntime):
    def gradients(self,optimizer,entry,step,evaluation_path):
        if len(entry['timesteps'])!=3 or len(entry['guards'])!=3 or len(entry['replay'])!=8:
            raise ValueError('Need3 transition probes,3 free guards and8 replay groups')
        if [c['action'] for c in entry['guards']]!=['sit','lie','write_board']:
            raise ValueError('All three actions require free protection in every update')
        if entry['guards'][-1]['filename']!='small_room_0404__go_write_board__g1.npz':
            raise ValueError('Missing exact Write0404 guard')
        if any([self.data.samples[i]['action'] for i in g]!=['sit','lie','write_board'] for g in entry['replay']):
            raise ValueError('Unbalanced replay')
        self.model.eval();optimizer.zero_grad(set_to_none=True)
        named=lora_named_parameters(self.model);params=list(named.values());target_terms=[];guard_terms=[]
        c=entry['case']
        for t in entry['timesteps']:
            pred,target=self.free_prediction(c['index'],t,*seeds(c))
            loss,_=self.objective(pred,target,c['index'])
            target_terms.append(dict(timestep=t,**backward_term(loss,POLICY['target_weight']/3.,params)))
            del pred,target,loss
        for guard in entry['guards']:
            pred,_=self.free_prediction(guard['index'],0,*seeds(guard))
            old=torch.from_numpy(reference(evaluation_path,guard)[None]).to(self.device)
            masks,_,_=surface_support(self.data,guard['index'])
            loss,parts=preserve_loss(pred*self.std_t+self.mean_t,old,torch.as_tensor(masks,device=self.device,dtype=torch.bool))
            values={k:float(v.detach()) for k,v in parts.items()}
            guard_terms.append(dict(case=guard,parts=values,**backward_term(loss,1./3.,params)))
            del pred,old,loss,parts
        replay=[]
        for group in entry['replay']:
            for index in group:
                slot=len(replay);t=int(np.random.default_rng(self.seed+step*1009+slot).integers(0,500))
                pred,target=self.predict(index,t,self.seed+step*10007+slot)
                loss,_=self.objective(pred,target,index)
                if not torch.isfinite(loss):raise ValueError('Nonfinite replay')
                (loss*POLICY['replay_weight']/24.).backward()
                replay.append(dict(index=index,timestep=t,loss=float(loss.detach())))
        grads=[p.grad for p in params if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):raise ValueError('Nonfinite accumulated gradient')
        norm=torch.nn.utils.clip_grad_norm_(params,POLICY['grad_clip'])
        if not torch.isfinite(norm) or float(norm)<=0:raise ValueError('Missing finite update direction')
        assert_frozen(self.model)
        return dict(step=step,case=c,target_terms=target_terms,guard_terms=guard_terms,replay=replay,
            gradient_norm_before_clip=float(norm),fresh_current_policy=True)


def tree_equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(tree_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(tree_equal(x,y) for x,y in zip(a,b))
    return a==b


def transactional_step(runtime,optimizer,evaluate):
    """Reuse one clipped gradient; retry from identical pre-step weights+moments."""
    old=adapter_state(runtime.model);state=copy.deepcopy(optimizer.state_dict())
    digest=state_digest(old);frozen=frozen_digest(runtime.model);attempts=[]
    def restore():
        restore_adapter(runtime.model,old);optimizer.load_state_dict(copy.deepcopy(state))
        if state_digest(adapter_state(runtime.model))!=digest or not tree_equal(optimizer.state_dict(),state):
            raise ValueError('Rollback mismatch')
    try:
        for number,scale in enumerate(POLICY['retry_lr_scales'],1):
            restore()
            for group in optimizer.param_groups:group['lr']=POLICY['lr']*scale
            optimizer.step()
            if state_digest(adapter_state(runtime.model))==digest or not all(torch.isfinite(p).all() for p in lora_named_parameters(runtime.model).values()):
                raise ValueError('Candidate update is absent/nonfinite')
            result=evaluate(number,scale)
            attempts.append(result)
            if frozen_digest(runtime.model)!=frozen:raise ValueError('Frozen model changed')
            if result['accepted']:
                optimizer.zero_grad(set_to_none=True)
                return dict(accepted=True,attempts=attempts,rollback_verified=True)
        restore();optimizer.zero_grad(set_to_none=True)
        return dict(accepted=False,attempts=attempts,rollback_verified=True)
    except Exception:
        restore();optimizer.zero_grad(set_to_none=True)
        raise


def baseline(data,bound,evaluation_path):
    rows=[]
    for role,items in [('failure',bound['failures']),('guard',bound['guards'])]:
        for item in items:
            c=item['case'];masks,names,_=surface_support(data,c['index'])
            means=surface_means(reference(evaluation_path,c),masks,names)
            rows.append(dict(case=c,role=role,before=item['score'],after=item['score'],
                before_surface_means=means,after_surface_means=means))
    return dict(rows=rows)


def measured_canary(runtime,bound,evaluation_path,output,reproduce=False):
    report=canary(runtime,bound,evaluation_path,output,reproduce=reproduce)
    for row in report['rows']:
        c=row['case'];masks,names,_=surface_support(runtime.data,c['index'])
        with np.load(Path(output)/c['filename'],allow_pickle=False) as a:
            row['before_surface_means']=surface_means(a['source_raw'],masks,names)
            row['after_surface_means']=surface_means(a['adapted_raw'],masks,names)
    report['adapter_digest']=state_digest(adapter_state(runtime.model))
    report['source_checkpoint_sha256']=bound['source_checkpoint']['sha256']
    write_json(Path(output)/'summary.json',report)
    return report


def acceptance(candidate,previous,assigned):
    if len(candidate['rows'])!=16 or [r['case'] for r in candidate['rows']]!=[r['case'] for r in previous['rows']]:
        raise ValueError('Candidate panel differs')
    reasons=[]
    if candidate['decision']['stop_for_regression']:reasons.append('existing_presence_or_action_regression')
    for row,old in zip(candidate['rows'],previous['rows']):
        name=row['case']['filename']
        if old['after']['all_furniture_present'] and not row['after']['all_furniture_present']:
            reasons.append('lost_previous_success:'+name)
        if row['role']=='guard':
            for b,n in zip(row['before']['instances'],row['after']['instances']):
                if b['id']!=n['id']:raise ValueError('Instance order differs')
                mid=b['id']
                if row['before_surface_means'][mid]-row['after_surface_means'][mid]>POLICY['guard_mean_drop_limit']:
                    reasons.append('surface_mean_drop:'+name+':'+mid)
                if b['hit_fraction']-n['hit_fraction']>POLICY['guard_hit_drop_limit']:
                    reasons.append('surface_hit_drop:'+name+':'+mid)
    row=next(r for r in candidate['rows'] if r['case']==assigned)
    old=next(r for r in previous['rows'] if r['case']==assigned)
    missing=[i['id'] for i in row['before']['instances'] if not i['present']]
    if len(missing)!=1:raise ValueError('Expected the single audited missing chair')
    mid=missing[0];gain=row['after_surface_means'][mid]-old['after_surface_means'][mid]
    if gain<POLICY['min_target_mean_gain']:reasons.append('no_measurable_assigned_target_gain')
    return dict(accepted=not reasons,reasons=reasons,assigned_target_mean_gain=gain,
        teacher_checkpoint_authorized=False)
