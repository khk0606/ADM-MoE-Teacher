"""Endpoint-only LoRA gradients. No GT fitting, TV suppression, or map editing."""
import math
from pathlib import Path
import numpy as np
import torch
from small_room30_targeted_v5 import (POLICY, TARGETS, instance_support, presence,
    reference, target_stats, acceptance)
from small_room30_targeted_v4_runtime import transactional_step, tree_equal
from small_room30_targeted_v2_runtime import backward_term
from small_room30_onpolicy_runtime import OnPolicyRuntime
from small_room30_evaluation_common import conditioning_arrays, seeds
from evaluate_small_room30_onpolicy import rollout
from fewshot_cdm_lora import lora_named_parameters
from small_room30_training_runtime import adapter_state, state_digest, frozen_digest, assert_frozen
from small_room30_training_contract import write_json
from small_room30_adm_dataset import sha


def missing_loss(physical, mask):
    if physical.shape != (1,8192,6) or mask.shape != (8192,) or mask.dtype != torch.bool or mask.sum()<16:
        raise ValueError('Invalid target')
    values = physical[0].max(dim=1).values[mask]
    k = int(math.ceil(values.numel()*POLICY['target_fraction']))
    return torch.relu(POLICY['training_margin']-values.topk(k).values).square().mean()


def preservation_loss(physical, old, masks):
    if physical.shape != old.shape or physical.shape != (1,8192,6): raise ValueError('Invalid preservation map')
    if masks.dtype != torch.bool or masks.ndim != 2 or masks.shape[1]!=8192: raise ValueError('Bad masks')
    now = physical[0].max(dim=1).values; ref = old[0].max(dim=1).values
    losses = []
    for mask in masks:
        # Preserve ALL visible target points of1570 up to.5, not just its best25%.
        active = mask & (ref >= .3)
        if active.any():
            losses.append(torch.relu(ref[active].clamp(max=.5).detach()-now[active]).square().mean())
    return torch.stack(losses).mean() if losses else physical.sum()*0.


class RepairRuntime(OnPolicyRuntime):
    def gradients(self, optimizer, entry, step, evaluation_path):
        c = entry['case']
        if c['filename'] not in TARGETS or c['action']!='sit' or c['scene_id'] not in POLICY['target_rooms']:
            raise ValueError('Outside two-path repair')
        if entry['timesteps'] != [0] or len(entry['guards']) != 5: raise ValueError('Wrong endpoint/guard plan')
        self.model.eval(); optimizer.zero_grad(set_to_none=True)
        params = list(lora_named_parameters(self.model).values())
        masks, names, _ = instance_support(self.data,c['index'])
        mask = torch.as_tensor(masks[names.index(c['scene_id']+'__100__sit')],device=self.device)
        pred, _ = self.free_prediction(c['index'],0,*seeds(c))
        loss = missing_loss(pred*self.std_t+self.mean_t,mask)
        term = dict(timestep=0,**backward_term(loss,POLICY['target_weight'],params))
        del pred,loss
        guards = []
        for guard in [c]+entry['guards']:
            pred, _ = self.free_prediction(guard['index'],0,*seeds(guard))
            old = torch.from_numpy(reference(evaluation_path,guard)[None]).to(self.device)
            gm, _, _ = instance_support(self.data,guard['index'])
            loss = preservation_loss(pred*self.std_t+self.mean_t,old,torch.as_tensor(gm,device=self.device))
            guards.append(dict(case=guard,**backward_term(loss,POLICY['preservation_weight']/6.,params)))
            del pred,old,loss
        grads = [p.grad for p in params if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads): raise ValueError('Invalid gradient')
        norm = torch.nn.utils.clip_grad_norm_(params,POLICY['grad_clip'])
        if not torch.isfinite(norm) or float(norm)<=0: raise ValueError('No finite correction')
        assert_frozen(self.model)
        return dict(step=step,case=c,target_terms=[term],preservation_terms=guards,
            gradient_norm_before_clip=float(norm),fresh_current_policy=True,
            gt_extent_loss=False,negative_loss=False,cross_room_gt_replay=False,
            preservation_anchor_step=1570,detached_prefix_steps=499)


def annotate(data, case, raw):
    masks,names,_ = instance_support(data,case['index'])
    p = presence(raw,masks,names)
    stats = target_stats(raw,masks[names.index(case['scene_id']+'__100__sit')]) if case['filename'] in TARGETS else None
    return p,stats


def baseline(data,bound,evaluation_path):
    rows=[]
    for item in bound['failures']+bound['guards']:
        c=item['case']; p,s=annotate(data,c,reference(evaluation_path,c))
        rows.append(dict(case=c,role='repair' if c['filename'] in TARGETS else 'preservation',
            before_presence=p,after_presence=p,repair=s))
    return dict(rows=rows)


def measured_canary(runtime,bound,evaluation_path,output,reproduce=False):
    before=(state_digest(adapter_state(runtime.model)),frozen_digest(runtime.model))
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for row in baseline(runtime.data,bound,evaluation_path)['rows']:
        c=row['case'];_,values=conditioning_arrays(runtime.data,c['index'])
        raw=rollout(runtime,values,*seeds(c),label='V5 CANARY '+c['filename'])
        expected=reference(evaluation_path,c)
        if reproduce and not np.allclose(raw,expected,atol=1e-5,rtol=1e-5):
            raise ValueError('Saved1570 reproduction failed: '+c['filename'])
        row['after_presence'],row['repair']=annotate(runtime.data,c,raw)
        file=output/c['filename']
        with file.open('xb') as f: np.savez_compressed(f,source_raw=expected,adapted_raw=raw)
        row['sha256']=sha(file);rows.append(row)
        print('[FINAL TARGET] %s %s'%(c['filename'],row['repair']),flush=True)
    if before!=(state_digest(adapter_state(runtime.model)),frozen_digest(runtime.model)):
        raise ValueError('Canary mutated weights')
    report=dict(rows=rows,adapter_digest=before[0],source_checkpoint_sha256=bound['source_checkpoint']['sha256'],
        policy=POLICY,decision=dict(optimization_complete=all(r['repair']['optimization_complete'] for r in rows if r['repair']),
        visual_approval_required=True,teacher_checkpoint_authorized=False),
        scope='2 repair +14 preservation paths; NOT a full30-room evaluation',teacher_checkpoint_authorized=False)
    write_json(output/'summary.json',report)
    return report


def check_missing_gradients(rt,bound,trace):
    params=list(lora_named_parameters(rt.model).values());rows=[]
    for item in bound['failures']:
        c=item['case']
        if c['filename'] not in TARGETS: continue
        pred,_=rt.free_prediction(c['index'],0,*seeds(c))
        masks,names,_=instance_support(rt.data,c['index'])
        mask=torch.as_tensor(masks[names.index(c['scene_id']+'__100__sit')],device=rt.device)
        loss=missing_loss(pred*rt.std_t+rt.mean_t,mask)
        grads=[g for g in torch.autograd.grad(loss,params,allow_unused=True) if g is not None]
        norm=sum(float(g.detach().double().square().sum()) for g in grads)**.5
        if not grads or not all(torch.isfinite(g).all() for g in grads) or norm<=0:
            raise ValueError('Missing final-output gradient')
        rows.append(dict(case=c,loss=float(loss.detach()),gradient_norm=norm,timestep=0))
    if len(rows)!=2: raise ValueError('Need exactly2 repair gradient probes')
    return rows
