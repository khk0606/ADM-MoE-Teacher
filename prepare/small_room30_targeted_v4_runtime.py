"""Positive-only missing-instance repair; other paths have preservation, not GT fitting."""
import math
from pathlib import Path
import numpy as np
import torch
from fewshot_cdm_lora import lora_named_parameters
from small_room30_onpolicy_runtime import OnPolicyRuntime
from small_room30_targeted_v4 import POLICY, instance_support, presence, acceptance
from small_room30_targeted_v2_runtime import tree_equal, backward_term
from small_room30_targeted_v2_runtime import transactional_step as original_transaction
from small_room30_targeted_v2 import POLICY as TRANSACTION_POLICY
from small_room30_targeted_runtime import canary, reference
from small_room30_evaluation_common import seeds
from small_room30_training_runtime import adapter_state, state_digest, assert_frozen
from small_room30_training_contract import write_json


def transactional_step(runtime, optimizer, evaluate):
    # Reuse the already-tested adapter+AdamW rollback, not v3's intensity projection.
    for key in ('lr', 'retry_lr_scales'):
        if POLICY[key] != TRANSACTION_POLICY[key]: raise ValueError('Transaction policy mismatch')
    return original_transaction(runtime, optimizer, evaluate)


def missing_loss(physical, mask):
    if physical.shape != (1, 8192, 6) or mask.shape != (8192,) or mask.dtype != torch.bool or mask.sum() < 16:
        raise ValueError('Invalid target instance')
    values = physical[0].max(dim=1).values[mask]
    k = int(math.ceil(values.numel() * POLICY['minimum_hit_fraction']))
    # Same point-count criterion as any_joint presence; no floor halo or GT channels.
    return torch.relu(POLICY['training_margin'] - values.topk(k).values).square().mean()


def preservation_loss(physical, old, masks):
    if physical.shape != old.shape or physical.shape != (1, 8192, 6): raise ValueError('Bad preservation shape')
    if masks.ndim != 2 or masks.shape[1] != 8192 or masks.dtype != torch.bool: raise ValueError('Bad masks')
    now = physical[0].max(dim=1).values; ref = old[0].max(dim=1).values
    losses = []
    for mask in masks:
        if mask.sum() < 16: raise ValueError('Tiny target')
        prior = ref[mask]; k = int(math.ceil(prior.numel() * POLICY['minimum_hit_fraction']))
        if int((prior >= POLICY['activation_threshold']).sum()) < k: continue
        values, indices = prior.topk(k)
        # Only maintain visible target points up to .5; stronger/brighter changes are free.
        floor = values.clamp(max=POLICY['training_margin']).detach()
        losses.append(torch.relu(floor - now[mask][indices]).square().mean())
    return torch.stack(losses).mean() if losses else physical.sum() * 0.


class RepairRuntime(OnPolicyRuntime):
    def gradients(self, optimizer, entry, step, evaluation_path):
        c = entry['case']
        if c['filename'] not in POLICY['target_paths'] or c['scene_id'] not in POLICY['target_rooms'] or c['action'] != 'sit':
            raise ValueError('Repair outside three rooms')
        if len(entry['timesteps']) != 3 or 0 not in entry['timesteps'] or len(entry['guards']) != 3:
            raise ValueError('Invalid probe/guard plan')
        self.model.eval(); optimizer.zero_grad(set_to_none=True)
        params = list(lora_named_parameters(self.model).values())
        masks, names, _ = instance_support(self.data, c['index'])
        masks_t = torch.as_tensor(masks, device=self.device, dtype=torch.bool)
        selected = names.index(c['scene_id'] + '__100__sit')
        terms, guards = [], []
        for t in entry['timesteps']:
            pred, _ = self.free_prediction(c['index'], t, *seeds(c))
            loss = missing_loss(pred*self.std_t+self.mean_t, masks_t[selected])
            terms.append(dict(timestep=t, **backward_term(loss, POLICY['target_weight']/3., params)))
            del pred, loss
        # Include the assigned case to protect its other, already visible furniture.
        for guard in [c] + entry['guards']:
            pred, _ = self.free_prediction(guard['index'], 0, *seeds(guard))
            prior = torch.from_numpy(reference(evaluation_path, guard)[None]).to(self.device)
            gm, _, _ = instance_support(self.data, guard['index'])
            loss = preservation_loss(pred*self.std_t+self.mean_t, prior,
                torch.as_tensor(gm, device=self.device, dtype=torch.bool))
            guards.append(dict(case=guard, **backward_term(loss, POLICY['preservation_weight']/4., params)))
            del pred, prior, loss
        grads = [p.grad for p in params if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads): raise ValueError('Invalid gradient')
        norm = torch.nn.utils.clip_grad_norm_(params, POLICY['grad_clip'])
        if not torch.isfinite(norm) or float(norm) <= 0: raise ValueError('No finite correction')
        assert_frozen(self.model)
        return dict(step=step, case=c, target_terms=terms, preservation_terms=guards,
            gradient_norm_before_clip=float(norm), fresh_current_policy=True,
            gt_extent_loss=False, negative_loss=False, cross_room_gt_replay=False)


def baseline(data, bound, evaluation_path):
    rows = []
    for role, items in [('failure', bound['failures']), ('guard', bound['guards'])]:
        for item in items:
            c = item['case']; masks, names, _ = instance_support(data, c['index'])
            p = presence(reference(evaluation_path, c), masks, names)
            rows.append(dict(case=c, role=role, before_presence=p, after_presence=p))
    return dict(rows=rows)


def measured_canary(runtime, bound, evaluation_path, output, reproduce=False):
    report = canary(runtime, bound, evaluation_path, output, reproduce=reproduce)
    report['legacy_gt_diagnostics'] = report.pop('decision')
    repaired = []
    for row in report['rows']:
        c = row['case']; masks, names, _ = instance_support(runtime.data, c['index'])
        with np.load(Path(output)/c['filename'], allow_pickle=False) as z:
            row['before_presence'] = presence(z['source_raw'], masks, names)
            row['after_presence'] = presence(z['adapted_raw'], masks, names)
        if row['role'] == 'failure' and row['after_presence']['all_furniture_present']:
            repaired.append(c['filename'])
        print('[PRESENCE] %s %s' % (c['filename'], row['after_presence']), flush=True)
    report['decision'] = dict(all_four_repaired=len(repaired)==4, repaired_failures=repaired,
        visual_approval_required=True, teacher_checkpoint_authorized=False)
    report['adapter_digest'] = state_digest(adapter_state(runtime.model))
    report['source_checkpoint_sha256'] = bound['source_checkpoint']['sha256']
    report['presence_policy'] = POLICY
    write_json(Path(output)/'summary.json', report)
    return report


def check_missing_gradients(rt, bound, trace):
    rows = []; params = list(lora_named_parameters(rt.model).values())
    for item in bound['failures']:
        c = item['case']; pred, _ = rt.free_prediction(c['index'], 0, *seeds(c))
        masks, names, _ = instance_support(rt.data, c['index'])
        mask = torch.as_tensor(masks[names.index(c['scene_id']+'__100__sit')], device=rt.device)
        loss = missing_loss(pred*rt.std_t+rt.mean_t, mask)
        grads = [g for g in torch.autograd.grad(loss, params, allow_unused=True) if g is not None]
        norm = sum(float(g.detach().double().square().sum()) for g in grads)**.5
        if not grads or not all(torch.isfinite(g).all() for g in grads) or norm <= 0:
            raise ValueError('Missing target has no finite corrective gradient')
        rows.append(dict(case=c, loss=float(loss.detach()), gradient_norm=norm))
    return rows
