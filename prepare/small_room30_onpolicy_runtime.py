"""Current-policy state training; the sampler prefix has no GT access."""
import numpy as np
import torch

from small_room30_onpolicy import POLICY
from small_room30_coverage_runtime import CoverageRuntime
from small_room30_training_runtime import assert_frozen, state_digest
from relational_teacher_v9_lora_runtime import deterministic_noise
from fewshot_cdm_lora import lora_named_parameters, set_frozen_base_eval_lora_train


def free_state(runtime, kwargs, timestep, initial_seed, reverse_seed):
    """Return normalized x_t BEFORE evaluating t, not sample AFTER t (= x_(t-1))."""
    if set(kwargs) != {'c_pc_xyz', 'c_pc_feat', 'c_text'}:
        raise ValueError('GT/instance conditioning prohibited')
    if not isinstance(timestep, int) or not 0 <= timestep < 500 or runtime.diffusion.num_timesteps != 500:
        raise ValueError('Expected t in 0..499 and original 500-step sampler')
    runtime.model.eval()
    device = torch.device(runtime.device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    state = deterministic_noise(torch.Size((1, 8192, 6)), initial_seed, runtime.device)
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        torch.manual_seed(reverse_seed)
        if device.type == 'cuda': torch.cuda.manual_seed(reverse_seed)
        if timestep != 499:
            steps = 0
            for steps, out in enumerate(runtime.diffusion.p_sample_loop_progressive(
                    runtime.model, state.shape, noise=state, clip_denoised=False,
                    model_kwargs=kwargs, device=runtime.device, progress=False), 1):
                state = out['sample']
                if steps % 100 == 0:
                    print('[PREFIX] %d/%d steps; no gradient through prefix' % (steps,499-timestep),flush=True)
                # After processing 499..t+1, state is x_t. Do NOT process t yet.
                if steps == 499 - timestep: break
            if steps != 499 - timestep: raise ValueError('Incomplete sampler prefix')
    if state.shape != (1, 8192, 6) or not torch.isfinite(state).all():
        raise ValueError('Invalid generated x_t')
    return state.detach().clone()


def clean_prediction(runtime, state, kwargs, timestep):
    if set(kwargs) != {'c_pc_xyz', 'c_pc_feat', 'c_text'} or state.requires_grad:
        raise ValueError('Use detached state and scene/text only')
    t = torch.tensor([timestep], dtype=torch.long, device=runtime.device)
    # Let diffusion perform its own timestep mapping exactly once, including SpacedDiffusion.
    pred = runtime.diffusion.p_mean_variance(runtime.model, state, t,
        clip_denoised=False, model_kwargs=kwargs)['pred_xstart']
    if pred.shape != state.shape or not torch.isfinite(pred).all():
        raise ValueError('Invalid clean prediction')
    return pred


class OnPolicyRuntime(CoverageRuntime):
    def free_prediction(self, index, timestep, initial_seed, reverse_seed):
        # Scene/text only. Target is intentionally loaded AFTER free state collection.
        from small_room30_evaluation_common import conditioning_arrays
        _, values = conditioning_arrays(self.data, index)
        kwargs = {k: v if k == 'c_text' else torch.from_numpy(v).to(self.device) for k, v in values.items()}
        state = free_state(self, kwargs, timestep, initial_seed, reverse_seed)
        prediction = clean_prediction(self, state, kwargs, timestep)
        gt = self.data.sample(index)['target']
        target = (torch.from_numpy(gt[None]).to(self.device) - self.mean_t) / self.std_t
        return prediction, target

    def update_onpolicy(self, optimizer, indices, step, timestep):
        if [self.data.samples[i]['action'] for i in indices] != ['sit', 'lie', 'write_board']:
            raise ValueError('Every update must replay Sit, Lie, and Write')
        set_frozen_base_eval_lora_train(self.model)
        optimizer.zero_grad(set_to_none=True)
        # Prefix uses current weights before this update, never the frozen 1500 adapter.
        initial = self.seed + 91000000 + step * 10007
        reverse = self.seed + 92000000 + step * 10009
        pred, target = self.free_prediction(indices[0], timestep, initial, reverse)
        loss, parts = self.objective(pred, target, indices[0])
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite on-policy loss')
        (POLICY['onpolicy_weight'] * loss).backward()
        onpolicy = {k: float(v.detach()) for k, v in parts.items()}
        del pred, target, loss, parts
        replay, replay_timesteps = {}, []
        for slot, index in enumerate(indices):
            t = int(np.random.default_rng(self.seed + step * 1009 + slot).integers(0, 500))
            pred, target = self.predict(index, t, self.seed + step * 10007 + slot)
            loss, parts = self.objective(pred, target, index)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite replay loss')
            (POLICY['replay_weight'] * loss / 3).backward()
            replay_timesteps.append(t)
            for key, value in parts.items(): replay[key] = replay.get(key, 0.) + float(value.detach()) / 3
        named = lora_named_parameters(self.model)
        grads = [p.grad for p in named.values() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise FloatingPointError('Missing/nonfinite LoRA gradient')
        norm = torch.nn.utils.clip_grad_norm_(list(named.values()), POLICY['grad_clip'])
        if not torch.isfinite(norm) or float(norm) <= 0: raise ValueError('Invalid gradient norm')
        assert_frozen(self.model)
        before = state_digest(named)
        optimizer.step()
        if before == state_digest(named) or not all(torch.isfinite(p).all() for p in named.values()):
            raise ValueError('No finite parameter update')
        return dict(step=step, samples=[self.data.samples[i]['id'] for i in indices],
            onpolicy_timestep=timestep, prefix_steps=499-timestep, initial_seed=initial, reverse_seed=reverse,
            onpolicy_losses=onpolicy, replay_losses=replay, replay_timesteps=replay_timesteps,
            gradient_norm=float(norm), state_refreshed_from_current_adapter=True)
