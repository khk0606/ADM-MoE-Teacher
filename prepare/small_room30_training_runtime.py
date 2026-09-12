"""Torch implementation shared by readiness smoke and finite research training."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from small_room30_adm_dataset import inside, sha
from small_room30_training_contract import POLICY
from small_room30_checkpoint_io import load_checkpoint
from fewshot_cdm_lora import (DEFAULT_EXCLUDED_TOKENS, install_lora, lora_named_parameters,
                             set_frozen_base_eval_lora_train)
from relational_teacher_v9_lora_runtime import (compose_cdm_config, configure_reproducibility,
                                               load_stats, predict_xstart, deterministic_noise)


def strict_load(model, path):
    state = load_checkpoint(path)
    if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError('Expected a plain ADM tensor state_dict: ' + str(path))
    normalized = {}
    for key, tensor in state.items():
        if not isinstance(key, str):
            raise ValueError('Checkpoint keys must be strings')
        key = key[7:] if key.startswith('module.') else key
        if key in normalized:
            raise ValueError('Duplicate normalized checkpoint key: ' + key)
        normalized[key] = tensor
    current = model.state_dict()
    required = {k for k in current if not any(t in k for t in DEFAULT_EXCLUDED_TOKENS)}
    missing = required - normalized.keys()
    unknown = normalized.keys() - current.keys()
    if missing or unknown:
        raise ValueError(f'Checkpoint coverage mismatch: missing={sorted(missing)[:12]} unknown={sorted(unknown)[:12]}')
    for key, tensor in normalized.items():
        if tensor.shape != current[key].shape or not torch.isfinite(tensor).all():
            raise ValueError('Invalid checkpoint tensor: ' + key)
    model.load_state_dict(normalized, strict=False)  # Only pretrained encoders may be absent.
    return dict(path=str(path), sha256=sha(path), loaded_keys=len(normalized), required_keys=len(required))


def state_digest(values):
    digest = hashlib.sha256()
    for key, value in sorted(values.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode()); digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def frozen_digest(model):
    return state_digest({k: v for k, v in model.state_dict().items() if 'lora_' not in k})


def assert_frozen(model):
    for name, parameter in model.named_parameters():
        if 'lora_' not in name and (parameter.requires_grad or parameter.grad is not None):
            raise RuntimeError('Frozen parameter became trainable/received gradient: ' + name)


def adapter_state(model):
    return {k: p.detach().cpu().clone() for k, p in lora_named_parameters(model).items()}


def restore_adapter(model, state):
    named = lora_named_parameters(model)
    if set(named) != set(state):
        raise ValueError('Adapter keys differ')
    with torch.no_grad():
        for k, parameter in named.items():
            if parameter.shape != state[k].shape or not torch.isfinite(state[k]).all():
                raise ValueError('Invalid adapter tensor: ' + k)
            parameter.copy_(state[k])


class Runtime:
    def __init__(self, data, inputs, device, seed):
        if not device.startswith('cuda') or not torch.cuda.is_available():
            raise RuntimeError('Real ADM readiness requires the existing NVIDIA CUDA environment')
        self.data, self.device, self.seed = data, device, seed
        configure_reproducibility(seed)
        self.mean, self.std = load_stats(Path(inputs['stats_file']['path']))
        self.mean_t = torch.as_tensor(self.mean[None], device=device)
        self.std_t = torch.as_tensor(self.std[None], device=device)
        cfg = compose_cdm_config(POLICY['diffusion_steps'], device)
        point_path = Path(cfg.model.scene_model.pretrained_weight).resolve()
        if not point_path.is_file():
            raise FileNotFoundError('Missing pretrained scene encoder: ' + str(point_path))
        print('[MODEL 1/3] constructing ADM and frozen encoders', flush=True)
        from models.base import create_model_and_diffusion
        self.model, self.diffusion = create_model_and_diffusion(cfg, device=device)
        self.model.to(device).eval()
        self.loads = [strict_load(self.model, inputs[n]['path'])
                      for n in ('original_checkpoint', 'v5_checkpoint')]
        if self.diffusion.model_mean_type.name != 'START_X' or self.diffusion.num_timesteps != 500:
            raise ValueError('Expected START_X, 500-step diffusion')
        print('[MODEL 2/3] checkpoint coverage and finite tensors verified', flush=True)
        self.encoder_weight = dict(path=str(point_path), sha256=sha(point_path))
        self.base_digest = frozen_digest(self.model)
        self.source_root = inside(data.root, data.index['source_contact_index']).parent
        source = json.loads((self.source_root / 'index.json').read_text())
        self.motions = {m['id']: m for m in source['motions']}
        self.modules = None

    def install(self):
        self.modules = install_lora(self.model, POLICY['rank'], POLICY['alpha'], dropout=0.)
        if len(self.modules) != 31:
            raise ValueError('Expected 31 ADM LoRA modules; inspect architecture before proceeding')
        set_frozen_base_eval_lora_train(self.model)
        assert_frozen(self.model)
        self.frozen_after_install = frozen_digest(self.model)
        print('[MODEL 3/3] fresh rank-16 LoRA; all base/scene/text parameters frozen', flush=True)

    def predict(self, index, timestep, noise_seed):
        x, kw = self.data.batch([index], self.mean, self.std, allow_research=True)
        target = torch.from_numpy(x).to(self.device)
        kw = {k: (v if k == 'c_text' else torch.from_numpy(v).to(self.device)) for k, v in kw.items()}
        t = torch.tensor([timestep], device=self.device, dtype=torch.long)
        noise = deterministic_noise(target.shape, noise_seed, self.device)
        prediction = predict_xstart(self.model, self.diffusion, target, t, kw, noise)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError('Nonfinite ADM forward: ' + self.data.samples[index]['id'])
        return prediction, target

    def objective(self, prediction, target, index):
        physical = prediction * self.std_t + self.mean_t  # no clamp during training
        truth = target * self.std_t + self.mean_t
        row = self.data.samples[index]
        masks = []
        for mid in row['source_motion_ids']:
            with np.load(inside(self.source_root, self.motions[mid]['file']), allow_pickle=False) as z:
                mask = torch.as_tensor(z['affordance'][None] >= POLICY['active_threshold'], device=self.device)
            if not mask.any():
                raise ValueError('Empty per-motion active support: ' + mid)
            masks.append(mask)
        return balanced_objective(prediction, target, physical, truth, masks)

    def update(self, optimizer, indices, step):
        set_frozen_base_eval_lora_train(self.model)
        optimizer.zero_grad(set_to_none=True)
        totals = {}
        for slot, index in enumerate(indices):
            rng = np.random.default_rng(self.seed + step * 1009 + slot)
            timestep = int(rng.integers(0, 500))
            prediction, target = self.predict(index, timestep, self.seed + step * 10007 + slot)
            loss, parts = self.objective(prediction, target, index)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            (loss / len(indices)).backward()
            for k, value in parts.items():
                totals[k] = totals.get(k, 0.) + float(value.detach()) / len(indices)
        named = lora_named_parameters(self.model)
        grads = [p.grad for p in named.values() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise FloatingPointError('Missing or nonfinite LoRA gradient')
        norm = torch.nn.utils.clip_grad_norm_(list(named.values()), POLICY['grad_clip'])
        if not torch.isfinite(norm) or float(norm) <= 0:
            raise RuntimeError('Nonfinite or zero LoRA gradient norm')
        assert_frozen(self.model)
        before = state_digest(named)
        optimizer.step()
        if before == state_digest(named) or not all(torch.isfinite(p).all() for p in named.values()):
            raise RuntimeError('No finite LoRA parameter update')
        return dict(step=step, samples=[self.data.samples[i]['id'] for i in indices],
                    gradient_norm=float(norm), loss=totals)


def balanced_objective(prediction, target, physical, truth, instance_masks):
    """Fit all-instance union; balance active instances, not TV-selected labels.

    Background means low contact-GT values, NOT semantically forbidden furniture.
    Each instance mask supervises the UNION value to avoid overlapping-motion conflicts.
    """
    if prediction.shape != target.shape or truth.shape != physical.shape:
        raise ValueError('Objective shape mismatch')
    if not instance_masks:
        raise ValueError('No instance support')
    error = (physical - truth).square()
    active = []
    for mask in instance_masks:
        if mask.shape != error.shape or not mask.any():
            raise ValueError('Invalid instance support')
        active.append(error[mask].mean())
    background = truth <= POLICY['background_threshold']
    background_loss = error[background].mean() if background.any() else error.sum() * 0.
    parts = dict(normalized_mse=(prediction-target).square().mean(),
                 instance_active=torch.stack(active).mean(), background=background_loss,
                 range=(torch.relu(-physical).square() + torch.relu(physical-1.).square()).mean())
    loss = sum(POLICY['loss_weights'][k] * v for k, v in parts.items())
    return loss, parts
