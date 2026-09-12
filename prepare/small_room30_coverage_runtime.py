"""Surface coverage extension. Original uniform timestep sampling stays intact."""
import math
import torch
from small_room30_training_runtime import Runtime
from small_room30_coverage import POLICY, surface_support


def coverage_terms(physical, truth, surface_masks):
    if physical.shape != truth.shape or physical.ndim != 3 or physical.shape[0] != 1:
        raise ValueError('Expected single-sample physical maps')
    deficits = []
    for mask in surface_masks:
        if mask.shape != physical.shape[1:] or mask.dtype != torch.bool:
            raise ValueError('Bad coverage mask')
        core = mask.any(dim=1)
        if int(core.sum()) < POLICY['min_core_points']: raise ValueError('Tiny/empty surface mask')
        # Only motion-supported channels can count, not arbitrary channel activation.
        scores = physical[0].masked_fill(~mask, -torch.inf).max(dim=1).values[core]
        k = int(math.ceil(scores.numel() * POLICY['training_top_fraction']))
        # Top half must activate: cannot satisfy this by a single large outlier.
        best = torch.topk(scores, k=k).values
        deficits.append(torch.relu(POLICY['training_activation_margin'] - best).square().mean())
    if not deficits: raise ValueError('No furniture for coverage objective')
    values = torch.stack(deficits)
    bg = truth[0].max(dim=1).values <= POLICY['background_gt_threshold']
    scores = physical[0].max(dim=1).values
    background = torch.relu(scores[bg] - POLICY['background_margin']).square().mean() if bg.any() else scores.sum() * 0
    return dict(surface_mean=values.mean(), surface_worst=values.max(), point_background=background)


class CoverageRuntime(Runtime):
    def objective(self, prediction, target, index):
        loss, parts = super().objective(prediction, target, index)
        if self.data.samples[index]['action'] != 'sit': return loss, parts
        if not hasattr(self, '_coverage_masks'): self._coverage_masks = {}
        if index not in self._coverage_masks:
            masks, _, _ = surface_support(self.data, index)
            self._coverage_masks[index] = torch.as_tensor(masks, device=self.device, dtype=torch.bool)
        physical = prediction * self.std_t + self.mean_t
        truth = target * self.std_t + self.mean_t
        extra = coverage_terms(physical, truth, self._coverage_masks[index])
        loss = loss + POLICY['coverage_weight'] * (extra['surface_mean'] + extra['surface_worst'])
        loss = loss + POLICY['background_weight'] * extra['point_background']
        return loss, dict(parts, **extra)
