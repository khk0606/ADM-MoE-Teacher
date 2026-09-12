"""Direct two-branch weighting student. No Teacher, GT or planner in forward.

Import this module directly from prepare, avoiding legacy models/__init__.py.
The frozen Teacher is still required at inference: this is not Teacher removal.
"""
import torch
from torch import nn
import torch.nn.functional as F

SCHEMA = 'small_room30_weight_student_v1'


def mlp(inp, width, out):
    return nn.Sequential(nn.Linear(inp, width), nn.SiLU(), nn.Linear(width, out))


class Branch(nn.Module):
    def __init__(self, local_dim, context_dim, width):
        super().__init__()
        self.encoder = mlp(local_dim + context_dim, width, width)
        self.experts = nn.ModuleList([mlp(width, width, 1) for _ in range(2)])
        self.router = mlp(context_dim, width, 2)

    def forward(self, local, context):
        features = self.encoder(torch.cat((local, context[:, None].expand(-1, local.size(1), -1)), -1))
        logits = torch.cat([expert(features) for expert in self.experts], -1)
        pi = self.router(context).softmax(-1)
        w = (logits * pi[:, None]).sum(-1).sigmoid()
        return w, pi


class WeightStudent(nn.Module):
    def __init__(self, text_dim=512, width=64):
        super().__init__()
        self.config = dict(text_dim=text_dim, width=width)
        # Entirely independent branch parameters. Text never enters history.
        self.relation_scene = mlp(6, width, width)
        self.text_encoder = mlp(text_dim, width, width)
        self.purpose_head = mlp(2 * width, width, 1)
        self.candidate_head = mlp(width, width, 1)
        self.relation = Branch(width + 5, 2 * width, width)
        self.history_scene = mlp(6, width, width)
        self.history_encoder = nn.GRU(6, width, batch_first=True)
        self.history = Branch(width + 5, 2 * width, width)

    def forward(self, points, text_features, observed_history, teacher_a):
        if points.ndim != 3 or points.shape[-1] != 6:
            raise ValueError('points must be [B,N,6], XYZ metres and RGB [0,1]')
        b, n, _ = points.shape
        if (teacher_a.shape != (b, n, 6) or observed_history.shape != (b, 8, 6)
                or text_features.shape != (b, self.config['text_dim'])):
            raise ValueError('Wrong Teacher/text/observed-history shape')
        for value in (points, text_features, observed_history, teacher_a):
            if not torch.isfinite(value).all():
                raise ValueError('Nonfinite input')
        if torch.any((points[..., 3:] < 0) | (points[..., 3:] > 1)):
            raise ValueError('RGB must be divided by 255 exactly once')
        if not torch.allclose(observed_history[..., 4:].norm(dim=-1),
                              torch.ones_like(observed_history[..., 0]), atol=1e-4):
            raise ValueError('Observed body headings must be unit XY vectors')
        a = teacher_a.detach()  # No gradient or optimizer access to Teacher.
        r = self.relation_scene(points)
        text = self.text_encoder(text_features)
        purpose_logits = self.purpose_head(torch.cat((r, text[:, None].expand(-1, n, -1)), -1)).squeeze(-1)
        candidate_logits = self.candidate_head(r).squeeze(-1)
        # Predicted soft localization only; no instance IDs/anchors/masks supplied.
        context_attention = purpose_logits.softmax(-1)
        context_xy = (points[..., :2] * context_attention[..., None]).sum(1)
        relative = points[..., :2] - context_xy[:, None]
        relation_local = torch.cat((r, relative, relative.norm(dim=-1, keepdim=True),
                                    purpose_logits.sigmoid()[..., None], candidate_logits.sigmoid()[..., None]), -1)
        wr, pir = self.relation(relation_local, torch.cat((r.mean(1), text), -1))
        h = self.history_scene(points)
        _, hidden = self.history_encoder(observed_history)
        delta = points[..., :2] - observed_history[:, -1, None, :2]
        direction = F.normalize(delta, dim=-1)
        heading = (direction * observed_history[:, -1, None, 4:6]).sum(-1, keepdim=True)
        velocity = (direction * observed_history[:, -1, None, 2:4]).sum(-1, keepdim=True)
        history_local = torch.cat((h, delta, delta.norm(dim=-1, keepdim=True), heading, velocity), -1)
        wh, pih = self.history(history_local, torch.cat((h.mean(1), hidden[-1]), -1))
        w = 0.6 * wr + 0.4 * wh  # Fixed, not learned router probabilities.
        return dict(a_w=a * w[..., None], w=w, w_r=wr, w_h=wh,
                    pi_r=pir, pi_h=pih, purpose_logits=purpose_logits,
                    candidate_logits=candidate_logits, context_xy=context_xy)


def balanced_bce_logits(logits, target):
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    parts = [loss[target > .5], loss[target <= .5]]
    return torch.stack([part.mean() for part in parts if part.numel()]).mean()


def student_losses(out, teacher_a, labels):
    """Labels used only here. One optimizer jointly updates both independent MoEs.

    Contact GT defines spatial target support; it is not fed to inference.
    History labels are an observed-geometry heuristic, not human preference GT.
    """
    masks = labels['candidate_masks'].bool()  # [B,C,N], padded candidates unsupported in this pilot batch=1.
    rmean = (out['w_r'][:, None] * masks).sum(-1) / masks.sum(-1).clamp_min(1)
    hmean = (out['w_h'][:, None] * masks).sum(-1) / masks.sum(-1).clamp_min(1)
    selected = labels['selected'].long()
    rtarget = F.one_hot(selected, masks.shape[1]).float()
    lr = F.binary_cross_entropy(rmean.clamp(1e-6, 1-1e-6), rtarget)
    chosen = rmean.gather(1, selected[:, None])
    others = ~rtarget.bool()
    if others.any():
        lr = lr + F.relu(.15 - chosen.expand_as(rmean)[others] + rmean[others]).mean()
    lr = lr + balanced_bce_logits(out['purpose_logits'], labels['purpose_mask'])
    lr = lr + balanced_bce_logits(out['candidate_logits'], masks.any(1).float())
    lh = F.mse_loss(hmean, labels['history_scores'])
    target_w = .6 * labels['relation_spatial'] + .4 * labels['history_spatial']
    target_map = teacher_a.detach() * target_w[..., None]
    support = labels['contact_support'][..., None] + .05
    lmap = ((out['a_w'] - target_map).square() * support).sum() / (6 * support.sum())
    # Batch=1: entropy regularization discourages a permanently unused expert.
    reg = sum((pi * (pi.clamp_min(1e-8).log() + 0.6931471805599453)).sum(-1).mean()
              for pi in (out['pi_r'], out['pi_h']))
    total = lr + lh + 2 * lmap + .01 * reg
    return dict(total=total, relation=lr, history=lh, map=lmap, router=reg)
