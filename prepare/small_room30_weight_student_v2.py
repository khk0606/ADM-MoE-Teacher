"""Predicted semantic context slots + independent two-expert weighting branches.

Only XYZRGB, frozen CLIP text, observed history and frozen Teacher A in forward.
No annotation masks/IDs, target anchors, room IDs or prompt IDs are inputs.
"""
import torch
from torch import nn
import torch.nn.functional as F
from small_room30_weight_student import Branch, mlp, balanced_bce_logits

SCHEMA = 'small_room30_weight_student_v2'


def geometric_features(points):
    """Permutation-equivariant multiscale local shape from XYZ alone.

    Integer voxel keys are geometry partitions, NOT exported instance IDs.
    index_add is supported on the original training stack (no scatter_reduce).
    """
    batch_features = []
    for cloud in points:
        xyz = cloud[:, :3]
        descriptors = [cloud]
        for scale in (.25, .6, 1.2):
            _, inverse = torch.unique(torch.floor(xyz / scale).long(), dim=0, return_inverse=True)
            count = xyz.new_zeros(int(inverse.max()) + 1).index_add(0, inverse, xyz.new_ones(len(xyz)))
            sums = xyz.new_zeros(len(count), 3).index_add(0, inverse, xyz)
            mean = sums / count[:, None]
            second = xyz.new_zeros(len(count), 3).index_add(0, inverse, xyz.square()) / count[:, None]
            std = (second - mean.square()).clamp_min(0).sqrt()
            descriptors.extend(((xyz-mean[inverse])/scale, std[inverse]/scale,
                                (count[inverse].log1p()/8)[:, None]))
        batch_features.append(torch.cat(descriptors, -1))
    return torch.stack(batch_features)


class SceneEncoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.local = mlp(27, width, width)
        self.fuse = mlp(3*width, width, width)

    def forward(self, descriptors):
        local = self.local(descriptors)
        context = torch.cat((local.mean(1), local.max(1).values), -1)
        return self.fuse(torch.cat((local, context[:, None].expand(-1, local.size(1), -1)), -1))


class WeightStudentV2(nn.Module):
    def __init__(self, text_dim=512, width=64):
        super().__init__()
        self.config = dict(text_dim=text_dim, width=width)
        self.relation_scene = SceneEncoder(width)
        self.text_encoder = mlp(text_dim, width, width)
        self.purpose_classifier = mlp(text_dim, width, 3)
        self.semantic_head = mlp(width, width, 6)
        self.candidate_head = mlp(width, width, 1)
        self.anchor_head = mlp(width, width, 2)
        self.presence_head = mlp(width, width, 3)
        self.relation = Branch(width + 19, 2*width, width)
        self.history_scene = SceneEncoder(width)
        self.history_encoder = nn.GRU(6, width, batch_first=True)
        self.history = Branch(width + 5, 2*width, width)

    def forward(self, points, text_features, observed_history, teacher_a):
        if points.ndim != 3 or points.shape[-1] != 6: raise ValueError('Expected [B,N,6] points')
        b, n, _ = points.shape
        if (teacher_a.shape != (b,n,6) or observed_history.shape != (b,8,6)
                or text_features.shape != (b,self.config['text_dim'])): raise ValueError('Wrong input shape')
        if any(not torch.isfinite(x).all() for x in (points, text_features, observed_history, teacher_a)):
            raise ValueError('Nonfinite input')
        if ((points[..., 3:] < 0) | (points[..., 3:] > 1)).any(): raise ValueError('RGB must be [0,1]')
        if not torch.allclose(observed_history[...,4:].norm(dim=-1), torch.ones_like(observed_history[...,0]), atol=1e-4):
            raise ValueError('Expected unit body heading')
        descriptors = geometric_features(points)
        r = self.relation_scene(descriptors)
        text = self.text_encoder(text_features)
        semantic_logits = self.semantic_head(r)
        semantic = semantic_logits.softmax(-1)
        purpose_class_logits = self.purpose_classifier(text_features)
        purpose_classes = purpose_class_logits.sigmoid()
        candidate_logits = self.candidate_head(r).squeeze(-1)
        point_anchor = points[..., :2] + self.anchor_head(r)
        presence_logits = self.presence_head(r.mean(1))
        # Three distinct predicted slots: TV, desk, whiteboard. A desk and board
        # are never averaged into one empty-space centroid. Supported dataset
        # has one object/category; data preflight rejects multiples explicitly.
        attention = semantic[..., 3:6].pow(4)
        attention = attention / attention.sum(1, keepdim=True).clamp_min(1e-8)
        context_xy = torch.einsum('bnk,bnd->bkd', attention, point_anchor)
        relative = point_anchor[:, :, None] - context_xy[:, None]
        distances = relative.norm(dim=-1)
        active = purpose_classes * presence_logits.sigmoid()
        local = torch.cat((r, semantic, active[:,None].expand(-1,n,-1),
                           distances/6, relative.flatten(-2)/6, candidate_logits.sigmoid()[...,None]), -1)
        wr, pir = self.relation(local, torch.cat((r.mean(1), text), -1))
        h = self.history_scene(descriptors)
        _, hidden = self.history_encoder(observed_history)
        delta = points[..., :2] - observed_history[:, -1, None, :2]
        direction = F.normalize(delta, dim=-1)
        heading = (direction * observed_history[:, -1, None, 4:6]).sum(-1, keepdim=True)
        velocity = (direction * observed_history[:, -1, None, 2:4]).sum(-1, keepdim=True)
        wh, pih = self.history(torch.cat((h, delta, delta.norm(dim=-1,keepdim=True), heading, velocity), -1),
                               torch.cat((h.mean(1), hidden[-1]), -1))
        w = .6*wr + .4*wh
        purpose_probability = (semantic[...,3:6]*purpose_classes[:,None]).sum(-1)
        return dict(a_w=teacher_a.detach()*w[...,None], w=w, w_r=wr, w_h=wh, pi_r=pir, pi_h=pih,
            semantic_logits=semantic_logits, purpose_class_logits=purpose_class_logits,
            candidate_logits=candidate_logits, presence_logits=presence_logits,
            point_anchor=point_anchor, context_xy=context_xy, purpose_probability=purpose_probability)


def candidate_mean(value, masks):
    return (value[:,None]*masks).sum(-1)/masks.sum(-1).clamp_min(1)


def region_loss(point_loss, labels):
    # Equal weighting of candidate objects, noncandidate furniture, environment.
    masks = labels['candidate_masks'].bool()
    groups = [masks[:,i] for i in range(masks.size(1))]
    groups += [labels['noncandidate_mask'].bool(), ~labels['anchor_mask'].bool()]
    parts = [point_loss[g].mean() for g in groups if g.any()]
    return torch.stack(parts).mean()


def final_candidate_scores(a_w, teacher, masks):
    # Compare a candidate's contact-bearing points; identical frozen support for
    # predictions/targets and paired conditions. Never used to change a map.
    support = teacher.detach().clamp_min(0).max(-1).values
    weighted = support[:,None] * masks
    scores = (a_w.max(-1).values[:,None]*weighted).sum(-1)/weighted.sum(-1).clamp_min(1e-6)
    return scores


def pair_contrast(pred, target, indices):
    if not indices: return pred.sum()*0
    return torch.stack([F.mse_loss(pred[i]-pred[j], target[i]-target[j]) for i,j in indices]).mean()


def student_losses_v2(out, teacher, labels, prompt_pairs=(), history_pairs=()):
    masks = labels['candidate_masks'].bool()
    rmean = candidate_mean(out['w_r'], masks); hmean = candidate_mean(out['w_h'], masks)
    rtarget = labels['candidate_relation']; htarget = labels['history_scores']
    semantic = labels['semantic'].long()
    ce = F.cross_entropy(out['semantic_logits'].transpose(1,2), semantic, reduction='none')
    semantics = torch.stack([ce[semantic==c].mean() for c in range(6) if (semantic==c).any()]).mean()
    purpose = F.binary_cross_entropy_with_logits(out['purpose_class_logits'], labels['purpose_classes'])
    purpose = purpose + F.binary_cross_entropy_with_logits(out['presence_logits'], labels['presence'])
    purpose = purpose + balanced_bce_logits(out['candidate_logits'], masks.any(1).float())
    purpose = purpose + region_loss((out['purpose_probability']-labels['purpose_mask']).square(), labels)
    am = labels['anchor_mask'].bool()
    anchors = F.smooth_l1_loss(out['point_anchor'][am], labels['point_anchor'][am])
    present = labels['presence'].bool()
    anchors = anchors + F.smooth_l1_loss(out['context_xy'][present], labels['context_anchors'][present])
    auxiliary = 2*semantics + 2*purpose + anchors
    relation = F.binary_cross_entropy(rmean.clamp(1e-6,1-1e-6), rtarget)
    relation = relation + region_loss((out['w_r']-labels['relation_spatial']).square(), labels)
    history = F.binary_cross_entropy(hmean.clamp(1e-6,1-1e-6), htarget)
    history = history + region_loss((out['w_h']-labels['history_spatial']).square(), labels)
    target_w = .6*labels['relation_spatial'] + .4*labels['history_spatial']
    target_map = teacher.detach()*target_w[...,None]
    map_loss = region_loss((out['a_w']-target_map).square().mean(-1), labels)
    pair = pair_contrast(rmean, rtarget, prompt_pairs) + pair_contrast(hmean, htarget, history_pairs)
    score = final_candidate_scores(out['a_w'], teacher, masks)
    target_score = final_candidate_scores(target_map, teacher, masks)
    pair = pair + pair_contrast(score, target_score, list(prompt_pairs)+list(history_pairs))
    # Batch mean balances aggregate usage without requiring every example to
    # use experts equally; specialization is allowed.
    router = sum((pi.mean(0)*(pi.mean(0).clamp_min(1e-8).log()+.6931471805599453)).sum()
                 for pi in (out['pi_r'],out['pi_h']))
    total = auxiliary + 2*relation + 2*history + 4*map_loss + 2*pair + .05*router
    return dict(total=total, auxiliary=auxiliary, semantic=semantics, purpose=purpose, anchor=anchors,
                relation=relation, history=history, map=map_loss, pair=pair, router=router)
