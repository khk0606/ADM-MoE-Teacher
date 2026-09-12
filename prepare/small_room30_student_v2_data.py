"""Corrected supervision only; keep v1 data/Teacher integrity checks unchanged."""
from functools import lru_cache
import numpy as np
from small_room30_student_data import StudentData, observed_scores

CATEGORIES = ('environment', 'chair', 'bed', 'tv', 'desk', 'whiteboard')
PURPOSE_CATEGORIES = ('tv', 'desk', 'whiteboard')
TARGET_POLICY = dict(single_relation_positive=.95, relation_negative=.05,
                     ambiguous_relation=.45, history_temperature=.35,
                     history_min=.05, history_max=.95, noncandidate_object=.02)


def eligible_candidates(anchors, contexts):
    """Union of nearest candidates PER purpose instance, not one global winner.

    Used for labels only. Equal-distance ties remain eligible. No room-ID rules.
    """
    distances = np.linalg.norm(anchors[:, None] - contexts[None], axis=-1)
    return (distances <= distances.min(0, keepdims=True) + 1e-5).any(1)


def calibrated_history(history, anchors):
    # Same observed-only distance/heading/velocity score; sharpen relative
    # preferences without using eventual target IDs or future motion frames.
    original = np.clip(observed_scores(history, anchors), 1e-6, 1-1e-6)
    logits = np.log(original / (1-original)) / TARGET_POLICY['history_temperature']
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    return (.05 + .90 * probabilities).astype(np.float32)


def corrected_targets(contacts, masks, candidate_r, candidate_h, noncandidate):
    affinity = np.maximum(contacts.max(-1), 0)
    denom = np.maximum(affinity.sum(0), 1e-6)
    # Weak contact halo, not a second broad object-selection target.
    rs = .15 * (affinity * candidate_r[:, None]).sum(0) / denom
    hs = .15 * (affinity * candidate_h[:, None]).sum(0) / denom
    for index, mask in enumerate(masks):
        rs[mask] = candidate_r[index]; hs[mask] = candidate_h[index]
    rs[noncandidate] = .02; hs[noncandidate] = .02
    return rs.astype(np.float32), hs.astype(np.float32)


class StudentDataV2(StudentData):
    def __init__(self, dataset, teacher_summary):
        super().__init__(dataset, teacher_summary)
        self.binding.update(schema='small_room30_weight_student_data_v2',
            target_policy=TARGET_POLICY,
            relation_target='Union of nearest candidate per compatible purpose instance; ambiguous candidates tied at 0.45',
            history_target='Observed-only geometric proxy, temperature 0.35; not human preference GT',
            formula_limit='At fixed P,q, history changes W by at most 0.4; no hard selection or normalization')

    @lru_cache(maxsize=128)
    def corrected_labels(self, scene, prompt, motion):
        points, ids, instances, names, masks, anchors, contacts, histories, relations = self.geometry(scene)
        relation = next(r for r in relations if r['prompt_id'] == prompt)
        contexts = np.asarray([[instances[str(i)]['anchor_coordinate_local_xyz'][k] for k in ('x', 'z')]
                               for i in relation['purpose_instance_ids']], np.float32)
        eligible = eligible_candidates(anchors, contexts)
        if not eligible.any(): raise ValueError('No eligible candidate')
        cr = np.where(eligible, .95 if eligible.sum() == 1 else .45, .05).astype(np.float32)
        ch = calibrated_history(histories[motion], anchors)
        semantic = np.full(len(points), -1, np.int64)
        point_anchor = points[:, :2].copy()
        anchor_mask = np.zeros(len(points), bool)
        presence = np.zeros(3, np.float32)
        context_anchors = np.zeros((3, 2), np.float32)
        for instance in instances.values():
            category = instance['category']
            if category not in CATEGORIES: raise ValueError('Unmapped semantic category: ' + category)
            mask = ids == instance['instance_id']
            semantic[mask] = CATEGORIES.index(category)
            if category != 'environment':
                xy = [instance['anchor_coordinate_local_xyz'][k] for k in ('x', 'z')]
                point_anchor[mask] = xy; anchor_mask[mask] = True
            if category in PURPOSE_CATEGORIES:
                j = PURPOSE_CATEGORIES.index(category)
                if presence[j]:
                    raise ValueError('V2 supports one instance per context category; do not silently average multiples')
                presence[j] = 1; context_anchors[j] = xy
        if (semantic < 0).any(): raise ValueError('Unlabeled point')
        noncandidate = anchor_mask & ~masks.any(0)
        rs, hs = corrected_targets(contacts, masks, cr, ch, noncandidate)
        purpose_classes = np.asarray([c in relation['purpose_categories'] for c in PURPOSE_CATEGORIES], np.float32)
        return dict(candidate_masks=masks.copy(), candidate_relation=cr, history_scores=ch,
            eligible=eligible, purpose_classes=purpose_classes, semantic=semantic,
            point_anchor=point_anchor, anchor_mask=anchor_mask, presence=presence,
            context_anchors=context_anchors, relation_spatial=rs, history_spatial=hs,
            noncandidate_mask=noncandidate,
            purpose_mask=np.isin(ids, [instances[str(i)]['instance_id'] for i in relation['purpose_instance_ids']]).astype(np.float32))

    def example(self, record):
        # Retain full Teacher/GT/point-order/hash validation from immutable v1.
        inputs, _ = super().example(record)
        labels = self.corrected_labels(record['scene_id'], record['prompt_id'], record['history_motion_id'])
        return inputs, {k: v.copy() for k, v in labels.items()}
