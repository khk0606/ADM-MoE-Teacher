"""Candidate competition supervision; geometry/GT remain training-only labels."""
from functools import lru_cache
import numpy as np
from small_room30_student_v2_data import StudentDataV2

POLICY = dict(relation_fraction=.7, history_fraction=.3, relation_temperature_m=.5,
    history_temperature=.35, selection_temperature=.1, candidates=2,
    context_categories=['tv','desk','whiteboard'], environment_support_multiplier=1.,
    scope='30-room development; two sittable candidates and at most one object per context category',
    formula='S_i=0.7*R_i+0.3*H_i; q=softmax(S/0.1); w_n=sum_i predicted_support[n,i]*q_i; A_w=A_raw*w')


def softmax(x, axis=-1):
    ex=np.exp(x-np.max(x,axis=axis,keepdims=True))
    return ex/ex.sum(axis=axis,keepdims=True)


def relation_scores(anchors, contexts, active):
    distances=np.linalg.norm(anchors[:,None]-contexts[None],axis=-1)
    per_context=softmax(-distances/POLICY['relation_temperature_m'],axis=0)
    per_context=per_context*np.asarray(active)[None]
    return per_context.max(-1).astype(np.float32), distances.astype(np.float32)


def history_scores(history, anchors):
    delta=anchors-history[-1,:2];distance=np.linalg.norm(delta,axis=1)
    direction=delta/np.maximum(distance[:,None],1e-6)
    velocity=history[-1,2:4];speed=float(np.linalg.norm(velocity))
    score=-distance+.5*(direction@history[-1,4:6])
    score+=.5*min(speed,1.)*(direction@(velocity/max(speed,1e-6)))
    return softmax(score/POLICY['history_temperature']).astype(np.float32)


def compete(r,h):
    score=.7*r+.3*h
    return score.astype(np.float32),softmax(score/POLICY['selection_temperature']).astype(np.float32)


def support_labels(points, contacts, candidate_masks, anchor_mask, noncandidate):
    affinity=np.maximum(contacts.max(-1),0).T  # N,C. All six channels, labels only.
    support=affinity/np.maximum(affinity.sum(-1,keepdims=True),1e-6)
    for i,mask in enumerate(candidate_masks):
        support[mask]=0;support[mask,i]=1
    support[noncandidate]=0
    null=(1-support.sum(-1,keepdims=True)).clip(0,1)
    support=np.concatenate((support,null),-1).astype(np.float32)
    environment=~anchor_mask
    if not environment.any():raise ValueError('Missing environment for floor check')
    floor=environment & (points[:,2]<=np.quantile(points[environment,2],.1)+.15)
    strength=affinity.max(-1).clip(0,1)
    supported=floor & (strength>=.1)
    if not supported.any():raise ValueError('No contact-supported floor; refusing silent omission')
    return support,floor,supported,(supported*strength).astype(np.float32)


class CompetitionData(StudentDataV2):
    def __init__(self,dataset,teacher_summary):
        super().__init__(dataset,teacher_summary)
        self.binding.update(schema='small_room30_competition_data_v1',target_policy=POLICY,
            relation_target='Per-purpose softmax of negative metric distance; maximum over matching purposes; no fixed 0.45 or 0.95 targets',
            history_target='Observed-only distance, heading and approach softmax; no future or eventual target ID in forward',
            formula_limit='Competition intentionally replaces old linear final weight; 0.7/0.3 combines candidate scores only')

    @lru_cache(maxsize=128)
    def corrected_labels(self,scene,prompt,motion):
        labels={k:v.copy() for k,v in super().corrected_labels(scene,prompt,motion).items()}
        points,_,_,names,masks,anchors,contacts,histories,_=self.geometry(scene)
        if len(names)!=2:raise ValueError('Competition v1 requires exactly two candidate objects')
        active=labels['purpose_classes']*labels['presence']
        if not active.any():raise ValueError('Purpose object absent')
        r,distances=relation_scores(anchors,labels['context_anchors'],active)
        h=history_scores(histories[motion],anchors)
        score,choice=compete(r,h)
        support,floor,supported,floor_weight=support_labels(points,contacts,masks,labels['anchor_mask'],labels['noncandidate_mask'])
        labels.update(candidate_relation=r,history_scores=h,score_target=score,selection_target=choice,
            candidate_anchors=anchors.copy(),distance_matrix=distances,support_target=support,
            target_weight=(support[:,:2]*choice[None]).sum(-1).astype(np.float32),
            relation_spatial=(support[:,:2]*r[None]).sum(-1).astype(np.float32),
            history_spatial=(support[:,:2]*h[None]).sum(-1).astype(np.float32),
            floor_mask=floor,floor_support_mask=supported,floor_support_weight=floor_weight)
        return labels
