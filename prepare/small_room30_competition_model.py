"""Learned candidate perception, two independent MoEs, explicit competition.

Inference takes only P (XYZRGB), text features, observed 8-frame history, A.
Candidate groups/anchors and spatial support are predicted, never GT-supplied.
"""
import torch
from torch import nn
import torch.nn.functional as F
from small_room30_weight_student import Branch,mlp,balanced_bce_logits
from small_room30_weight_student_v2 import SceneEncoder,geometric_features,region_loss
from small_room30_competition_data import POLICY

SCHEMA='small_room30_competition_student_v1'


def candidate_slots(votes,probability,iterations=5):
    """Two soft clusters of predicted candidate anchor votes, without IDs.

    Geometry-only farthest seeding; differentiable soft updates. GT matching is
    a separate loss operation. Returned slot names do not identify real objects.
    """
    b,n,_=votes.shape
    mass=probability.pow(4).clamp_min(1e-8)
    mean=(votes*mass[...,None]).sum(1)/mass.sum(1)[:,None]
    first=((votes-mean[:,None]).square().sum(-1)*mass).argmax(1)
    batch=torch.arange(b,device=votes.device)
    c0=votes[batch,first]
    second=((votes-c0[:,None]).square().sum(-1)*mass).argmax(1)
    centers=torch.stack((c0,votes[batch,second]),1)
    for _ in range(iterations):
        distance=(votes[:,:,None]-centers[:,None]).square().sum(-1)
        assignment=(-distance/(.35**2)).softmax(-1)
        attention=assignment*mass[...,None]
        centers=torch.einsum('bnc,bnd->bcd',attention,votes)/attention.sum(1).clamp_min(1e-8)[...,None]
    order=(centers[...,0]+1e-4*centers[...,1]).argsort(-1)
    centers=centers.gather(1,order[...,None].expand(-1,-1,2))
    assignment=(-(votes[:,:,None]-centers[:,None]).square().sum(-1)/(.35**2)).softmax(-1)
    attention=assignment*mass[...,None]
    pool=attention/attention.sum(1,keepdim=True).clamp_min(1e-8)
    return centers,assignment,pool


def fuse_candidates(r,h,temperature=.1):
    if temperature<=0:raise ValueError('Selection temperature must be positive')
    score=.7*r+.3*h
    return score,(score/temperature).softmax(-1)


def spatial_weight(support,selection):
    # The final column is null/background, with weight zero. Full spatial
    # support is inferred; no GT floor mask or arbitrary .15 suppression here.
    return (support[...,:2]*selection[:,None]).sum(-1)


class CompetitionStudent(nn.Module):
    def __init__(self,text_dim=512,width=64,selection_temperature=.1):
        super().__init__()
        if selection_temperature!=POLICY['selection_temperature']:
            raise ValueError('This release pins selection temperature to 0.1; change policy/schema for another experiment')
        self.config=dict(text_dim=text_dim,width=width,selection_temperature=selection_temperature)
        self.perception=SceneEncoder(width)
        self.semantic_head=mlp(width,width,6)
        self.candidate_head=mlp(width,width,1)
        self.anchor_head=mlp(width,width,2)
        self.presence_head=mlp(width,width,3)
        self.purpose_classifier=mlp(text_dim,width,3)
        self.relation_scene=SceneEncoder(width)
        self.text_encoder=mlp(text_dim,width,width)
        self.relation=Branch(width+13,2*width,width)
        self.history_scene=SceneEncoder(width)
        self.history_encoder=nn.GRU(6,width,batch_first=True)
        self.history=Branch(width+6,2*width,width)
        self.support_head=mlp(2*width+4,width,1)
        self.null_head=mlp(width,width,1)

    def forward(self,points,text_features,observed_history,teacher_a):
        if points.ndim!=3 or points.shape[-1]!=6:raise ValueError('Expected [B,N,6] XYZRGB')
        b,n,_=points.shape
        if n<2 or teacher_a.shape!=(b,n,6) or observed_history.shape!=(b,8,6) or text_features.shape!=(b,self.config['text_dim']):
            raise ValueError('Wrong input shapes')
        if any(not torch.isfinite(x).all() for x in (points,text_features,observed_history,teacher_a)):
            raise ValueError('Nonfinite input')
        if ((points[...,3:]<0)|(points[...,3:]>1)).any():raise ValueError('RGB must be [0,1]')
        if not torch.allclose(observed_history[...,4:].norm(dim=-1),torch.ones_like(observed_history[...,0]),atol=1e-4):
            raise ValueError('History heading must be unit XY')
        desc=geometric_features(points)
        perception=self.perception(desc)
        semantic_logits=self.semantic_head(perception);semantic=semantic_logits.softmax(-1)
        candidate_logits=self.candidate_head(perception).squeeze(-1)
        candidate_probability=candidate_logits.sigmoid()
        votes=points[...,:2]+self.anchor_head(perception)
        centers,assignment,pool=candidate_slots(votes,candidate_probability)
        slot_features=torch.einsum('bnc,bnd->bcd',pool,perception)
        context_attention=semantic[...,3:6].pow(4)
        context_attention=context_attention/context_attention.sum(1,keepdim=True).clamp_min(1e-8)
        context_xy=torch.einsum('bnk,bnd->bkd',context_attention,votes)
        presence_logits=self.presence_head(perception.mean(1))
        purpose_logits=self.purpose_classifier(text_features)
        active=presence_logits.sigmoid()*purpose_logits.sigmoid()
        relative=centers[:,:,None]-context_xy[:,None]
        distance=relative.norm(dim=-1)
        per_context=(-distance/POLICY['relation_temperature_m']).softmax(1)*active[:,None]
        geometric_r=per_context.max(-1).values
        rf=self.relation_scene(desc);rf_slot=torch.einsum('bnc,bnd->bcd',pool,rf)
        text=self.text_encoder(text_features)
        relation_local=torch.cat((rf_slot,distance/6,relative.flatten(-2)/6,
            active[:,None].expand(-1,2,-1),geometric_r[...,None]),-1)
        r,pi_r=self.relation(relation_local,torch.cat((rf.mean(1),text),-1))
        hf=self.history_scene(desc);hf_slot=torch.einsum('bnc,bnd->bcd',pool,hf)
        _,history_hidden=self.history_encoder(observed_history)
        delta=centers-observed_history[:,-1,None,:2]
        direction=F.normalize(delta,dim=-1)
        heading=(direction*observed_history[:,-1,None,4:6]).sum(-1)
        velocity=observed_history[:,-1,2:4];speed=velocity.norm(dim=-1)
        approach=(direction*F.normalize(velocity,dim=-1)[:,None]).sum(-1)*speed.clamp(max=1)[:,None]
        geometric_h=((-delta.norm(dim=-1)+.5*heading+.5*approach)/POLICY['history_temperature']).softmax(-1)
        history_local=torch.cat((hf_slot,delta/6,delta.norm(dim=-1,keepdim=True)/6,
            heading[...,None],approach[...,None],geometric_h[...,None]),-1)
        h,pi_h=self.history(history_local,torch.cat((hf.mean(1),history_hidden[-1]),-1))
        score,selection=fuse_candidates(r,h,self.config['selection_temperature'])
        support_delta=points[:,:,None,:2]-centers[:,None]
        support_features=torch.cat((perception[:,:,None].expand(-1,-1,2,-1),
            slot_features[:,None].expand(-1,n,-1,-1),support_delta/6,
            support_delta.norm(dim=-1,keepdim=True)/6,
            candidate_probability[:,:,None,None].expand(-1,-1,2,-1)),-1)
        support_logits=torch.cat((self.support_head(support_features).squeeze(-1),self.null_head(perception)),-1)
        learned_support=support_logits.softmax(-1)
        candidate_support=torch.cat((assignment,torch.zeros_like(assignment[...,:1])),-1)
        support=candidate_probability[...,None]*candidate_support+(1-candidate_probability[...,None])*learned_support
        w=spatial_weight(support,selection)
        return dict(a_w=teacher_a.detach()*w[...,None],w=w,selection=selection,score=score,
            relation_score=r,history_score=h,slot_xy=centers,assignment=assignment,support=support,
            relation_map=(support[...,:2]*r[:,None]).sum(-1),history_map=(support[...,:2]*h[:,None]).sum(-1),
            score_map=(support[...,:2]*score[:,None]).sum(-1),pi_r=pi_r,pi_h=pi_h,
            semantic_logits=semantic_logits,candidate_logits=candidate_logits,point_anchor=votes,
            context_xy=context_xy,presence_logits=presence_logits,purpose_class_logits=purpose_logits,
            purpose_probability=(semantic[...,3:6]*purpose_logits.sigmoid()[:,None]).sum(-1))


def match_slots(predicted,truth):
    direct=(predicted-truth).square().sum((1,2))
    reverse=(predicted.flip(1)-truth).square().sum((1,2))
    base=torch.tensor([0,1],device=predicted.device).expand(len(predicted),-1)
    return torch.where((reverse<direct)[:,None],base.flip(1),base)


def align(out,labels):
    # Annotation-based assignment is confined to loss/evaluation, never forward.
    order=match_slots(out['slot_xy'].detach(),labels['candidate_anchors'])
    b,n,_=out['support'].shape
    aligned={}
    for key in ('relation_score','history_score','score','selection'):
        aligned[key]=out[key].gather(1,order)
    aligned['slot_xy']=out['slot_xy'].gather(1,order[...,None].expand(-1,-1,2))
    aligned['support']=torch.cat((out['support'][...,:2].gather(2,order[:,None].expand(-1,n,-1)),out['support'][...,2:]),-1)
    return aligned


def contrast(pred,target,pairs):
    if not pairs:return pred.sum()*0
    return torch.stack([F.mse_loss(pred[i]-pred[j],target[i]-target[j]) for i,j in pairs]).mean()


def competition_losses(out,teacher,labels,prompt_pairs=(),history_pairs=()):
    aligned=align(out,labels);semantic=labels['semantic'].long()
    ce=F.cross_entropy(out['semantic_logits'].transpose(1,2),semantic,reduction='none')
    semantic_loss=torch.stack([ce[semantic==c].mean() for c in range(6) if (semantic==c).any()]).mean()
    purpose=F.binary_cross_entropy_with_logits(out['purpose_class_logits'],labels['purpose_classes'])
    purpose+=F.binary_cross_entropy_with_logits(out['presence_logits'],labels['presence'])
    purpose+=balanced_bce_logits(out['candidate_logits'],labels['candidate_masks'].any(1).float())
    purpose+=region_loss((out['purpose_probability']-labels['purpose_mask']).square(),labels)
    am=labels['anchor_mask'].bool();present=labels['presence'].bool()
    anchors=F.smooth_l1_loss(out['point_anchor'][am],labels['point_anchor'][am])
    anchors+=F.smooth_l1_loss(out['context_xy'][present],labels['context_anchors'][present])
    slots=F.smooth_l1_loss(aligned['slot_xy'],labels['candidate_anchors'])
    support_ce=-(labels['support_target']*aligned['support'].clamp_min(1e-7).log()).sum(-1)
    support=region_loss(support_ce,labels)
    floor_weight=labels['floor_support_weight'];denom=floor_weight.sum(-1).clamp_min(1e-6)
    support+=((support_ce*floor_weight).sum(-1)/denom).mean()
    auxiliary=2*semantic_loss+2*purpose+anchors+3*slots+2*support
    relation=F.mse_loss(aligned['relation_score'],labels['candidate_relation'])
    history=F.mse_loss(aligned['history_score'],labels['history_scores'])
    choice=-(labels['selection_target']*aligned['selection'].clamp_min(1e-7).log()).sum(-1).mean()
    target_aw=teacher.detach()*labels['target_weight'][...,None]
    point_error=(out['a_w']-target_aw).square().mean(-1)
    map_loss=region_loss(point_error,labels)
    floor_error=(out['w']-labels['target_weight']).square()+2*point_error
    floor=((floor_error*floor_weight).sum(-1)/denom).mean()
    pairs=list(prompt_pairs)+list(history_pairs)
    paired=contrast(aligned['relation_score'],labels['candidate_relation'],prompt_pairs)
    paired+=contrast(aligned['history_score'],labels['history_scores'],history_pairs)
    paired+=contrast(aligned['selection'],labels['selection_target'],pairs)
    router=sum((pi.mean(0)*(pi.mean(0).clamp_min(1e-8).log()+.6931471805599453)).sum() for pi in (out['pi_r'],out['pi_h']))
    total=auxiliary+4*relation+4*history+3*choice+4*map_loss+4*floor+2*paired+.05*router
    return dict(total=total,auxiliary=auxiliary,semantic=semantic_loss,purpose=purpose,anchors=anchors,
        slots=slots,support=support,relation=relation,history=history,choice=choice,map=map_loss,
        floor=floor,paired=paired,router=router)
