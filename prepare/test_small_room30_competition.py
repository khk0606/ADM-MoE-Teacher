"""Structural, numerical and synthetic workflow tests; not trained quality proof."""
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from small_room30_competition_data import POLICY,relation_scores,history_scores,compete,support_labels
from small_room30_competition_model import CompetitionStudent,candidate_slots,spatial_weight,align,competition_losses
from train_small_room30_competition import save_model,load_model,train,evaluate,diagnostic,main
from small_room30_competition_review import validate_maps
from small_room30_adm_dataset import sha
from small_room30_student_data import TEXTS
from test_small_room30_student_v2 import fixture as old_fixture


def fixture():
    x,l=old_fixture()
    l['candidate_anchors']=torch.tensor([[[.5,.5],[3.,3.]]])
    r,h=l['candidate_relation'],l['history_scores']
    l['score_target']=.7*r+.3*h
    l['selection_target']=(l['score_target']/.1).softmax(-1)
    support=torch.zeros(1,64,3)
    support[:,:16,0]=1;support[:,16:32,1]=1;support[:,32:56,2]=1
    support[:,56:,0]=.75;support[:,56:,1]=.25
    l['support_target']=support
    l['target_weight']=spatial_weight(support,l['selection_target'])
    l['floor_support_mask']=torch.zeros(1,64,dtype=torch.bool);l['floor_support_mask'][:,56:]=True
    l['floor_mask']=l['floor_support_mask'].clone()
    l['floor_support_weight']=l['floor_support_mask'].float()
    return x,l


class CompetitionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(14)
        self.x,self.l=fixture();self.m=CompetitionStudent(width=16)

    def out(self,**kw):return self.m(**dict(self.x,**kw))

    def test_exact_formula_and_raw_teacher(self):
        a=self.x['teacher_a'].detach().clone();a[...,0]=-.1;a[...,1]=1.2
        o=self.out(teacher_a=a)
        self.assertTrue(torch.equal(o['score'],.7*o['relation_score']+.3*o['history_score']))
        self.assertTrue(torch.equal(o['selection'],(o['score']/.1).softmax(-1)))
        self.assertTrue(torch.equal(o['w'],spatial_weight(o['support'],o['selection'])))
        self.assertTrue(torch.equal(o['a_w'],a*o['w'][...,None]))

    def test_teacher_frozen_and_not_a_selection_feature(self):
        o=self.out();o['a_w'].sum().backward();self.assertIsNone(self.x['teacher_a'].grad)
        z=self.out(teacher_a=torch.zeros_like(self.x['teacher_a']))
        self.assertTrue(torch.equal(o['w'],z['w']));self.assertEqual(z['a_w'].abs().sum().item(),0)

    def test_no_gt_input_and_independent_conditions(self):
        self.assertEqual(list(inspect.signature(self.m.forward).parameters),['points','text_features','observed_history','teacher_a'])
        o=self.out();h=self.x['observed_history'].clone();h[...,:2]+=3
        oh=self.out(observed_history=h);ot=self.out(text_features=self.x['text_features']+2)
        for k in ('slot_xy','support'):
            self.assertTrue(torch.equal(o[k],oh[k]));self.assertTrue(torch.equal(o[k],ot[k]))
        self.assertTrue(torch.equal(o['relation_score'],oh['relation_score']))
        self.assertTrue(torch.equal(o['history_score'],ot['history_score']))
        self.assertFalse(torch.equal(o['history_score'],oh['history_score']))
        self.assertFalse(torch.equal(o['relation_score'],ot['relation_score']))

    def test_two_experts_routers_receive_map_gradient(self):
        self.out()['a_w'].square().mean().backward()
        for branch in (self.m.relation,self.m.history):
            self.assertEqual(len(branch.experts),2)
            for part in list(branch.experts)+[branch.router]:
                self.assertGreater(sum(p.grad.abs().sum().item() for p in part.parameters()),0)

    def test_joint_finite_connected(self):
        losses=competition_losses(self.out(),self.x['teacher_a'],self.l)
        losses['total'].backward()
        for name,p in self.m.named_parameters():
            self.assertIsNotNone(p.grad,name);self.assertTrue(torch.isfinite(p.grad).all(),name)

    def test_branch_supervision_isolated_at_moe_heads(self):
        competition_losses(self.out(),self.x['teacher_a'],self.l)['history'].backward()
        self.assertTrue(all(p.grad is None for p in self.m.relation.parameters()))
        self.m.zero_grad(set_to_none=True)
        competition_losses(self.out(),self.x['teacher_a'],self.l)['relation'].backward()
        self.assertTrue(all(p.grad is None for p in self.m.history.parameters()))

    def test_point_permutation_equivariance(self):
        p=torch.randperm(64);o=self.out();v=self.out(points=self.x['points'][:,p],teacher_a=self.x['teacher_a'][:,p])
        torch.testing.assert_close(v['a_w'],o['a_w'][:,p],atol=3e-6,rtol=3e-6)

    def test_oracle_votes_localize_and_match_swapped_slots(self):
        votes=torch.tensor([[[-2.,0],[-2.,0],[2.,0],[2.,0],[40.,40.]]])
        c,a,p=candidate_slots(votes,torch.tensor([[1.,1.,1.,1.,0.]]))
        torch.testing.assert_close(c,torch.tensor([[[-2.,0],[2.,0]]]),atol=1e-5,rtol=1e-5)
        o=self.out();l=dict(self.l,candidate_anchors=o['slot_xy'].detach().flip(1))
        aligned=align(o,l)
        torch.testing.assert_close(aligned['selection'],o['selection'].flip(1))
        torch.testing.assert_close(spatial_weight(aligned['support'],aligned['selection']),o['w'])

    def test_continuous_distance_targets(self):
        a=np.array([[0.,0],[4.,0]],np.float32);c=np.array([[.5,0],[0,0],[0,0]],np.float32)
        r,d=relation_scores(a,c,[1,0,0]);c[0,0]+=.001;r2,_=relation_scores(a,c,[1,0,0])
        self.assertGreater(r[0],.99);self.assertLess(abs(r[0]-r2[0]),.001)
        self.assertNotEqual(float(r[0]),float(r2[0]));np.testing.assert_allclose(d[:,0],[.5,3.5])

    def test_multi_context_history_switch_and_single_purpose_priority(self):
        a=np.array([[0.,0],[4.,0]],np.float32);c=np.array([[0,0],[0,1],[4,1]],np.float32)
        r,_=relation_scores(a,c,[0,1,1]);np.testing.assert_allclose(r[0],r[1])
        h=np.zeros((8,6),np.float32);h[:,0]=-1;h[:,4]=1
        h1=history_scores(h,a);h[:,0]=5;h[:,4]=-1;h2=history_scores(h,a)
        self.assertGreater(compete(r,h1)[1][0],.94);self.assertGreater(compete(r,h2)[1][1],.94)
        r,_=relation_scores(a,c,[0,1,0]);self.assertGreater(compete(r,h2)[1][0],.97)

    def test_no_forced_winner_without_evidence(self):
        _,q=compete(np.array([.5,.5]),np.array([.5,.5]));np.testing.assert_allclose(q,[.5,.5])

    def test_numpy_torch_choice_equivalence(self):
        from small_room30_competition_model import fuse_candidates
        r=np.array([.3,.85],np.float32);h=np.array([.91,.09],np.float32)
        s,q=fuse_candidates(torch.from_numpy(r),torch.from_numpy(h))
        sn,qn=compete(r,h);np.testing.assert_allclose(s.numpy(),sn);np.testing.assert_allclose(q.numpy(),qn,rtol=1e-6)

    def test_floor_support_not_point_one_five_and_null_objects(self):
        points=np.zeros((8,6),np.float32);contacts=np.zeros((2,8,6),np.float32)
        contacts[0,:4]=1;contacts[1,4:]=1
        masks=np.zeros((2,8),bool);masks[0,0]=1;masks[1,7]=1
        am=masks.any(0);am[3]=True;nc=np.zeros(8,bool);nc[3]=True
        s,f,fs,fw=support_labels(points,contacts,masks,am,nc)
        np.testing.assert_allclose(s[1],[1,0,0]);np.testing.assert_allclose(s[3],[0,0,1])
        q=np.array([.95,.05]);w=(s[:,:2]*q).sum(-1)
        self.assertAlmostEqual(w[1],.95);self.assertAlmostEqual(w[5],.05);self.assertGreater(fw.sum(),0)

    def test_numeric_checkpoint_roundtrip_without_pickle(self):
        with tempfile.TemporaryDirectory() as tmp,patch('torch.load',side_effect=AssertionError('No pickle')):
            root=Path(tmp);(root/'manifest.json').write_text('{}')
            save_model(self.m,root,{})
            clone=load_model(root/'student_model.json')
            self.assertTrue(torch.equal(clone(**self.x)['a_w'],self.out()['a_w']))
            with (root/'student_weights.npz').open('ab') as f:f.write(b'tamper')
            with self.assertRaises(ValueError):load_model(root/'student_model.json')

    def test_invalid_inputs(self):
        for kwargs in (dict(points=self.x['points']+2),dict(observed_history=torch.zeros(1,9,6)),
                       dict(teacher_a=torch.full_like(self.x['teacher_a'],float('nan')))):
            with self.assertRaises(ValueError):self.out(**kwargs)

    def test_wrong_confident_choice_and_floor_collapse_fail(self):
        with torch.no_grad():
            o=self.out();o['slot_xy']=self.l['candidate_anchors'];o['selection']=self.l['selection_target'].flip(1)
            o['a_w']=torch.zeros_like(o['a_w'])
            d=diagnostic(o,self.l,self.x['teacher_a'])
        self.assertIn('wrong_confident_candidate',d['failures'])
        self.assertIn('floor_support_mismatch',d['failures']);self.assertFalse(d['diagnostic_gate_pass'])

    def test_viewer_rejects_old_or_tampered_formula(self):
        with torch.no_grad():o=self.out()
        a={k:v[0].numpy() for k,v in o.items()}
        a.update(points=self.x['points'][0].numpy(),teacher_a=self.x['teacher_a'][0].detach().numpy(),
            observed_history=self.x['observed_history'][0].numpy(),baseline_aw=a['a_w'].copy(),training_target_aw=a['a_w'].copy())
        for k in ('points','teacher_a','a_w','baseline_aw','training_target_aw','support','w','relation_map','history_map','score_map'):
            a[k]=np.tile(a[k],(128,)+(1,)*(a[k].ndim-1))
        validate_maps(a);a['selection']=np.array([.99,.01],np.float32)
        with self.assertRaisesRegex(ValueError,'formula'):validate_maps(a)

    def test_train_export_reload_and_g2_not_optimized(self):
        x,l=self.x,self.l
        records=[dict(scene_id='synthetic',generation=g,prompt_id='sit_watch_v1',history_motion_id='h',
                      split='train' if g<2 else 'same_scene_noise_repeat',sample_id='s') for g in range(3)]
        class Data:
            def __init__(self):self.records=records;self.seen=[]
            def example(self,r):
                self.seen.append(r['generation'])
                return ({k:v[0].detach().numpy() for k,v in x.items() if k!='text_features'},
                        {k:v[0].numpy() for k,v in l.items()})
            def geometry(self,s):return (None,None,None,['a','b'])
        data=Data();text={'sit_watch_v1':x['text_features']}
        with tempfile.TemporaryDirectory() as tmp,patch('torch.load',side_effect=AssertionError('No pickle')):
            root=Path(tmp);out=root/'new';out.mkdir();(out/'manifest.json').write_text('{}')
            model,logs=train(data,text,out,'cpu',1,2,.001,14,16,{})
            self.assertNotIn(2,data.seen);self.assertEqual(logs[-1]['step'],3)
            np.savez(root/'old.npz',points=x['points'][0].numpy(),teacher_a=x['teacher_a'][0].detach().numpy(),a_w=x['teacher_a'][0].detach().numpy())
            comparison=dict(rows=[dict(record=records[-1],file='old.npz',sha256=sha(root/'old.npz'))])
            rows,report=evaluate(model,data,text,out,root/'summary.json',comparison,'cpu')
            self.assertEqual(len(rows),1);self.assertEqual(report['status'],'DIAGNOSTICS_ONLY_NOT_APPROVAL')
            clone=load_model(out/'student_model.json');self.assertTrue(torch.equal(clone(**x)['a_w'],model(**x)['a_w']))


if __name__=='__main__':unittest.main(verbosity=2)
