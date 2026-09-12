"""Bounded synthetic structural/entrypoint tests, not model-quality evidence."""
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from small_room30_weight_student_v2 import WeightStudentV2, student_losses_v2, geometric_features
from small_room30_student_v2_data import eligible_candidates, calibrated_history, corrected_targets
from train_small_room30_student_v2 import pair_indices, group_records, run_training, evaluate
from small_room30_student_data import TEXTS
from small_room30_adm_dataset import sha
from relation_aware_moe_v2_contract import observed_history_prefix_numpy


def fixture():
    points=torch.rand(1,64,6); points[...,:2]*=4
    history=torch.zeros(1,8,6); history[...,4]=1
    teacher=torch.rand(1,64,6,requires_grad=True)
    masks=torch.zeros(1,2,64,dtype=torch.bool); masks[:,0,:16]=1; masks[:,1,16:32]=1
    semantic=torch.zeros(1,64,dtype=torch.long); semantic[:,:32]=1; semantic[:,32:48]=4; semantic[:,48:56]=3
    anchors=torch.rand(1,64,2); anchor_mask=semantic!=0
    labels=dict(candidate_masks=masks,candidate_relation=torch.tensor([[.95,.05]]),history_scores=torch.tensor([[.2,.8]]),
        eligible=torch.tensor([[True,False]]),semantic=semantic,purpose_classes=torch.tensor([[0.,1.,0.]]),
        point_anchor=anchors,anchor_mask=anchor_mask,presence=torch.tensor([[1.,1.,0.]]),
        context_anchors=torch.rand(1,3,2),relation_spatial=torch.rand(1,64),history_spatial=torch.rand(1,64),
        noncandidate_mask=anchor_mask & ~masks.any(1),purpose_mask=(semantic==4).float())
    return dict(points=points,text_features=torch.randn(1,512),observed_history=history,teacher_a=teacher),labels


class StudentV2Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(14); torch.set_num_threads(1)
        self.model=WeightStudentV2(width=16)
        self.inputs,self.labels=fixture()

    def output(self, **changes): return self.model(**dict(self.inputs,**changes))

    def test_exact_shared_formula_raw_teacher(self):
        a=self.inputs['teacher_a'].detach().clone(); a[...,0]=-0.1; a[...,1]=1.2
        out=self.output(teacher_a=a)
        self.assertTrue(torch.equal(out['w'],.6*out['w_r']+.4*out['w_h']))
        self.assertTrue(torch.equal(out['a_w'],a*out['w'][...,None]))
        self.assertTrue((out['a_w'][...,0]<0).all())

    def test_teacher_frozen_and_zero_stays_zero(self):
        self.output()['a_w'].sum().backward(); self.assertIsNone(self.inputs['teacher_a'].grad)
        self.assertEqual(float(self.output(teacher_a=torch.zeros_like(self.inputs['teacher_a']))['a_w'].detach().abs().sum()),0)

    def test_forward_no_supervision(self):
        self.assertEqual(list(inspect.signature(self.model.forward).parameters),
                         ['points','text_features','observed_history','teacher_a'])

    def test_independent_branches(self):
        h=self.inputs['observed_history'].clone(); h[...,:2]+=2
        first=self.output(); second=self.output(observed_history=h)
        self.assertTrue(torch.equal(first['w_r'],second['w_r']))
        self.assertFalse(torch.equal(first['w_h'],second['w_h']))
        second=self.output(text_features=self.inputs['text_features']+1)
        self.assertTrue(torch.equal(first['w_h'],second['w_h']))
        self.assertFalse(torch.equal(first['w_r'],second['w_r']))

    def test_two_experts_and_disjoint_parameters(self):
        r={id(p) for p in self.model.relation.parameters()}; h={id(p) for p in self.model.history.parameters()}
        self.assertFalse(r & h)
        self.assertEqual(len(self.model.relation.experts),2); self.assertEqual(len(self.model.history.experts),2)

    def test_map_gradients_reach_experts_routers(self):
        self.output()['a_w'].square().mean().backward()
        for branch in (self.model.relation,self.model.history):
            for part in list(branch.experts)+[branch.router]:
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in part.parameters()),0)

    def test_all_joint_parameters_connected(self):
        loss=student_losses_v2(self.output(),self.inputs['teacher_a'],self.labels)
        loss['total'].backward()
        for name,p in self.model.named_parameters():
            self.assertIsNotNone(p.grad,name); self.assertTrue(torch.isfinite(p.grad).all(),name)

    def test_loss_branch_independence(self):
        student_losses_v2(self.output(),self.inputs['teacher_a'],self.labels)['history'].backward()
        self.assertTrue(all(p.grad is None for p in self.model.relation_scene.parameters()))
        self.model.zero_grad(set_to_none=True)
        student_losses_v2(self.output(),self.inputs['teacher_a'],self.labels)['relation'].backward()
        self.assertTrue(all(p.grad is None for p in self.model.history_scene.parameters()))

    def test_separate_context_slots(self):
        out=self.output(); self.assertEqual(out['context_xy'].shape,(1,3,2))
        self.assertFalse(torch.equal(out['context_xy'][:,1],out['context_xy'][:,2]))

    def test_permutation_equivariance(self):
        permutation=torch.randperm(64)
        out=self.output()
        changed=self.output(points=self.inputs['points'][:,permutation],teacher_a=self.inputs['teacher_a'][:,permutation])
        torch.testing.assert_close(changed['a_w'],out['a_w'][:,permutation],atol=2e-6,rtol=2e-6)

    def test_checkpoint_roundtrip_weights_only(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'student.pt'; torch.save(dict(config=self.model.config,model=self.model.state_dict()),path)
            ckpt=torch.load(path,weights_only=True); clone=WeightStudentV2(**ckpt['config']); clone.load_state_dict(ckpt['model'])
            self.assertTrue(torch.equal(clone(**self.inputs)['a_w'],self.output()['a_w']))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError): self.output(observed_history=torch.zeros(1,9,6))
        with self.assertRaises(ValueError): self.output(points=self.inputs['points']+2)
        with self.assertRaises(ValueError): self.output(teacher_a=torch.full_like(self.inputs['teacher_a'],float('nan')))

    def test_multi_context_union_not_global_winner(self):
        a=np.array([[0,0],[4,0]],np.float32); c=np.array([[.5,0],[4,2]],np.float32)
        np.testing.assert_array_equal(eligible_candidates(a,c),[True,True])
        np.testing.assert_array_equal(eligible_candidates(a,c[:1]),[True,False])

    def test_history_reversal_without_target_ids(self):
        h=np.zeros((8,6),np.float32); h[:,4]=1
        a=np.array([[1,0],[-1,0]],np.float32)
        left=calibrated_history(h,a); h[:,4]=-1; right=calibrated_history(h,a)
        self.assertGreater(left[0],left[1]); self.assertGreater(right[1],right[0])
        self.assertTrue(np.all(left>=.05)); self.assertTrue(np.all(left<=.95))

    def test_future_prefix_invariance(self):
        poses=np.zeros((20,22,3),np.float32); poses[:,2,0]=1; poses[:,17,0]=1
        first=observed_history_prefix_numpy(poses,np.arange(20)/20)
        poses[8:]=np.nan
        np.testing.assert_array_equal(first,observed_history_prefix_numpy(poses,np.arange(20)/20))

    def test_single_eligibility_dominates_hostile_history(self):
        r=np.array([.95,.05]); h=np.array([.05,.95]); w=.6*r+.4*h
        self.assertGreater(w[0],w[1])

    def test_ambiguous_equal_relation_allows_reversal(self):
        r=np.array([.45,.45]); h=np.array([.95,.05])
        self.assertGreater((.6*r+.4*h)[0],(.6*r+.4*h)[1])
        self.assertLess((.6*r+.4*h[::-1])[0],(.6*r+.4*h[::-1])[1])
        # Explicitly not a binary on/off target.
        self.assertAlmostEqual(float((.6*r+.4*h).min()),.29)

    def test_spatial_negative_furniture(self):
        masks=np.zeros((2,8),bool); masks[0,:2]=1; masks[1,2:4]=1
        nc=np.zeros(8,bool); nc[6:]=1
        r,h=corrected_targets(np.ones((2,8,6)),masks,np.array([.95,.05]),np.array([.8,.2]),nc)
        np.testing.assert_allclose(r[nc],.02); np.testing.assert_allclose(h[nc],.02)
        np.testing.assert_allclose(r[:2],.95)

    def test_same_scene_paired_batch_and_g2_exclusion(self):
        records=[dict(scene_id='synthetic',generation=g,prompt_id=p,history_motion_id=h,split='train' if g<2 else 'same_scene_noise_repeat')
                 for g in range(3) for p in ('watch','write') for h in ('a','b')]
        train_groups=group_records([r for r in records if r['split']=='train'])
        self.assertEqual(len(train_groups),2)
        for group in train_groups:
            p,h=pair_indices(group); self.assertEqual(len(p),2); self.assertEqual(len(h),2)
            self.assertTrue(all(r['generation']<2 for r in group))

    def test_synthetic_train_and_export(self):
        outer=self
        class FakeData:
            def __init__(self):
                self.records=[dict(scene_id='synthetic',generation=g,prompt_id=p,history_motion_id=h,
                    split='train' if g<2 else 'same_scene_noise_repeat') for g in range(3)
                    for p in ('sit_watch_v1','sit_write_v1') for h in ('a','b')]
            def example(self,record):
                inputs={k:v.detach()[0].numpy().copy() for k,v in outer.inputs.items() if k!='text_features'}
                labels={k:v[0].numpy().copy() for k,v in outer.labels.items()}
                if record['history_motion_id']=='b': inputs['observed_history'][:,:2]+=1
                return inputs,labels
            def geometry(self,scene): return (None,None,None,['chair_a','chair_b'])
        data=FakeData(); text={k:torch.randn(1,512) for k in TEXTS}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); output=root/'v2'; output.mkdir(); base=root/'baseline'; base.mkdir()
            baseline=dict(rows=[])
            for i,r in enumerate(x for x in data.records if x['generation']==2):
                inputs,_=data.example(r); file=base/('%d.npz'%i)
                np.savez_compressed(file,points=inputs['points'],teacher_a=inputs['teacher_a'],a_w=inputs['teacher_a']*.5)
                baseline['rows'].append(dict(record=r,file=file.name,sha256=sha(file)))
            model,logs=run_training(data,text,output,dict(clip_weights_sha256='synthetic'), 'cpu',1,2,.001,5,16)
            self.assertEqual(logs[-1]['phase'],'joint_moe')
            rows=evaluate(model,data,text,output,base/'summary.json',baseline,'cpu')
            self.assertEqual(len(rows),4)
            self.assertTrue((output/'comparison.json').is_file())
            for row in rows:
                with np.load(output/row['file']) as z:
                    np.testing.assert_array_equal(z['w'],.6*z['w_r']+.4*z['w_h'])
                    np.testing.assert_array_equal(z['a_w'],z['teacher_a']*z['w'][:,None])

    def test_existing_output_refused_before_preflight(self):
        import train_small_room30_student_v2 as entry
        with tempfile.TemporaryDirectory() as d, patch('sys.argv',['train','--output-dir',d]), patch.object(entry,'StudentDataV2') as dataset:
            with self.assertRaises(FileExistsError): entry.main()
            dataset.assert_not_called()

    def test_preflight_entrypoint_never_trains(self):
        import train_small_room30_student_v2 as entry
        fake=type('FakeData',(),{'binding':{'records':[]}})()
        with tempfile.TemporaryDirectory() as d, patch('sys.argv',['train','--preflight-only','--output-dir',str(Path(d)/'new')]), \
                patch.object(entry,'StudentDataV2',return_value=fake), \
                patch.object(entry,'load_baseline',return_value=({}, {}, 'synthetic')), \
                patch.object(entry,'target_audit',return_value={}), patch.object(entry,'run_training') as train:
            entry.main(); train.assert_not_called()
            self.assertEqual(json.loads((Path(d)/'new/summary.json').read_text())['status'],'DATA_PREFLIGHT_PASS_NO_TRAINING')


if __name__=='__main__': unittest.main()
