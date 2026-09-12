"""CPU scope, endpoint gradient,1570 provenance, transaction and runner tests."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
torch.set_num_threads(1)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from small_room30_adm_dataset import SmallRoom30ADM,sha
from small_room30_evaluation_common import panel
from small_room30_training_contract import write_json
from small_room30_training_runtime import adapter_state,state_digest,frozen_digest
from small_room30_targeted_v5 import (POLICY,TARGETS,PASSED,plan,instance_support,
    presence,target_stats,acceptance,check_ready,reference,source_contract)
from small_room30_targeted_v5_runtime import RepairRuntime,missing_loss,preservation_loss,transactional_step,tree_equal
from fewshot_cdm_lora import lora_named_parameters
from test_small_room30_onpolicy import ToyRuntime


class RepairToy(ToyRuntime):
    gradients=RepairRuntime.gradients


class V5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=SmallRoom30ADM(os.environ.get('SMALL_ROOM30_DATASET','data/small_room30_adm_lora_v1'))
        ToyRuntime.data_source=cls.data
        allcases=panel(cls.data,'full')
        cases=[next(c for c in allcases if c['filename']==n) for n in [TARGETS[0],PASSED[0],TARGETS[1],PASSED[1]]]
        guards=[]
        for action in ('sit','lie','write_board'):
            rooms=set()
            for c in allcases:
                if c['action']==action and c not in cases and c['scene_id'] not in rooms:
                    guards.append(dict(case=c));rooms.add(c['scene_id'])
                    if len(rooms)==4:break
        cls.cases=cases
        cls.bound=dict(failures=[dict(case=c) for c in cases],guards=guards,reference_summary='unused')

    def reports(self):
        rows=[]
        for item in self.bound['failures']+self.bound['guards']:
            c=item['case'];m,n,_=instance_support(self.data,c['index'])
            raw=np.full((8192,6),.7,np.float32)
            stats=None
            if c['filename'] in TARGETS:
                mask=m[n.index(c['scene_id']+'__100__sit')]
                raw[mask]=.1
                if c['filename']==TARGETS[0]:raw[np.flatnonzero(mask)[:601]]=.4
                stats=target_stats(raw,mask)
            p=presence(raw,m,n)
            rows.append(dict(case=c,before_presence=p,after_presence=copy.deepcopy(p),repair=stats))
        return dict(rows=rows)

    def prepared(self):
        rt=RepairToy();params=list(lora_named_parameters(rt.model).values())
        opt=torch.optim.AdamW(params,lr=POLICY['lr'],weight_decay=0.)
        for p in params:p.grad=torch.ones_like(p)*.01
        opt.step()
        for p in params:p.grad=torch.ones_like(p)*.01
        return rt,opt

    def test_exact_two_paths_only_endpoint(self):
        entries=plan(self.data,self.bound,72)
        self.assertEqual(len(entries),16)
        self.assertEqual({e['case']['filename'] for e in entries},set(TARGETS))
        self.assertTrue(all(e['timesteps']==[0] for e in entries))
        self.assertTrue(all([g['filename'] for g in e['guards'][:2]]==PASSED for e in entries))
        self.assertEqual(len({g['filename'] for e in entries for g in e['guards'][2:]}),12)

    def test_out_of_scope_rejected(self):
        b=copy.deepcopy(self.bound);b['failures'][0]['case']['action']='lie'
        with self.assertRaises(ValueError):plan(self.data,b,72)

    def test_0205_old_presence_does_not_skip(self):
        row=self.reports()['rows'][0]
        self.assertTrue(row['after_presence']['all_furniture_present'])
        self.assertFalse(row['repair']['optimization_complete'])
        self.assertFalse(acceptance(self.reports(),self.reports(),self.cases[0])['accepted'])

    def test_half_chair_completion_not_whole_gt(self):
        raw=np.zeros((8192,6),np.float32);mask=np.zeros(8192,bool);mask[:16]=True
        raw[:8,5]=.41
        s=target_stats(raw,mask);self.assertTrue(s['optimization_complete'])
        raw[7]=0
        self.assertFalse(target_stats(raw,mask)['optimization_complete'])

    def test_no_finite_map_no_score(self):
        raw=np.zeros((8192,6));raw[0,0]=np.nan
        with self.assertRaises(ValueError):target_stats(raw,np.ones(8192,bool))

    def test_loss_reaches_half_chair_not_only_top_quarter(self):
        mask=torch.zeros(8192,dtype=torch.bool);mask[:64]=True
        x=torch.full((1,8192,6),.1);x[0,:16,5]=.7;x.requires_grad_()
        loss=missing_loss(x,mask);loss.backward()
        self.assertGreater(float(loss),0)
        self.assertGreater(float(x.grad[0,16:64].abs().sum()),0)
        self.assertEqual(float(x.grad[0,64:].abs().sum()),0)
        self.assertEqual(float(x.grad[0,:16].abs().sum()),0)

    def test_sufficient_values_no_more_boost(self):
        x=torch.full((1,8192,6),.6,requires_grad=True)
        self.assertEqual(float(missing_loss(x,torch.ones(8192,dtype=torch.bool))),0)

    def test_preserve_all1570_visible_points(self):
        x=torch.zeros((1,8192,6),requires_grad=True);old=torch.full_like(x,.8)
        masks=torch.zeros((1,8192),dtype=torch.bool);masks[0,:64]=True
        preservation_loss(x,old,masks).backward()
        self.assertEqual(int((x.grad[0,:64].abs().sum(dim=1)>0).sum()),64)
        self.assertEqual(float(x.grad[0,64:].abs().sum()),0)

    def test_no_suppression_or_filling_invisible_targets(self):
        masks=torch.ones((1,8192),dtype=torch.bool)
        x=torch.full((1,8192,6),.6,requires_grad=True)
        self.assertEqual(float(preservation_loss(x,torch.ones_like(x),masks)),0)
        self.assertEqual(float(preservation_loss(x*0,torch.full_like(x,.1),masks)),0)

    def test_real_endpoint_progress_required_even_old_gate_passed(self):
        p=self.reports();c=copy.deepcopy(p);c['rows'][0]['repair']['top_mean']+=.01
        self.assertTrue(acceptance(c,p,self.cases[0])['accepted'])

    def test_accepted0403g1_loss_rejected(self):
        p=self.reports();c=copy.deepcopy(p);c['rows'][0]['repair']['top_mean']+=.01
        c['rows'][1]['after_presence']['instances'][0]['present']=False
        self.assertFalse(acceptance(c,p,self.cases[0])['accepted'])

    def test_other_actions_not_positive_targets(self):
        p=self.reports()
        with self.assertRaises(ValueError):acceptance(p,p,self.bound['guards'][0]['case'])

    def test_changed_panel_rejected(self):
        p=self.reports();c=copy.deepcopy(p);c['rows'].reverse()
        with self.assertRaises(ValueError):acceptance(c,p,self.cases[0])

    def test_gradients_only_final_no_gt_objective(self):
        rt,opt=self.prepared();before=state_digest(adapter_state(rt.model));frozen=frozen_digest(rt.model)
        with patch.object(rt,'objective',side_effect=AssertionError('No GT objective')), \
                patch('small_room30_targeted_v5_runtime.reference',return_value=np.full((8192,6),.7,np.float32)), \
                contextlib.redirect_stdout(io.StringIO()):
            row=rt.gradients(opt,plan(self.data,self.bound,72)[0],1571,Path('unused'))
        self.assertEqual([t['timestep'] for t in row['target_terms']],[0])
        self.assertEqual(len(row['preservation_terms']),6)
        self.assertEqual(row['preservation_anchor_step'],1570)
        self.assertEqual(before,state_digest(adapter_state(rt.model)))
        self.assertEqual(frozen,frozen_digest(rt.model))

    def test_rollback_both_attempts_nonempty_adam(self):
        rt,opt=self.prepared();state=state_digest(adapter_state(rt.model));moments=copy.deepcopy(opt.state_dict())
        result=transactional_step(rt,opt,lambda n,s:dict(accepted=False))
        self.assertFalse(result['accepted']);self.assertEqual(len(result['attempts']),2)
        self.assertEqual(state,state_digest(adapter_state(rt.model)));self.assertTrue(tree_equal(moments,opt.state_dict()))

    def test_old_readiness_rejected(self):
        with self.assertRaises(ValueError):check_ready(dict(status='TARGETED_V4_RUNTIME_READY'),self.bound)

    def test_1560_checkpoint_rejected(self):
        from run_small_room30_targeted_v5 import load_runtime
        with patch('small_room30_checkpoint_io.load_checkpoint',return_value=dict(step=1560)):
            with self.assertRaises(ValueError):load_runtime(self.data,dict(last_checkpoint=dict(path='unused')),{},'cpu')

    def test_unreviewed_run_rejected_before_model_loading(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'summary.json';write_json(f,{})
            with self.assertRaises(ValueError):source_contract(self.data,f,f,f,Path(d))

    def test_reference_reads1570_adapted_not1560_source(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);c=self.cases[0];f=root/c['filename']
            np.savez_compressed(f,source_raw=np.zeros((8192,6)),adapted_raw=np.ones((8192,6)))
            write_json(root/'summary.json',dict(rows=[dict(case=c,sha256=sha(f))]))
            self.assertTrue((reference(root/'summary.json',c)==1).all())
            np.savez_compressed(f,adapted_raw=np.zeros((8192,6)))
            with self.assertRaises(ValueError):reference(root/'summary.json',c)

    def test_runner_smoke_reject_and_saved1571(self):
        import run_small_room30_targeted_v5 as runner
        from small_room30_checkpoint_io import load_checkpoint
        for accept in (False,True):
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);cp=root/'source.pt';cp.write_bytes(b'saved1570')
                source=dict(seed=72,last_checkpoint=dict(path=str(cp),sha256=sha(cp),step=1570))
                src,ev,diag=[root/n for n in ('source.json','eval.json','diag.json')]
                for p,value in ((src,source),(ev,{}),(diag,{})):write_json(p,value)
                previous=self.reports()
                def measured(rt,b,ev,out,**kw):
                    report=copy.deepcopy(previous)
                    report.update(adapter_digest=state_digest(adapter_state(rt.model)),decision=dict(optimization_complete=accept))
                    out.mkdir(parents=True);write_json(out/'summary.json',report);return report
                def gradients(rt,*args):
                    for p in lora_named_parameters(rt.model).values():p.grad=torch.ones_like(p)*.01
                    return {'case':'toy'}
                for mode in ('smoke','train'):
                    argv=['run','--mode',mode,'--dataset',str(self.data.root),'--source-summary',str(src),
                        '--evaluation-summary',str(ev),'--diagnosis-summary',str(diag),'--output-dir',str(root/mode),'--device','cpu']
                    if mode=='train':argv+=['--readiness',str(root/'smoke/summary.json'),'--allow-research-training']
                    with patch.object(sys,'argv',argv),patch('pathlib.Path.cwd',return_value=runner.REPO), \
                            patch.object(runner,'source_contract',return_value=(source,{},self.bound,[])), \
                            patch.object(runner,'load_runtime',side_effect=lambda *args:RepairToy()), \
                            patch.object(runner,'check_transitions',return_value=[]),patch.object(RepairToy,'gradients',gradients), \
                            patch('small_room30_targeted_v5_runtime.measured_canary',side_effect=measured), \
                            patch('small_room30_targeted_v5_runtime.acceptance',return_value=dict(accepted=accept)), \
                            contextlib.redirect_stdout(io.StringIO()):runner.main()
                result=json.loads((root/'train/summary.json').read_text())
                self.assertEqual(result['accepted_updates'],int(accept));self.assertEqual(cp.read_bytes(),b'saved1570')
                if accept:
                    saved=load_checkpoint(result['last_checkpoint']['path'])
                    self.assertEqual(saved['step'],1571);self.assertEqual(saved['purpose'],'RESEARCH_TARGETED_V5_PILOT')
                    self.assertEqual(state_digest(saved['adapter']),result['last_checkpoint']['adapter_digest'])
                else:self.assertIsNone(result['last_checkpoint'])


if __name__=='__main__':unittest.main()
