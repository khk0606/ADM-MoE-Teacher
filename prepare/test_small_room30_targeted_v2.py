"""CPU tests for bounded plans, surface guards, transactions and runner."""
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
from test_small_room30_onpolicy import ToyRuntime
from small_room30_adm_dataset import SmallRoom30ADM,sha
from small_room30_evaluation_common import panel
from small_room30_training_contract import write_json
from small_room30_training_runtime import adapter_state,restore_adapter,state_digest,frozen_digest
from fewshot_cdm_lora import lora_named_parameters
from small_room30_targeted_v2 import transition,plan,check_ready,POLICY,source_contract
from small_room30_targeted_v2_runtime import RepairRuntime,preserve_loss,transactional_step,tree_equal,acceptance


class RepairToy(ToyRuntime):
    gradients=RepairRuntime.gradients


class V2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=SmallRoom30ADM(os.environ.get('SMALL_ROOM30_DATASET','data/small_room30_adm_lora_v1'))
        ToyRuntime.data_source=cls.data
        cases=panel(cls.data,'full')
        def case(name):return next(c for c in cases if c['filename']==name)
        from small_room30_targeted import FAILURES
        cls.cases=[case(n) for n in FAILURES]
        cls.critical=[case('small_room_0603__sit_anywhere__g1.npz'),case('small_room_0204__lie_bed__g0.npz'),case('small_room_0404__go_write_board__g1.npz')]
        cls.bound=dict(transitions=[dict(case=c,timesteps=ts) for c,ts in zip(cls.cases,
            [[490,475,0],[300,200,0],[475,450,0],[495,490,0]])],critical_guards=cls.critical)

    def test_transition_uses_last_stable_loss_not_max_deficit(self):
        row=dict(case=self.cases[0],records=[dict(timestep=t,score=dict(all_furniture_present=ok),surface_loss=loss)
            for t,ok,loss in [(499,True,0),(498,False,1),(495,True,0),(490,True,0),(475,False,.1),(0,False,9)]])
        self.assertEqual(transition(row)['timesteps'],[490,475,0])

    def test_transition_without_presence_rejected(self):
        row=dict(case=self.cases[0],records=[dict(timestep=t,score=dict(all_furniture_present=False)) for t in (499,490,0)])
        with self.assertRaises(ValueError):transition(row)
        row['records'].reverse()
        with self.assertRaises(ValueError):transition(row)

    def test_plan_protects_all_actions_from_first_update(self):
        entries=plan(self.data,self.bound,72)
        self.assertEqual(len(entries),4)
        for e in entries:self.assertEqual(e['guards'],self.critical)
        self.assertEqual([e['timesteps'] for e in entries],[[490,475,0],[300,200,0],[475,450,0],[495,490,0]])

    def test_plan_replays_all30_sit_rooms(self):
        groups=[g for e in plan(self.data,self.bound,72) for g in e['replay']]
        self.assertEqual(len(groups),32)
        self.assertEqual(len({self.data.samples[g[0]]['scene_id'] for g in groups}),30)
        self.assertTrue(all([self.data.samples[i]['action'] for i in g]==['sit','lie','write_board'] for g in groups))

    def test_surface_preservation_not_diluted_by_background(self):
        old=torch.ones(1,8192,6)*.4;new=old.clone();mask=torch.zeros(1,8192,6,dtype=torch.bool);mask[0,:16,0]=True
        new[0,:16,0]=.1;new.requires_grad_(True)
        loss,parts=preserve_loss(new,old,mask)
        self.assertGreater(float(parts['surface'].detach()),.1);self.assertLess(float(parts['full_map'].detach()),.001)
        loss.backward();self.assertLess(float(new.grad[0,:16,0].mean()),0)

    def test_no_surface_penalty_for_identical_or_increased_scores(self):
        old=torch.ones(1,16,6)*.4;mask=torch.ones(1,16,6,dtype=torch.bool)
        for new in (old,old+.1):
            _,parts=preserve_loss(new,old,mask);self.assertEqual(float(parts['surface']),0)

    def prepared(self,nonempty=False):
        rt=RepairToy();opt=torch.optim.AdamW(list(lora_named_parameters(rt.model).values()),lr=POLICY['lr'],weight_decay=0.)
        def grad():
            opt.zero_grad(set_to_none=True)
            for p in lora_named_parameters(rt.model).values():p.grad=torch.ones_like(p)*.01
        grad()
        if nonempty:opt.step();grad()
        return rt,opt

    def test_rejection_restores_adapter_and_nonempty_optimizer(self):
        rt,opt=self.prepared(True);old=adapter_state(rt.model);state=copy.deepcopy(opt.state_dict());frozen=frozen_digest(rt.model)
        seen=[]
        result=transactional_step(rt,opt,lambda n,s:(seen.append((n,s)) or dict(accepted=False)))
        self.assertFalse(result['accepted']);self.assertEqual(seen,[(1,1.),(2,.25)])
        self.assertEqual(state_digest(old),state_digest(adapter_state(rt.model)));self.assertTrue(tree_equal(state,opt.state_dict()))
        self.assertEqual(frozen,frozen_digest(rt.model))

    def test_retry_is_quarter_step_from_same_weights_and_moments(self):
        rt,opt=self.prepared(True);old=adapter_state(rt.model);state=copy.deepcopy(opt.state_dict())
        expected=RepairToy();restore_adapter(expected.model,old)
        eo=torch.optim.AdamW(list(lora_named_parameters(expected.model).values()),lr=POLICY['lr'],weight_decay=0.)
        eo.load_state_dict(copy.deepcopy(state))
        for group in eo.param_groups:group['lr']=POLICY['lr']*.25
        for p in lora_named_parameters(expected.model).values():p.grad=torch.ones_like(p)*.01
        eo.step()
        result=transactional_step(rt,opt,lambda n,s:dict(accepted=n==2))
        self.assertTrue(result['accepted'])
        self.assertEqual(state_digest(adapter_state(rt.model)),state_digest(adapter_state(expected.model)))
        self.assertTrue(tree_equal(opt.state_dict(),eo.state_dict()))

    def test_exception_rolls_back(self):
        rt,opt=self.prepared(True);old=adapter_state(rt.model);state=copy.deepcopy(opt.state_dict())
        def fail(n,s):raise RuntimeError('evaluation failed')
        with self.assertRaises(RuntimeError):transactional_step(rt,opt,fail)
        self.assertEqual(state_digest(old),state_digest(adapter_state(rt.model)));self.assertTrue(tree_equal(state,opt.state_dict()))

    def test_first_accept_does_not_retry(self):
        rt,opt=self.prepared();result=transactional_step(rt,opt,lambda n,s:dict(accepted=True))
        self.assertEqual(len(result['attempts']),1)

    def reports(self):
        metric=dict(all_furniture_present=True,instances=[dict(id='f',hit_fraction=.8,present=True)])
        rows=[]
        for i in range(16):
            c=self.cases[i] if i<4 else dict(filename='guard%d'%i)
            m=copy.deepcopy(metric)
            if i<4:m=dict(all_furniture_present=False,instances=[dict(id='f',hit_fraction=0.,present=False)])
            rows.append(dict(case=c,role='failure' if i<4 else 'guard',before=copy.deepcopy(m),after=copy.deepcopy(m),
                before_surface_means={'f':.2 if i<4 else .5},after_surface_means={'f':.2 if i<4 else .5}))
        previous=dict(rows=copy.deepcopy(rows));rows[0]['after_surface_means']['f']+=.01
        return dict(rows=rows,decision=dict(stop_for_regression=False)),previous

    def test_accept_requires_real_target_gain(self):
        candidate,previous=self.reports();self.assertTrue(acceptance(candidate,previous,self.cases[0])['accepted'])
        candidate['rows'][0]['after_surface_means']['f']=.2
        self.assertFalse(acceptance(candidate,previous,self.cases[0])['accepted'])

    def test_guard_surface_drop_rejected_even_if_still_present(self):
        for key,amount in [('mean',.03),('hit',.06)]:
            candidate,previous=self.reports()
            if key=='mean':candidate['rows'][-1]['after_surface_means']['f']-=amount
            else:candidate['rows'][-1]['after']['instances'][0]['hit_fraction']-=amount
            self.assertFalse(acceptance(candidate,previous,self.cases[0])['accepted'])

    def test_newly_repaired_target_cannot_be_lost(self):
        candidate,previous=self.reports();previous['rows'][1]['after']['all_furniture_present']=True
        self.assertFalse(acceptance(candidate,previous,self.cases[0])['accepted'])

    def test_readiness_rejects_old_protocol(self):
        with self.assertRaises(ValueError):check_ready(dict(status='TARGETED_RUNTIME_READY'),{})

    def test_source_diagnosis_hash_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'diagnosis.json';write_json(p,{})
            with patch('small_room30_targeted_v2.old_contract',return_value=({}, {}, {})):
                with self.assertRaises(ValueError):source_contract(self.data,p,p,p,Path(d))

    def test_warmstart1564_rejected_before_model_construction(self):
        from run_small_room30_targeted_v2 import load_runtime
        with patch('small_room30_checkpoint_io.load_checkpoint',return_value=dict(step=1564)):
            with self.assertRaises(ValueError):load_runtime(self.data,dict(last_checkpoint={'path':'unused'}),{},'cpu')

    def test_old_regression_flag_cannot_be_ignored_by_new_surface_gate(self):
        candidate,previous=self.reports();candidate['decision']['stop_for_regression']=True
        self.assertFalse(acceptance(candidate,previous,self.cases[0])['accepted'])

    def test_gradients_do_not_mutate_model_and_include_three_free_guards(self):
        rt=RepairToy();opt=torch.optim.AdamW(list(lora_named_parameters(rt.model).values()),lr=POLICY['lr'])
        entry=plan(self.data,self.bound,72)[0];old=state_digest(adapter_state(rt.model));frozen=frozen_digest(rt.model)
        with patch('small_room30_targeted_v2_runtime.reference',return_value=np.zeros((8192,6),np.float32)),contextlib.redirect_stdout(io.StringIO()):
            row=rt.gradients(opt,entry,1561,Path('unused'))
        self.assertEqual(old,state_digest(adapter_state(rt.model)));self.assertEqual(frozen,frozen_digest(rt.model))
        self.assertEqual(len(row['target_terms']),3);self.assertEqual(len(row['guard_terms']),3);self.assertEqual(len(row['replay']),24)
        self.assertGreater(row['gradient_norm_before_clip'],0)

    def test_missing_write_protection_rejected(self):
        rt,opt=self.prepared();entry=copy.deepcopy(plan(self.data,self.bound,72)[0]);entry['guards'][-1]=self.cases[0]
        with self.assertRaises(ValueError):rt.gradients(opt,entry,1561,Path('unused'))

    def test_runner_smoke_and_zero_accepted_train(self):
        import run_small_room30_targeted_v2 as runner
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cp=root/'source.pt';cp.write_bytes(b'original1560');src=root/'source.json';ev=root/'eval.json';diag=root/'diag.json'
            source=dict(seed=72,last_checkpoint=dict(path=str(cp),sha256=sha(cp),step=1560))
            for p,value in [(src,source),(ev,{}),(diag,{})]:write_json(p,value)
            for mode in ('smoke','train'):
                argv=['run','--mode',mode,'--dataset',str(self.data.root),'--source-summary',str(src),
                    '--evaluation-summary',str(ev),'--diagnosis-summary',str(diag),'--output-dir',str(root/mode),'--device','cpu']
                if mode=='train':argv+=['--readiness',str(root/'smoke/summary.json'),'--allow-research-training']
                fake=dict(rows=[],adapter_digest='toy',decision={'all_four_repaired':False})
                def measured(rt,b,ev,out,**kw):
                    out.mkdir(parents=True);write_json(out/'summary.json',fake);return fake
                with patch.object(sys,'argv',argv),patch('pathlib.Path.cwd',return_value=runner.REPO), \
                        patch.object(runner,'source_contract',return_value=(source,{},self.bound,[])), \
                        patch.object(runner,'load_runtime',side_effect=lambda *a:RepairToy()), \
                        patch.object(runner,'check_transitions',return_value=[]), \
                        patch('small_room30_targeted_v2_runtime.reference',return_value=np.zeros((8192,6),np.float32)), \
                        patch('small_room30_targeted_v2_runtime.measured_canary',side_effect=measured), \
                        patch('small_room30_targeted_v2_runtime.baseline',return_value=fake), \
                        patch('small_room30_targeted_v2_runtime.acceptance',return_value=dict(accepted=False)),contextlib.redirect_stdout(io.StringIO()):runner.main()
            result=json.loads((root/'train/summary.json').read_text())
            self.assertEqual(result['status'],'TARGETED_V2_STOPPED_REJECTED');self.assertIsNone(result['last_checkpoint'])
            self.assertEqual(result['completed_step'],1560);self.assertEqual(cp.read_bytes(),b'original1560')

    def test_runner_saves_only_accepted_evaluated_weights(self):
        import run_small_room30_targeted_v2 as runner
        from small_room30_checkpoint_io import load_checkpoint
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cp=root/'source.pt';cp.write_bytes(b'original1560');src=root/'source.json';ev=root/'eval.json';diag=root/'diag.json'
            source=dict(seed=72,last_checkpoint=dict(path=str(cp),sha256=sha(cp),step=1560))
            for p,value in [(src,source),(ev,{}),(diag,{})]:write_json(p,value)
            ready=root/'ready.json';write_json(ready,dict(torch_version=str(torch.__version__),cuda_version=torch.version.cuda))
            argv=['run','--mode','train','--allow-research-training','--readiness',str(ready),'--dataset',str(self.data.root),
                '--source-summary',str(src),'--evaluation-summary',str(ev),'--diagnosis-summary',str(diag),
                '--output-dir',str(root/'train'),'--device','cpu']
            def measured(rt,b,ev,out,**kw):
                result=dict(rows=[],adapter_digest=state_digest(adapter_state(rt.model)),decision={'all_four_repaired':True})
                out.mkdir(parents=True);write_json(out/'summary.json',result);return result
            with patch.object(sys,'argv',argv),patch('pathlib.Path.cwd',return_value=runner.REPO), \
                    patch.object(runner,'source_contract',return_value=(source,{},self.bound,[])), \
                    patch.object(runner,'check_ready'),patch.object(runner,'load_runtime',side_effect=lambda *a:RepairToy()), \
                    patch('small_room30_targeted_v2_runtime.reference',return_value=np.zeros((8192,6),np.float32)), \
                    patch('small_room30_targeted_v2_runtime.measured_canary',side_effect=measured), \
                    patch('small_room30_targeted_v2_runtime.baseline',return_value={'rows':[]}), \
                    patch('small_room30_targeted_v2_runtime.acceptance',return_value=dict(accepted=True)),contextlib.redirect_stdout(io.StringIO()):runner.main()
            result=json.loads((root/'train/summary.json').read_text())
            self.assertEqual(result['accepted_updates'],1);self.assertEqual(result['completed_step'],1561)
            saved=load_checkpoint(result['last_checkpoint']['path'])
            self.assertEqual(saved['purpose'],'RESEARCH_TARGETED_V2_PILOT')
            self.assertEqual(state_digest(saved['adapter']),result['last_checkpoint']['adapter_digest'])
            self.assertFalse(saved['teacher_checkpoint_authorized']);self.assertEqual(cp.read_bytes(),b'original1560')


if __name__=='__main__':unittest.main(verbosity=2)
