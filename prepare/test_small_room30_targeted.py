"""CPU contract/integration tests, not real pretrained ADM performance evidence."""
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
from small_room30_targeted import FAILURES,POLICY,plan,check_ready,source_contract,validate_run
from small_room30_targeted_runtime import TargetedRuntime,trace_case,regression_decision,canary
from small_room30_adm_dataset import SmallRoom30ADM,sha
from small_room30_evaluation_common import panel,seeds,conditioning_arrays
from small_room30_training_runtime import frozen_digest,adapter_state,state_digest
from small_room30_training_contract import write_json
from evaluate_small_room30_onpolicy import rollout
from fewshot_cdm_lora import lora_named_parameters


class RepairToy(ToyRuntime):
    update_targeted=TargetedRuntime.update_targeted


class TargetedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=SmallRoom30ADM(os.environ.get('SMALL_ROOM30_DATASET','data/small_room30_adm_lora_v1'))
        ToyRuntime.data_source=cls.data
        allcases=panel(cls.data,'full')
        cls.failures=[next(c for c in allcases if c['filename']==n) for n in FAILURES]
        cls.guards=[]
        for action in ('sit','lie','write_board'):
            rooms=set()
            for c in allcases:
                if c['action']==action and c['scene_id'] not in rooms:
                    rooms.add(c['scene_id']);cls.guards.append(dict(case=c))
                    if len(rooms)==4:break
        cls.bound=dict(guards=cls.guards)
        cls.trace=[dict(case=c,selected_timesteps=[499,490,0]) for c in cls.failures]

    def test_failure_indices_and_seeds(self):
        self.assertEqual([c['index'] for c in self.failures],[22,41,41,71])
        self.assertNotEqual(seeds(self.failures[1]),seeds(self.failures[2]))

    def test_bounded_plan_and_exact_paths(self):
        entries=plan(self.data,self.bound,self.trace,12,72)
        for i,e in enumerate(entries):
            self.assertEqual(e['case'],self.failures[i%4]);self.assertEqual(e['timestep'],[499,490,0][i//4])
        for count in (0,1,13,60):
            with self.assertRaises(ValueError):plan(self.data,self.bound,self.trace,count,72)

    def test_replay_all30_rooms_and_three_actions(self):
        entries=plan(self.data,self.bound,self.trace,12,72)
        rooms=set()
        for e in entries:
            for g in e['replay']:
                self.assertEqual([self.data.samples[i]['action'] for i in g],['sit','lie','write_board'])
                rooms.add(self.data.samples[g[0]]['scene_id'])
        self.assertEqual(len(rooms),30)
        self.assertEqual([e['guard']['action'] for e in entries],['sit','lie','write_board']*4)
        self.assertEqual(len({e['guard']['filename'] for e in entries}),12)

    def test_trace_tampering_rejected(self):
        bad=copy.deepcopy(self.trace);bad[0]['selected_timesteps']=[499,499,499]
        with self.assertRaises(ValueError):plan(self.data,self.bound,bad,12,72)
        with self.assertRaises(ValueError):plan(self.data,self.bound,list(reversed(self.trace)),12,72)

    def test_readiness_fail_closed(self):
        for r in ({},{'status':'TARGETED_RUNTIME_READY','binding':{}}):
            with self.assertRaises(ValueError):check_ready(r,{})

    def test_trace_replays_sampler_and_nonzero_gradient(self):
        rt=RepairToy();case=self.failures[0];_,kw=conditioning_arrays(self.data,case['index'])
        before=(state_digest(adapter_state(rt.model)),frozen_digest(rt.model))
        with contextlib.redirect_stdout(io.StringIO()):
            expected=rollout(rt,kw,*seeds(case),label='toy')
            result=trace_case(rt,case,expected)
        self.assertEqual(result['replay_max_abs_error'],0)
        self.assertEqual(len(result['records']),len(POLICY['trace_timesteps']))
        self.assertEqual(len(result['selected_timesteps']),3)
        for r in result['records']:
            if r['timestep'] in result['selected_timesteps']:self.assertGreater(r['corrective_gradient_norm'],0)
        self.assertEqual(before,(state_digest(adapter_state(rt.model)),frozen_digest(rt.model)))
        self.assertTrue(all(p.grad is None for p in rt.model.parameters()))

    def test_trace_wrong_final_rejected(self):
        with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(ValueError):
            trace_case(RepairToy(),self.failures[0],np.ones((8192,6),np.float32)*99)

    def test_targeted_update_only_lora(self):
        rt=RepairToy();entry=plan(self.data,self.bound,self.trace,12,72)[0]
        original=frozen_digest(rt.model);before=state_digest(adapter_state(rt.model))
        opt=torch.optim.AdamW(list(lora_named_parameters(rt.model).values()),lr=POLICY['lr'])
        with patch('small_room30_targeted_runtime.reference',return_value=np.zeros((8192,6),np.float32)),contextlib.redirect_stdout(io.StringIO()):
            row=rt.update_targeted(opt,entry,1561,Path('unused'))
        self.assertEqual(original,frozen_digest(rt.model));self.assertNotEqual(before,state_digest(adapter_state(rt.model)))
        self.assertGreater(row['gradient_norm'],0);self.assertEqual(len(row['replay']),9)
        self.assertEqual(row['seeds'],seeds(entry['case']))

    def rows(self):
        metric=dict(mae=.2,background_point_hit_fraction=.1,all_furniture_present=True)
        return [dict(case=r['case'],role='guard',before=metric.copy(),after=metric.copy()) for r in self.guards]+[
            dict(case=c,role='failure',before=dict(metric,all_furniture_present=False),after=metric.copy()) for c in self.failures]

    def test_canary_primary_presence_separate_from_gt(self):
        result=regression_decision(self.rows())
        self.assertTrue(result['all_four_repaired']);self.assertFalse(result['stop_for_regression'])
        self.assertFalse(result['teacher_checkpoint_authorized'])

    def test_lost_success_stops(self):
        rows=self.rows();rows[0]['after']['all_furniture_present']=False
        self.assertTrue(regression_decision(rows)['stop_for_regression'])

    def test_action_mae_and_background_warnings_stop(self):
        for key,delta in [('mae',.2),('background_point_hit_fraction',.4)]:
            rows=self.rows();rows[4]['after'][key]+=delta
            decision=regression_decision(rows)
            self.assertTrue(decision['stop_for_regression']);self.assertEqual(decision['regression_warnings'][0]['action'],'lie')

    def test_incomplete_guard_panel_rejected(self):
        with self.assertRaises(ValueError):regression_decision(self.rows()[1:])

    def test_canary_checks_frozen_and_saves_arrays(self):
        rt=RepairToy();metric=self.rows()[0]['before'];raw=np.zeros((8192,6),np.float32)
        bound=dict(guards=[dict(r,score=metric) for r in self.guards],failures=[dict(case=c,score=metric) for c in self.failures])
        with tempfile.TemporaryDirectory() as d,patch('small_room30_targeted_runtime.rollout',return_value=raw), \
                patch('small_room30_targeted_runtime.reference',return_value=raw), \
                patch('small_room30_targeted_runtime.score',return_value=metric),contextlib.redirect_stdout(io.StringIO()):
            out=Path(d)/'canary';result=canary(rt,bound,Path('unused'),out,reproduce=True)
            self.assertEqual(len(result['rows']),16);self.assertEqual(len(list(out.glob('*.npz'))),16)
            self.assertTrue((out/'summary.json').is_file())

    def test_wrong_source_checkpoint_rejected(self):
        with patch('small_room30_targeted.validate_parent',return_value=(dict(completed_step=1500),{},{})):
            with self.assertRaises(ValueError):source_contract(self.data,Path('a'),Path('b'),Path('c'))

    def test_runner_smoke_train_and_stop_roundtrip(self):
        import run_small_room30_targeted as runner
        from small_room30_checkpoint_io import load_checkpoint
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cp=root/'source.pt';cp.write_bytes(b'untouched');sourcepath=root/'source.json';ev=root/'eval.json'
            source=dict(seed=72,last_checkpoint=dict(path=str(cp),sha256=sha(cp),step=1560))
            write_json(sourcepath,source);write_json(ev,{'toy':True})
            bound=dict(self.bound,failures=[dict(case=c) for c in self.failures])
            decision=dict(all_four_repaired=False,stop_for_regression=True)
            for mode in ('smoke','train'):
                out=root/mode
                argv=['run','--mode',mode,'--dataset',str(self.data.root),'--source-summary',str(sourcepath),
                    '--evaluation-summary',str(ev),'--output-dir',str(out),'--device','cpu']
                if mode=='train':argv+=['--readiness',str(root/'smoke/summary.json'),'--allow-research-training']
                with patch.object(sys,'argv',argv),patch('pathlib.Path.cwd',return_value=runner.REPO), \
                        patch.object(runner,'source_contract',return_value=(source,{},bound)), \
                        patch.object(runner,'load_runtime',side_effect=lambda *a:RepairToy()), \
                        patch('small_room30_targeted_runtime.reference',return_value=np.zeros((8192,6),np.float32)), \
                        patch('small_room30_targeted_runtime.trace_case',side_effect=lambda rt,c,r:dict(case=c,selected_timesteps=[499,490,0])), \
                        patch('small_room30_targeted_runtime.canary',return_value=dict(decision=decision)),contextlib.redirect_stdout(io.StringIO()):
                    runner.main()
            report=json.loads((root/'train/summary.json').read_text())
            self.assertEqual(report['status'],'TARGETED_STOPPED_REGRESSION');self.assertEqual(report['completed_step'],1564)
            self.assertEqual(cp.read_bytes(),b'untouched')
            saved=load_checkpoint(report['last_checkpoint']['path'])
            self.assertEqual(saved['purpose'],'RESEARCH_TARGETED_PILOT');self.assertFalse(saved['teacher_checkpoint_authorized'])


if __name__=='__main__':unittest.main(verbosity=2)
