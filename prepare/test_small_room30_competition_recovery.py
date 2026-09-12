"""Synthetic failure/recovery workflow; no actual user checkpoint available locally."""
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from small_room30_competition_data import POLICY
from small_room30_competition_model import CompetitionStudent
from small_room30_adm_dataset import sha
from train_small_room30_competition import CODE_FILES,save_model
from test_small_room30_competition import fixture
import recover_small_room30_competition_eval as recovery


def write(path,value):path.write_text(json.dumps(value))


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(14)
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.source=self.root/'failed';self.source.mkdir();(self.source/'maps').mkdir()
        self.previous=self.root/'previous';self.previous.mkdir()
        x,l=fixture();self.records=[];examples={}
        for prompt in ('sit_watch_v1','sit_write_v1'):
            for h in ('h0','h1'):
                r=dict(scene_id='synthetic',generation=2,prompt_id=prompt,history_motion_id=h,split='same_scene_noise_repeat',sample_id='s')
                self.records.append(r)
                inputs={k:v[0].detach().numpy().copy() for k,v in x.items() if k!='text_features'}
                if h=='h1':inputs['observed_history'][:,:2]+=2
                examples[(prompt,h)]=(inputs,{k:v[0].numpy().copy() for k,v in l.items()})
        self.text={p:x['text_features']+i for i,p in enumerate(('sit_watch_v1','sit_write_v1'))}
        binding=dict(records=self.records,teacher_checkpoint_sha256='fixture',teacher_summary_sha256='fixture',
            dataset_index_sha256='fixture',dataset_manifest_sha256='fixture')
        self.binding=binding
        class Data:
            records=self.records
            def __init__(inner,*args):inner.binding=binding
            def example(inner,r):return examples[(r['prompt_id'],r['history_motion_id'])]
            def geometry(inner,s):return (None,None,None,['a','b'])
        self.data=Data
        self.model=CompetitionStudent(width=16).eval();rows=[]
        with torch.no_grad():
            for i,r in enumerate(self.records):
                inputs,_=Data().example(r)
                o=self.model(**{k:torch.from_numpy(v[None]) for k,v in inputs.items()},text_features=self.text[r['prompt_id']])
                arrays={k:v[0].numpy() for k,v in o.items()};arrays.update(inputs)
                # Reproduce the observed terminal failure explicitly, not by claiming CUDA availability.
                arrays['slot_xy']=arrays['slot_xy'].copy();arrays['slot_xy'][0,0]+=i*.001
                np.savez_compressed(self.source/'maps'/('%03d.npz'%i),**arrays)
                name='%03d.npz'%i;np.savez_compressed(self.previous/name,**arrays)
                rows.append(dict(record=r,file=name,sha256=sha(self.previous/name)))
        write(self.previous/'manifest.json',dict(binding=binding))
        write(self.previous/'summary.json',dict(status='STUDENT_PILOT_COMPLETE_REVIEW_REQUIRED',manifest_sha256=sha(self.previous/'manifest.json'),rows=rows))
        self.manifest=dict(schema='small_room30_competition_run_v1',policy=POLICY,binding=binding,
            warmup_steps=1,joint_steps=2,clip_weights_sha256='fixture',
            comparison_summary_sha256=sha(self.previous/'summary.json'),
            code_sha256={n:sha(Path(recovery.__file__).parent/n) for n in CODE_FILES})
        write(self.source/'manifest.json',self.manifest)
        save_model(self.model,self.source,self.manifest)
        write(self.source/'training_log.json',[dict(step=3,phase='joint_competition')])
        write(self.source/'failure.json',dict(error="ValueError('Candidate localization condition leakage')"))

    def tearDown(self):self.tmp.cleanup()

    def test_saved_pair_audit_reports_difference_and_swap(self):
        result=recovery.audit_saved_pairs(self.source,self.records)
        self.assertGreater(result['failed_slot_pairs'],0)
        self.assertTrue(all(r['identical_points'] for r in result['rows']))
        self.assertGreater(max(r['direct_slot_max_error_m'] for r in result['rows']),.0009)

    def test_incomplete_training_rejected(self):
        write(self.source/'training_log.json',[dict(step=2,phase='joint_competition')])
        with self.assertRaisesRegex(ValueError,'did not finish'):recovery.verify_completed_training(self.source)

    def test_changed_weights_rejected(self):
        with (self.source/'student_weights.npz').open('ab') as f:f.write(b'changed')
        with self.assertRaisesRegex(ValueError,'weights changed'):recovery.verify_completed_training(self.source)

    def test_different_failure_not_silently_recovered(self):
        write(self.source/'failure.json',dict(error="FloatingPointError('Nonfinite loss')"))
        with self.assertRaisesRegex(ValueError,'only handles'):recovery.verify_completed_training(self.source)

    def test_cpu_repeat_detects_real_instability(self):
        class Unstable:
            n=0
            def __call__(inner,**inputs):
                inner.n+=1
                return {k:torch.tensor(inner.n) for k in ('slot_xy','support','relation_score','history_score','a_w')}
        with self.assertRaisesRegex(ValueError,'instability'):
            recovery.verify_repeat(Unstable(),self.data(),self.text,self.records)

    def test_full_recovery_main_no_optimizer_no_source_writes(self):
        before={str(p.relative_to(self.source)):sha(p) for p in self.source.rglob('*') if p.is_file()}
        out=self.root/'recovered'
        argv=['recover','--source-dir',str(self.source),'--output-dir',str(out),'--comparison-summary',str(self.previous/'summary.json')]
        with patch('sys.argv',argv),patch.object(recovery,'CompetitionData',self.data),\
             patch.object(recovery,'load_baseline_without_pickle',return_value=({},self.text,'fixture')),\
             patch('torch.load',side_effect=AssertionError('No pickle')),\
             patch('torch.optim.Adam',side_effect=AssertionError('No optimizer')):
            recovery.main()
        report=json.loads((out/'summary.json').read_text())
        self.assertEqual(report['status'],'COMPETITION_COMPLETE_REVIEW_REQUIRED')
        self.assertFalse(report['approved']);self.assertEqual(report['evaluation_recovery']['optimizer_steps'],0)
        self.assertFalse(report['evaluation_recovery']['checks_relaxed'])
        self.assertEqual(len(report['rows']),4)
        for n,digest in before.items():self.assertEqual(sha(self.source/n),digest)
        self.assertEqual(sha(out/'student_weights.npz'),sha(self.source/'student_weights.npz'))


if __name__=='__main__':unittest.main(verbosity=2)
