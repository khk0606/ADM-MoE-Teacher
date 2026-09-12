"""CPU evaluation-only tests; synthetic rollout integration is not real ADM validation."""
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
from small_room30_training_contract import SCHEMA as TRAIN_SCHEMA,write_json
from small_room30_training_runtime import adapter_state
from small_room30_evaluation_common import panel,conditioning_arrays,supervision,case_metrics,seeds,validate_report
from small_room30_v5_full import comparison,original_case,verify_reviewed_path,validate_run
from test_small_room30_onpolicy import ToyRuntime


class FullTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=SmallRoom30ADM(os.environ.get('SMALL_ROOM30_DATASET','data/small_room30_adm_lora_v1'))
        ToyRuntime.data_source=cls.data

    def arrays(self,c):
        points,_=conditioning_arrays(self.data,c['index']);gt,masks,ids=supervision(self.data,c['index'])
        return dict(points=points,gt=gt,active_masks=masks,motion_ids=np.asarray(ids),base_raw=gt*.2,adapted_raw=gt*.7)

    def test_full_scope240(self):
        cases=panel(self.data,'full')
        self.assertEqual(len(cases),240);self.assertEqual(len({c['scene_id'] for c in cases}),30)
        self.assertEqual(len({c['sample_id'] for c in cases}),80)
        self.assertEqual({c['generation'] for c in cases},{0,1,2})

    def test_unapproved_run_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'summary.json';write_json(p,{})
            with self.assertRaises(ValueError):validate_run(self.data,p,Path(d))

    def test_original_base_reuse_checks_hash_seeds_geometry(self):
        c=panel(self.data,'full')[0];a=self.arrays(c)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'cases').mkdir();f=root/'cases'/c['filename']
            np.savez_compressed(f,**a)
            row=dict(case=c,sha256=sha(f),seeds=seeds(c),metrics=case_metrics(a))
            write_json(root/'summary.json',dict(rows=[row]))
            np.testing.assert_array_equal(original_case(self.data,root/'summary.json',c)['base_raw'],a['base_raw'])
            row['seeds']=[1,2];write_json(root/'summary.json',dict(rows=[row]))
            with self.assertRaises(ValueError):original_case(self.data,root/'summary.json',c)
            row['seeds']=seeds(c);a['points']=a['points']+1;np.savez_compressed(f,**a)
            row['sha256']=sha(f);write_json(root/'summary.json',dict(rows=[row]))
            with self.assertRaises(ValueError):original_case(self.data,root/'summary.json',c)

    def test_pairing_changes_rejected(self):
        c=panel(self.data,'full')[0];a=self.arrays(c);b=copy.deepcopy(a)
        self.assertFalse(comparison(self.data,c,a,b)['lost_original_targets'])
        b['base_raw']+=.1
        with self.assertRaises(ValueError):comparison(self.data,c,a,b)

    def test_reviewed1578_must_reproduce(self):
        c=panel(self.data,'full')[0];raw=self.arrays(c)['adapted_raw']
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/'proposal_12_try_1';folder.mkdir();f=folder/c['filename']
            np.savez_compressed(f,adapted_raw=raw)
            write_json(folder/'summary.json',dict(rows=[dict(case=c,sha256=sha(f))]))
            self.assertEqual(verify_reviewed_path(root/'summary.json',c,raw),0)
            with self.assertRaises(ValueError):verify_reviewed_path(root/'summary.json',c,raw+.1)

    def test_new_paths_do_not_claim_reproduction(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'proposal_12_try_1').mkdir()
            write_json(root/'proposal_12_try_1/summary.json',dict(rows=[]))
            self.assertIsNone(verify_reviewed_path(root/'summary.json',panel(self.data,'full')[0],np.zeros((8192,6))))

    def test_no_optimizer_and_safe_resume_end_to_end(self):
        import evaluate_small_room30_v5_full as runner
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cp=root/'adapter.pt';cp.write_bytes(b'fixture1578')
            src=root/'training.json';write_json(src,{})
            template=ToyRuntime()
            bound=dict(original_binding=dict(inputs={}))
            training=dict(last_checkpoint=dict(path=str(cp),sha256=sha(cp),step=1578),seed=72,
                completed_step=1578,evaluation_summary='unused')
            saved=dict(schema=TRAIN_SCHEMA,purpose='RESEARCH_TARGETED_V5_PILOT',binding=bound,seed=72,
                step=1578,teacher_checkpoint_authorized=False,base_digest=template.base_digest,
                encoder_weight={},adapter=adapter_state(template.model))
            argv=['eval','--training-summary',str(src),'--dataset',str(self.data.root),'--mode','preview',
                '--output-dir',str(root/'eval'),'--device','cpu','--resume']
            def original(data,summary,case):return self.arrays(case)
            with patch.object(sys,'argv',argv),patch('pathlib.Path.cwd',return_value=runner.REPO), \
                    patch('small_room30_v5_full.validate_run',return_value=(training,{},bound)), \
                    patch('small_room30_checkpoint_io.load_checkpoint',return_value=saved), \
                    patch('small_room30_training_runtime.Runtime',ToyRuntime), \
                    patch('torch.cuda.get_device_name',return_value='CPU SYNTHETIC TEST'), \
                    patch('torch.optim.AdamW',side_effect=AssertionError('No optimizer allowed')), \
                    patch('small_room30_v5_full.original_case',side_effect=original), \
                    patch('small_room30_v5_full.verify_reviewed_path',return_value=0), \
                    patch.object(runner,'rollout',return_value=np.zeros((8192,6),np.float32)) as call, \
                    contextlib.redirect_stdout(io.StringIO()):
                runner.main();self.assertEqual(call.call_count,16) # Not32: no Frozen Base rerun.
                validate_report(root/'eval/summary.json')
                runner.main();self.assertEqual(call.call_count,16) # Completed evaluation resumes read-only.
                # Remove only synthetic test's completion marker to exercise per-case cache.
                (root/'eval/summary.json').unlink()
                runner.main();self.assertEqual(call.call_count,16)
                (root/'eval/summary.json').unlink()
                c=panel(self.data,'preview')[0];rpath=root/'eval/cases'/Path(c['filename']).with_suffix('.json')
                row=json.loads(rpath.read_text());row['sha256']='bad';write_json(rpath,row)
                with self.assertRaises(ValueError):runner.main()
            self.assertEqual(cp.read_bytes(),b'fixture1578')


if __name__=='__main__':unittest.main()
