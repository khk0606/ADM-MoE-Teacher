"""Anywhere extension contract and synthetic end-to-end tests, not quality claims."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from small_room30_anywhere_data import ANYWHERE,TEXTS,AnywhereData
from small_room30_competition_data import CompetitionData,compete
from small_room30_anywhere_text import encode_texts,prepare_text
import test_small_room30_competition_recovery as recovery_fixture
from test_small_room30_competition_recovery import write
from train_small_room30_competition import load_model
from small_room30_adm_dataset import sha
import train_small_room30_anywhere as runner


class AnywhereTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(14)

    def test_real_sentence_not_prompt_id(self):
        self.assertEqual(TEXTS[ANYWHERE],'Sit on something');self.assertEqual(len(TEXTS),5)

    def test_anywhere_removes_purpose_not_history_or_support(self):
        from test_small_room30_competition import fixture
        _,l=fixture();base={k:v[0].numpy() for k,v in l.items()}
        data=AnywhereData.__new__(AnywhereData)
        with patch.object(AnywhereData,'reference_prompt',return_value='sit_write_v1'),\
             patch.object(CompetitionData,'corrected_labels',return_value=base):
            new=data.corrected_labels('fixture',ANYWHERE,'h')
        np.testing.assert_array_equal(new['candidate_relation'],[1,1])
        self.assertFalse(new['purpose_classes'].any());self.assertFalse(new['purpose_mask'].any())
        for key in ('history_scores','support_target','floor_support_weight','candidate_masks'):
            np.testing.assert_array_equal(base[key],new[key])
        self.assertFalse(np.array_equal(base['candidate_relation'],new['candidate_relation']))

    def test_equal_relation_history_controls_choice_and_ties_allowed(self):
        r=np.ones(2,np.float32);h=np.array([.99,.01],np.float32)
        self.assertGreater(compete(r,h)[1][0],.94)
        self.assertGreater(compete(r,h[::-1])[1][1],.94)
        np.testing.assert_allclose(compete(r,np.array([.5,.5]))[1],[.5,.5])

    def test_clip_file_guard_before_deserialization(self):
        with tempfile.TemporaryDirectory() as tmp,patch('torch.jit.load',side_effect=AssertionError('Unsafe load')):
            path=Path(tmp)/'ViT-B-32.pt'
            with self.assertRaises(FileNotFoundError):encode_texts(path,[TEXTS[ANYWHERE]])
            path.write_bytes(b'untrusted')
            with self.assertRaisesRegex(ValueError,'checkpoint differs'):encode_texts(path,[TEXTS[ANYWHERE]])

    def test_numeric_text_contract_preserves_legacy_exact(self):
        keys=sorted(TEXTS);vectors=torch.nn.functional.normalize(torch.randn(5,512),dim=-1)
        old={k:vectors[keys.index(k):keys.index(k)+1].clone() for k in keys if k!=ANYWHERE}
        with patch('small_room30_anywhere_text.load_baseline_without_pickle',return_value=({},dict(old),'fixture')),\
             patch('small_room30_anywhere_text.encode_texts',return_value=(vectors,{'fixture':True})):
            actual,info=prepare_text(None,None,None)
        for k in old:self.assertTrue(torch.equal(old[k],actual[k]))
        self.assertTrue(torch.equal(actual[ANYWHERE],vectors[keys.index(ANYWHERE):keys.index(ANYWHERE)+1]))

    def test_mismatched_encoder_feature_rejected(self):
        keys=sorted(TEXTS);vectors=torch.nn.functional.normalize(torch.randn(5,512),dim=-1)
        old={k:-vectors[keys.index(k):keys.index(k)+1] for k in keys if k!=ANYWHERE}
        with patch('small_room30_anywhere_text.load_baseline_without_pickle',return_value=({},old,'fixture')),\
             patch('small_room30_anywhere_text.encode_texts',return_value=(vectors,{})):
            with self.assertRaisesRegex(ValueError,'does not match'):prepare_text(None,None,None)

    def test_synthetic_cli_finetune_and_cpu_eval_keep_original(self):
        fixture=recovery_fixture.RecoveryTests();fixture.setUp()
        try:
            root=fixture.source;olddata=fixture.data();legacy=fixture.binding
            rows=[dict(record=r,file='maps/%03d.npz'%i,sha256=sha(root/'maps'/('%03d.npz'%i))) for i,r in enumerate(fixture.records)]
            write(root/'summary.json',dict(status='COMPETITION_COMPLETE_REVIEW_REQUIRED',
                manifest_sha256=sha(root/'manifest.json'),model_sha256=sha(root/'student_model.json'),
                student_sha256=sha(root/'student_weights.npz'),rows=rows))
            records=list(fixture.records)+[dict(r,prompt_id=ANYWHERE) for r in fixture.records if r['prompt_id']=='sit_watch_v1']
            records+= [dict(r,generation=g,split='train') for g in (0,1) for r in records.copy()]
            class Data:
                def __init__(self,*args):
                    self.records=records;self.legacy_binding=legacy;self.binding=dict(legacy,records=records)
                def geometry(self,scene):return (None,None,None,['a','b'])
                def example(self,r):
                    raw=dict(r,prompt_id='sit_watch_v1') if r['prompt_id']==ANYWHERE else r
                    x,l=olddata.example(raw);x=copy.deepcopy(x);l=copy.deepcopy(l)
                    l['distance_matrix']=np.zeros((2,3),np.float32)
                    if r['prompt_id']==ANYWHERE:
                        l['candidate_relation']=np.ones(2,np.float32);l['purpose_classes']*=0;l['purpose_mask']*=0
                        l['score_target'],l['selection_target']=compete(l['candidate_relation'],l['history_scores'])
                        l['target_weight']=(l['support_target'][:,:2]*l['selection_target']).sum(-1)
                    return x,l
            text={k:torch.nn.functional.normalize(torch.randn(1,512),dim=-1) for k in TEXTS}
            before=sha(root/'student_weights.npz');out=fixture.root/'anywhere'
            argv=['anywhere','--initial-summary',str(root/'summary.json'),'--output-dir',str(out),'--device','cpu','--joint-steps','2']
            with patch('sys.argv',argv),patch.object(runner,'CompetitionData',Data),\
                 patch.object(runner,'prepare_text',return_value=(text,{'synthetic_fixture_only':True})),\
                 patch('torch.load',side_effect=AssertionError('No pickle')):
                runner.main()
            report=json.loads((out/'summary.json').read_text())
            self.assertEqual(report['status'],'ANYWHERE_COMPLETE_REVIEW_REQUIRED');self.assertFalse(report['approved'])
            self.assertEqual(report['evaluation_device'],'cpu');self.assertEqual(len(report['rows']),6)
            self.assertEqual(sha(root/'student_weights.npz'),before)
            self.assertNotEqual(sha(out/'student_weights.npz'),before)
            load_model(out/'student_model.json')
            for row in report['rows']:
                with np.load(out/row['file']) as z:
                    flag=row['record']['prompt_id']==ANYWHERE
                    self.assertEqual(bool(z['baseline_is_teacher']),flag)
                    if flag:np.testing.assert_array_equal(z['baseline_aw'],z['teacher_a'])
        finally:fixture.tearDown()


if __name__=='__main__':unittest.main(verbosity=2)
