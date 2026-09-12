"""CPU contract/gradient/rollback/runner tests, not pretrained-model repair proof."""
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from small_room30_adm_dataset import SmallRoom30ADM, sha
from small_room30_evaluation_common import panel
from small_room30_targeted_v4 import POLICY, FAILURES, instance_support, presence, acceptance, plan, check_ready
from small_room30_targeted_v4_runtime import RepairRuntime, missing_loss, preservation_loss, transactional_step, tree_equal
from small_room30_training_runtime import adapter_state, state_digest, frozen_digest
from small_room30_training_contract import write_json
from fewshot_cdm_lora import lora_named_parameters
from test_small_room30_onpolicy import ToyRuntime


class RepairToy(ToyRuntime):
    gradients = RepairRuntime.gradients


class V4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = SmallRoom30ADM(os.environ.get('SMALL_ROOM30_DATASET', 'data/small_room30_adm_lora_v1'))
        ToyRuntime.data_source = cls.data
        cases = panel(cls.data, 'full')
        cls.cases = [next(c for c in cases if c['filename'] == n) for n in FAILURES]
        guards = []
        for action in ('sit', 'lie', 'write_board'):
            rooms = set()
            for c in cases:
                if c['action'] == action and c['filename'] not in FAILURES and c['scene_id'] not in rooms:
                    rooms.add(c['scene_id']); guards.append(dict(case=c))
                    if len(rooms) == 4: break
        cls.bound = dict(transitions=[dict(case=c, timesteps=ts) for c, ts in zip(cls.cases,
            [[490,475,0],[300,200,0],[475,450,0],[495,490,0]])], guards=guards,
            failures=[dict(case=c) for c in cls.cases])

    def reports(self):
        rows = []
        for i, item in enumerate(self.bound['failures'] + self.bound['guards']):
            c = item['case']; masks, names, _ = instance_support(self.data, c['index'])
            raw = np.ones((8192, 6), np.float32) * .7
            if i < 4: raw[masks[names.index(c['scene_id']+'__100__sit')]] = .1
            pr = presence(raw, masks, names)
            rows.append(dict(case=c, role='failure' if i<4 else 'guard', before_presence=pr,
                after_presence=copy.deepcopy(pr)))
        previous = dict(rows=rows)
        candidate = copy.deepcopy(previous)
        target = next(r for r in candidate['rows'][0]['after_presence']['instances'] if '__100__sit' in r['id'])
        target['top_mean'] += .01
        return candidate, previous

    def test_plan_only_three_rooms_four_exact_paths(self):
        entries = plan(self.data, self.bound, 72)
        self.assertEqual(len(entries), 12)
        self.assertEqual({e['case']['filename'] for e in entries}, set(FAILURES))
        self.assertEqual({e['case']['scene_id'] for e in entries}, set(POLICY['target_rooms']))
        self.assertTrue(all('replay' not in e for e in entries))
        self.assertEqual(len({g['filename'] for e in entries[:4] for g in e['guards']}), 12)

    def test_out_of_scope_plan_rejected(self):
        b = copy.deepcopy(self.bound); b['transitions'][0]['case']['scene_id'] = 'small_room_0101'
        with self.assertRaises(ValueError): plan(self.data, b, 72)

    def test_whole_instance_masks_no_gt_extent_or_body_channel(self):
        for index in range(len(self.data)):
            masks, names, targets = instance_support(self.data, index)
            self.assertEqual(masks.shape, (len(names), 8192))
            self.assertTrue((masks.sum(axis=0) <= 1).all())
            self.assertTrue(all(m.sum() >= 16 for m in masks))
            self.assertEqual(len(set(targets)), len(names))

    def test_any_joint_presence_uses_all_channels(self):
        mask = np.zeros((1,8192), bool); mask[0,:16] = True
        raw = np.zeros((8192,6), np.float32); raw[:4,5] = .4
        p = presence(raw, mask, ['chair'])
        self.assertTrue(p['all_furniture_present']); self.assertEqual(p['instances'][0]['hit_fraction'], .25)
        raw[3,5] = 0
        self.assertFalse(presence(raw, mask, ['chair'])['all_furniture_present'])

    def test_nonfinite_map_rejected(self):
        masks, names, _ = instance_support(self.data, self.cases[0]['index'])
        raw = np.zeros((8192,6)); raw[0,0] = np.nan
        with self.assertRaises(ValueError): presence(raw, masks, names)

    def test_loss_gradient_only_on_missing_instance(self):
        physical = torch.full((1,8192,6), .1, requires_grad=True)
        mask = torch.zeros(8192, dtype=torch.bool); mask[:64] = True
        loss = missing_loss(physical, mask); loss.backward()
        self.assertTrue((physical.grad[0,:64] < 0).any())
        self.assertEqual(float(physical.grad[0,64:].abs().sum()), 0.)

    def test_margin_satisfied_no_extra_gt_fitting(self):
        physical = torch.full((1,8192,6), .8, requires_grad=True)
        mask = torch.ones(8192, dtype=torch.bool)
        self.assertEqual(float(missing_loss(physical, mask)), 0.)

    def test_preservation_does_not_fill_source_missing_objects(self):
        old = torch.full((1,8192,6), .1)
        now = torch.zeros_like(old, requires_grad=True)
        masks = torch.ones((1,8192), dtype=torch.bool)
        self.assertEqual(float(preservation_loss(now, old, masks)), 0.)

    def test_preservation_allows_intensity_reduction_above_floor(self):
        old = torch.ones(1,8192,6); now = torch.full_like(old, .6, requires_grad=True)
        masks = torch.ones((1,8192), dtype=torch.bool)
        self.assertEqual(float(preservation_loss(now, old, masks)), 0.)

    def test_preservation_only_target_points_not_background(self):
        old = torch.ones(1,8192,6); now = torch.zeros_like(old, requires_grad=True)
        masks = torch.zeros((1,8192), dtype=torch.bool); masks[0,:64] = True
        preservation_loss(now, old, masks).backward()
        self.assertGreater(float(now.grad[0,:64].abs().sum()), 0.)
        self.assertEqual(float(now.grad[0,64:].abs().sum()), 0.)

    def test_legacy_gt_drop_does_not_veto_presence(self):
        c, p = self.reports(); c['decision'] = {'stop_for_regression': True}
        for row in c['rows'][4:]:
            for target in row['after_presence']['instances']:
                target['top_mean'] = .35; target['hit_fraction'] = .26
        self.assertTrue(acceptance(c, p, self.cases[0])['accepted'])

    def test_losing_individual_target_rejected_even_when_other_already_missing(self):
        c, p = self.reports()
        good = next(i for i in c['rows'][1]['after_presence']['instances'] if i['present'])
        good['present'] = False
        self.assertFalse(acceptance(c, p, self.cases[0])['accepted'])

    def test_newly_repaired_target_cannot_be_lost(self):
        c, p = self.reports()
        missing = next(i for i in p['rows'][1]['after_presence']['instances'] if not i['present'])
        missing['present'] = True
        self.assertFalse(acceptance(c, p, self.cases[0])['accepted'])

    def test_no_actual_progress_rejected(self):
        _, p = self.reports()
        self.assertFalse(acceptance(copy.deepcopy(p), p, self.cases[0])['accepted'])

    def test_presence_improvement_can_pass_without_mean_requirement(self):
        c, p = self.reports()
        missing = next(i for i in c['rows'][0]['after_presence']['instances'] if not i['present'])
        missing['present'] = True; missing['top_mean'] = .1
        self.assertTrue(acceptance(c, p, self.cases[0])['accepted'])

    def test_changed_panel_rejected(self):
        c, p = self.reports(); c['rows'].reverse()
        with self.assertRaises(ValueError): acceptance(c, p, self.cases[0])

    def prepared(self):
        rt = RepairToy(); params = list(lora_named_parameters(rt.model).values())
        opt = torch.optim.AdamW(params, lr=POLICY['lr'], weight_decay=0.)
        for param in params: param.grad = torch.ones_like(param)*.01
        opt.step()
        for param in params: param.grad = torch.ones_like(param)*.01
        return rt, opt

    def test_both_rejections_restore_nonempty_optimizer(self):
        rt, opt = self.prepared(); old = state_digest(adapter_state(rt.model)); moments = copy.deepcopy(opt.state_dict())
        result = transactional_step(rt, opt, lambda n,s: dict(accepted=False))
        self.assertFalse(result['accepted']); self.assertEqual(len(result['attempts']), 2)
        self.assertEqual(old, state_digest(adapter_state(rt.model))); self.assertTrue(tree_equal(moments, opt.state_dict()))

    def test_exception_rolls_back(self):
        rt, opt = self.prepared(); old = state_digest(adapter_state(rt.model)); moments = copy.deepcopy(opt.state_dict())
        def fail(n,s): raise RuntimeError('test')
        with self.assertRaises(RuntimeError): transactional_step(rt, opt, fail)
        self.assertEqual(old, state_digest(adapter_state(rt.model))); self.assertTrue(tree_equal(moments, opt.state_dict()))

    def test_real_toy_gradient_no_replay_or_gt_objective(self):
        rt, opt = self.prepared(); before = state_digest(adapter_state(rt.model)); frozen = frozen_digest(rt.model)
        with patch.object(rt, 'objective', side_effect=AssertionError('GT objective forbidden')), \
                patch('small_room30_targeted_v4_runtime.reference', return_value=np.full((8192,6), .7, np.float32)), \
                contextlib.redirect_stdout(io.StringIO()):
            row = rt.gradients(opt, plan(self.data, self.bound, 72)[0], 1561, Path('unused'))
        self.assertEqual(before, state_digest(adapter_state(rt.model))); self.assertEqual(frozen, frozen_digest(rt.model))
        self.assertEqual(len(row['target_terms']), 3); self.assertEqual(len(row['preservation_terms']), 4)
        self.assertFalse(row['gt_extent_loss']); self.assertFalse(row['cross_room_gt_replay'])
        self.assertGreater(row['gradient_norm_before_clip'], 0)

    def test_old_readiness_rejected(self):
        with self.assertRaises(ValueError): check_ready({'status':'TARGETED_V3_RUNTIME_READY'}, self.bound)

    def test_nonoriginal_checkpoint_rejected(self):
        from run_small_room30_targeted_v4 import load_runtime
        with patch('small_room30_checkpoint_io.load_checkpoint', return_value=dict(step=1562)):
            with self.assertRaises(ValueError): load_runtime(self.data, {'last_checkpoint':{'path':'unused'}}, {}, 'cpu')

    def test_runner_smoke_reject_and_saved_candidate(self):
        import run_small_room30_targeted_v4 as runner
        from small_room30_checkpoint_io import load_checkpoint
        for accept in (False, True):
            with tempfile.TemporaryDirectory() as d:
                root = Path(d); cp = root/'source.pt'; cp.write_bytes(b'original1560')
                source = dict(seed=72, last_checkpoint=dict(path=str(cp), sha256=sha(cp), step=1560))
                src, ev, diag = [root/n for n in ('source.json','eval.json','diag.json')]
                for p, value in ((src,source),(ev,{}),(diag,{})): write_json(p,value)
                _, previous = self.reports()
                def measured(rt,b,ev,out,**kw):
                    report = copy.deepcopy(previous)
                    report.update(adapter_digest=state_digest(adapter_state(rt.model)), decision={'all_four_repaired':accept})
                    out.mkdir(parents=True); write_json(out/'summary.json',report); return report
                def gradients(rt,*args):
                    for param in lora_named_parameters(rt.model).values(): param.grad=torch.ones_like(param)*.01
                    return {'case':'toy'}
                for mode in ('smoke','train'):
                    argv=['run','--mode',mode,'--dataset',str(self.data.root),'--source-summary',str(src),
                        '--evaluation-summary',str(ev),'--diagnosis-summary',str(diag),'--output-dir',str(root/mode),'--device','cpu']
                    if mode=='train': argv += ['--readiness',str(root/'smoke/summary.json'),'--allow-research-training']
                    with patch.object(sys,'argv',argv), patch('pathlib.Path.cwd',return_value=runner.REPO), \
                            patch.object(runner,'source_contract',return_value=(source,{},self.bound,[])), \
                            patch.object(runner,'load_runtime',side_effect=lambda *args:RepairToy()), \
                            patch.object(runner,'check_transitions',return_value=[]), \
                            patch.object(RepairToy,'gradients',gradients), \
                            patch('small_room30_targeted_v4_runtime.measured_canary',side_effect=measured), \
                            patch('small_room30_targeted_v4_runtime.baseline',return_value=previous), \
                            patch('small_room30_targeted_v4_runtime.acceptance',return_value=dict(accepted=accept)), \
                            contextlib.redirect_stdout(io.StringIO()): runner.main()
                result=json.loads((root/'train/summary.json').read_text())
                self.assertEqual(result['accepted_updates'], int(accept))
                self.assertEqual(cp.read_bytes(), b'original1560')
                if accept:
                    saved=load_checkpoint(result['last_checkpoint']['path'])
                    self.assertEqual(state_digest(saved['adapter']), result['last_checkpoint']['adapter_digest'])
                    self.assertEqual(saved['purpose'], 'RESEARCH_TARGETED_V4_PILOT')
                else: self.assertIsNone(result['last_checkpoint'])


if __name__ == '__main__': unittest.main()
