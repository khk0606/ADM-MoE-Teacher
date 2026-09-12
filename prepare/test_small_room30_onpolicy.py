"""CPU safety/contract tests; real CUDA readiness is a separate mandatory stage."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
torch.set_num_threads(1)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diffusion.gaussian_diffusion import GaussianDiffusion, ModelMeanType, ModelVarType, LossType, get_named_beta_schedule
from diffusion.respace import SpacedDiffusion
from small_room30_onpolicy_runtime import free_state, clean_prediction, OnPolicyRuntime
from small_room30_onpolicy import timestep_plan, check_ready, source_contract, POLICY
from small_room30_adm_dataset import SmallRoom30ADM, sha
from small_room30_training_contract import schedule, write_json, SCHEMA
from fewshot_cdm_lora import install_lora, lora_named_parameters
from small_room30_training_runtime import adapter_state, frozen_digest, state_digest
from relational_teacher_v9_lora_runtime import deterministic_noise
from compare_small_room30_onpolicy import compare_reports


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(6,6)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(6)*.1); self.linear.bias.fill_(.05)
        self.record = False
        self.inputs = {}

    def forward(self, x, t, **kwargs):
        if set(kwargs) != {'c_pc_xyz','c_pc_feat','c_text'}: raise ValueError('Unexpected inputs')
        self.last_t = t.detach().clone()
        if self.record and int(t[0]) in (499,498,490,450,0):
            self.inputs[int(t[0])] = x.detach().clone()
        return self.linear(x)


def diffusion(spaced=False):
    kw = dict(betas=get_named_beta_schedule('cosine',1000 if spaced else 500),
        model_mean_type=ModelMeanType.START_X, model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE, rescale_timesteps=False)
    return SpacedDiffusion(use_timesteps=set(range(0,1000,2)), **kw) if spaced else GaussianDiffusion(**kw)


class ToyRuntime(OnPolicyRuntime):
    data_source = None
    def __init__(self, *args):
        self.device='cpu'; self.seed=72; self.data=self.data_source
        self.model=ToyModel(); self.diffusion=diffusion()
        self.mean=np.zeros((1,6),np.float32); self.std=np.ones((1,6),np.float32)
        self.mean_t=torch.from_numpy(self.mean[None]); self.std_t=torch.from_numpy(self.std[None])
        self.base_digest=frozen_digest(self.model); self.encoder_weight={}
        self.install()

    def install(self):
        if not any('lora_' in n for n,_ in self.model.named_parameters()): install_lora(self.model,2,2.)
        self.frozen_after_install=frozen_digest(self.model)

    def objective(self, pred, target, index):
        # Lightweight differentiable toy objective; NOT ADM performance evidence.
        loss=(pred-target).square().mean()
        return loss, dict(normalized_mse=loss, surface_mean=loss, surface_worst=loss)


class OnPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=SmallRoom30ADM(os.environ.get('SMALL_ROOM30_DATASET','data/small_room30_adm_lora_v1'))
        ToyRuntime.data_source=cls.data
        cls.kw=dict(c_pc_xyz=torch.zeros(1,8192,3),c_pc_feat=torch.zeros(1,8192,3),c_text=['sit'])

    def test_state_is_input_before_t_not_output_after_t(self):
        rt=ToyRuntime(); rt.model.record=True
        with torch.no_grad(),torch.random.fork_rng():
            torch.manual_seed(34)
            noise=deterministic_noise((1,8192,6),12,'cpu')
            for _ in rt.diffusion.p_sample_loop_progressive(rt.model,noise.shape,noise=noise,
                    clip_denoised=False,model_kwargs=self.kw,device='cpu'): pass
        expected=rt.model.inputs.copy(); rt.model.record=False
        for t in (499,498,490,450,0):
            state=free_state(rt,self.kw,t,12,34)
            self.assertTrue(torch.equal(state,expected[t]),t)
            self.assertFalse(state.requires_grad)
            self.assertIsNone(state.grad_fn)
        self.assertFalse(torch.equal(expected[499],expected[498]))

    def test_rng_is_restored(self):
        rt=ToyRuntime(); before=torch.random.get_rng_state().clone()
        free_state(rt,self.kw,490,12,34)
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))

    def test_timestep_mapping_applied_once(self):
        rt=ToyRuntime(); rt.diffusion=diffusion(spaced=True)
        state=free_state(rt,self.kw,490,12,34)
        clean_prediction(rt,state,self.kw,490)
        self.assertEqual(int(rt.model.last_t[0]),980)

    def test_clean_prediction_matches_sampler_and_has_gradient(self):
        rt=ToyRuntime(); state=free_state(rt,self.kw,490,12,34)
        pred=clean_prediction(rt,state,self.kw,490)
        with torch.no_grad(): expected=rt.diffusion.p_sample(rt.model,state,torch.tensor([490]),
            clip_denoised=False,model_kwargs=self.kw)['pred_xstart']
        self.assertTrue(torch.equal(pred,expected))
        pred.square().mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in lora_named_parameters(rt.model).values()))
        self.assertTrue(all(p.grad is None for n,p in rt.model.named_parameters() if 'lora_' not in n))

    def test_gt_or_labels_rejected(self):
        for key in ('gt','instance_ids','target'):
            with self.assertRaises(ValueError): free_state(ToyRuntime(),dict(self.kw,**{key:0}),490,12,34)

    def test_invalid_timestep_and_incomplete_prefix_rejected(self):
        rt=ToyRuntime()
        for t in (-1,500,1.5):
            with self.assertRaises(ValueError): free_state(rt,self.kw,t,1,2)
        with patch.object(rt.diffusion,'p_sample_loop_progressive',return_value=iter([])):
            with self.assertRaises(ValueError): free_state(rt,self.kw,490,1,2)

    def test_current_weights_refresh_state(self):
        rt=ToyRuntime(); a=free_state(rt,self.kw,490,12,34)
        with torch.no_grad():
            for p in lora_named_parameters(rt.model).values(): p.add_(.1)
        b=free_state(rt,self.kw,490,12,34)
        self.assertFalse(torch.equal(a,b))

    def test_timestep_schedule_balanced_and_bounded(self):
        ts=timestep_plan(60,72)
        self.assertEqual(ts,timestep_plan(60,72)); self.assertNotEqual(ts,timestep_plan(60,73))
        self.assertEqual(sum(t>=490 for t in ts),48)
        self.assertEqual(set(ts),set(range(490,500))|{450,300,100,0})
        for count in (0,61):
            with self.assertRaises(ValueError): timestep_plan(count,72)

    def test_all_rooms_and_actions_replayed(self):
        plan=schedule(self.data,60,72+300001)
        counts={}
        for ids in plan:
            self.assertEqual([self.data.samples[i]['action'] for i in ids],['sit','lie','write_board'])
            room=self.data.samples[ids[0]]['scene_id'];counts[room]=counts.get(room,0)+1
        self.assertEqual(len(counts),30); self.assertEqual(set(counts.values()),{2})

    def test_update_changes_only_lora_and_logs_real_t(self):
        rt=ToyRuntime(); before=frozen_digest(rt.model); adapter=state_digest(adapter_state(rt.model))
        opt=torch.optim.AdamW(list(lora_named_parameters(rt.model).values()),lr=POLICY['lr'])
        row=rt.update_onpolicy(opt,schedule(self.data,1,72)[0],1501,490)
        self.assertEqual(row['prefix_steps'],9);self.assertEqual(len(row['replay_timesteps']),3)
        self.assertEqual(before,frozen_digest(rt.model)); self.assertNotEqual(adapter,state_digest(adapter_state(rt.model)))
        self.assertGreater(row['gradient_norm'],0)

    def test_missing_replay_action_rejected(self):
        rt=ToyRuntime(); opt=torch.optim.AdamW(list(lora_named_parameters(rt.model).values()))
        with self.assertRaises(ValueError):rt.update_onpolicy(opt,[0,0,0],1501,490)

    def test_readiness_fail_closed(self):
        with self.assertRaises(ValueError):check_ready({}, {})
        with self.assertRaises(ValueError):check_ready(dict(status='ONPOLICY_RUNTIME_READY',binding={}),{})

    def test_comparison_warns_lie_write_regression(self):
        before={'rows':[dict(case=dict(action='lie',sample_id='bed'),seeds=[1,2],metrics={'adapted_raw':{'mae_raw':.1}})]}
        after=copy.deepcopy(before);after['rows'][0]['metrics']['adapted_raw']['mae_raw']=.2
        self.assertEqual(len(compare_reports(before,after)['lie_write_worsened']),1)
        after['rows'][0]['seeds']=[2,3]
        with self.assertRaises(ValueError):compare_reports(before,after)

    def test_source_contract_hash_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'cases').mkdir();sourcepath=root/'source.json'; cp=root/'adapter.pt';cp.write_bytes(b'toy')
            parent={'synthetic':True}; source=dict(completed_step=1500,last_checkpoint=dict(path=str(cp),sha256=sha(cp),step=1500))
            write_json(sourcepath,source)
            manifest=dict(checkpoint_sha256=sha(cp),binding=parent,training_summary_sha256=sha(sourcepath))
            write_json(root/'manifest.json',manifest);rows=[]
            for i in range(6):
                path=root/'cases'/('%d.npz'%i);path.write_bytes(b'toy')
                rows.append(dict(case=dict(scene_id=str(i),sample_id=str(i)+'__sit_anywhere',generation=0,filename=path.name),
                    sha256=sha(path),metrics=dict(replay_matches_saved=True,replay_max_abs_error=0.)))
            diag=dict(status='DIAGNOSTIC_COMPLETE',model_state_unchanged=True,training_started=False,rows=rows,
                manifest_sha256=sha(root/'manifest.json'))
            write_json(root/'summary.json',diag)
            repo=Path(__file__).resolve().parents[1]
            with patch('small_room30_onpolicy.validate_experiment',return_value=(source,{'binding':{}},parent)):
                source_contract(self.data,sourcepath,root/'summary.json',repo)
                (root/'cases/0.npz').write_bytes(b'tamper')
                with self.assertRaises(ValueError):source_contract(self.data,sourcepath,root/'summary.json',repo)

    def test_runner_smoke_then_train_checkpoint_roundtrip(self):
        from run_small_room30_onpolicy import main, REPO
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cp=root/'source.pt';cp.write_bytes(b'source');sourcepath=root/'source.json';diagpath=root/'diag.json'
            source=dict(seed=72,last_checkpoint=dict(path=str(cp),sha256=sha(cp),step=1500))
            write_json(sourcepath,source);write_json(diagpath,{'synthetic':True})
            bound={'synthetic':True}
            for mode in ('smoke','train'):
                out=root/mode
                args=['run','--mode',mode,'--dataset',str(self.data.root),'--source-summary',str(sourcepath),
                    '--diagnostic-summary',str(diagpath),'--output-dir',str(out),'--device','cpu']
                if mode=='train':args += ['--readiness',str(root/'smoke/summary.json'),'--allow-research-training','--updates','2']
                with patch.object(sys,'argv',args),patch('pathlib.Path.cwd',return_value=REPO), \
                     patch('run_small_room30_onpolicy.source_contract',return_value=(source,{},bound)), \
                     patch('run_small_room30_onpolicy.load_runtime',side_effect=lambda *args:ToyRuntime()), \
                     patch('run_small_room30_onpolicy.diagnostic_gradient_check',return_value=[]), \
                     contextlib.redirect_stdout(io.StringIO()): main()
            report=json.loads((root/'train/summary.json').read_text())
            self.assertEqual(report['completed_step'],1502)
            self.assertEqual(report['status'],'ONPOLICY_PILOT_FINISHED_NOT_EVALUATED')
            self.assertEqual(cp.read_bytes(),b'source')
            self.assertFalse(report['teacher_checkpoint_authorized'])

    def test_diagnostic_correction_gradient_check_is_executed(self):
        from run_small_room30_onpolicy import diagnostic_gradient_check
        from small_room30_evaluation_common import panel
        rt=ToyRuntime();rows=[]
        cases=[c for c in panel(self.data,'preview') if c['sample_id'].endswith('__sit_anywhere')]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'cases').mkdir()
            for i,case in enumerate(cases):
                with torch.no_grad():pred,_=rt.free_prediction(case['index'],490,12,34)
                np.savez_compressed(root/'cases'/case['filename'],free_clean_t490=pred[0].numpy())
                rows.append(dict(case=case,seeds=[12,34],metrics=dict(free_clean_estimates=[
                    dict(timestep=490,surface=dict(all_furniture_present=i>=4))])))
            before=state_digest(adapter_state(rt.model))
            with contextlib.redirect_stdout(io.StringIO()):result=diagnostic_gradient_check(rt,{'rows':rows},root/'summary.json')
            self.assertEqual(sum(r['corrective_gradient_norm'] is not None and r['corrective_gradient_norm']>0 for r in result),4)
            self.assertEqual(before,state_digest(adapter_state(rt.model)))
            # A changed reference is a hard failure, not silently accepted training.
            np.savez_compressed(root/'cases'/cases[0]['filename'],free_clean_t490=np.ones((8192,6),np.float32)*99)
            with self.assertRaises(ValueError):diagnostic_gradient_check(rt,{'rows':rows},root/'summary.json')

    def test_evaluator_and_unchanged_viewer_contract_end_to_end(self):
        from evaluate_small_room30_onpolicy import main, REPO
        from small_room30_evaluation_common import validate_report
        rt=ToyRuntime()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cp=root/'adapter.pt';train=root/'train.json';out=root/'eval'
            bound={'original_binding':{'inputs':{}},'source_checkpoint':{'sha256':'synthetic-source'}}
            torch.save(dict(schema=SCHEMA,purpose='RESEARCH_ONPOLICY_PILOT',binding=bound,seed=72,step=1502,
                teacher_checkpoint_authorized=False,adapter=adapter_state(rt.model),base_digest=rt.base_digest,encoder_weight={}),cp)
            source=dict(seed=72,last_checkpoint=dict(path=str(cp),sha256=sha(cp),step=1502),completed_step=1502)
            write_json(train,source)
            args=['eval','--training-summary',str(train),'--dataset',str(self.data.root),'--output-dir',str(out),'--device','cpu']
            with patch.object(sys,'argv',args),patch('pathlib.Path.cwd',return_value=REPO), \
                 patch('small_room30_onpolicy.validate_run',return_value=(source,{},bound)), \
                 patch('small_room30_training_runtime.Runtime',ToyRuntime), \
                 patch('torch.cuda.get_device_name',return_value='TOY_CPU'),contextlib.redirect_stdout(io.StringIO()):main()
            report,_=validate_report(out/'summary.json')
            self.assertEqual(report['total_cases'],16);self.assertFalse(report['teacher_checkpoint_authorized'])


if __name__=='__main__':unittest.main(verbosity=2)
