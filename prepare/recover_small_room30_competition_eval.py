#!/usr/bin/env python3
"""Recover completed competition training by CPU evaluation only; never optimize.

Do not relax invariance tolerances, edit weights, or modify the failed run.
Old GPU pair differences are evidence, not proof of a particular failure cause.
"""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from small_room30_adm_dataset import sha,inside
from small_room30_competition_data import CompetitionData,POLICY
from small_room30_student_data import TEACHER_SHA
from train_small_room30_competition import (load_model,evaluate,comparison_source,
    verify_comparison_arrays,write_json,make_batch,group_records,pair_indices)
from train_small_room30_student_v2_compat import load_baseline_without_pickle


def verify_completed_training(root):
    """Require the exact immutable release and completed optimizer log."""
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest.get('schema')!='small_room30_competition_run_v1' or manifest.get('policy')!=POLICY:
        raise ValueError('Not a compatible competition run')
    if (root/'summary.json').exists():raise ValueError('Source already has a completed summary; do not overwrite/recover it')
    failure=json.loads((root/'failure.json').read_text())
    if failure.get('error') not in ("ValueError('Candidate localization condition leakage')","ValueError('Branch condition leakage')"):
        raise ValueError('This recovery only handles the post-training invariance failure')
    if not manifest.get('code_sha256'):raise ValueError('Missing training code provenance')
    for name,digest in manifest['code_sha256'].items():
        if sha(inside(Path(__file__).parent,name))!=digest:
            raise ValueError('Training source changed: '+name)
    logs=json.loads((root/'training_log.json').read_text())
    expected=manifest['warmup_steps']+manifest['joint_steps']
    if not logs or logs[-1]['step']!=expected or logs[-1]['phase']!='joint_competition':
        raise ValueError('Optimizer run did not finish; refusing to call this evaluation-only recovery')
    # load_model validates numeric weights, config and original manifest hashes.
    model=load_model(root/'student_model.json','cpu')
    return manifest,logs,model


def audit_saved_pairs(root,records):
    """Report direct and swapped candidate-coordinate differences without excuses."""
    cached={};files=[]
    for i,r in enumerate(records):
        path=root/'maps'/('%03d.npz'%i)
        with np.load(path,allow_pickle=False) as z:
            arrays={k:z[k].copy() for k in ('points','teacher_a','observed_history','slot_xy','relation_score','history_score')}
        if not all(np.isfinite(v).all() for v in arrays.values()):raise ValueError('Nonfinite saved GPU map')
        if arrays['slot_xy'].shape!=(2,2):raise ValueError('Wrong saved candidate coordinates')
        cached[json.dumps(r,sort_keys=True)]=arrays
        files.append(dict(file='maps/'+path.name,sha256=sha(path)))
    rows=[]
    for group in group_records(records):
        for kind,pairs in zip(('purpose','history'),pair_indices(group)):
            for i,j in pairs:
                left,right=group[i],group[j]
                a,b=(cached[json.dumps(r,sort_keys=True)] for r in (left,right))
                branch='relation_score' if kind=='history' else 'history_score'
                rows.append(dict(kind=kind,left=left,right=right,
                    identical_points=bool(np.array_equal(a['points'],b['points'])),
                    identical_teacher=bool(np.array_equal(a['teacher_a'],b['teacher_a'])),
                    identical_history=bool(np.array_equal(a['observed_history'],b['observed_history'])),
                    direct_slot_max_error_m=float(np.linalg.norm(a['slot_xy']-b['slot_xy'],axis=-1).max()),
                    swapped_slot_max_error_m=float(np.linalg.norm(a['slot_xy']-b['slot_xy'][::-1],axis=-1).max()),
                    original_slot_check_pass=bool(np.allclose(a['slot_xy'],b['slot_xy'],atol=1e-5)),
                    invariant_branch=branch,invariant_branch_max_abs_error=float(np.abs(a[branch]-b[branch]).max()),
                    original_branch_check_pass=bool(np.allclose(a[branch],b[branch],atol=1e-5))))
    return dict(rows=rows,files=files,
        failed_slot_pairs=sum(not r['original_slot_check_pass'] for r in rows),
        failed_branch_pairs=sum(not r['original_branch_check_pass'] for r in rows),
        note='Original saved outputs had no completed hash manifest; hashes here record current files, not retroactive provenance. Differences do not prove condition leakage or CUDA nondeterminism.')


def verify_repeat(model,data,text,records):
    """Single-thread CPU repeated identical-input check, before final evaluation."""
    import torch
    with torch.no_grad():
        for scene in sorted({r['scene_id'] for r in records}):
            record=next(r for r in records if r['scene_id']==scene)
            inputs,_=make_batch(data,[record],text,'cpu')
            a=model(**inputs);b=model(**inputs)
            for key in ('slot_xy','support','relation_score','history_score','a_w'):
                if not torch.equal(a[key],b[key]):raise ValueError('CPU repeated-input instability: '+key)


def main():
    import torch
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path,default=Path('outputs/small_room30_student_competition01'))
    p.add_argument('--output-dir',type=Path,default=Path('outputs/small_room30_student_competition_cpu_eval01'))
    p.add_argument('--dataset',type=Path,default=Path('data/small_room30_adm_lora_v1'))
    p.add_argument('--teacher-summary',type=Path,default=Path('outputs/small_room30_targeted_v5_eval_full01/summary.json'))
    p.add_argument('--baseline-summary',type=Path,default=Path('outputs/small_room30_student_pilot01/summary.json'))
    p.add_argument('--comparison-summary',type=Path,default=Path('outputs/small_room30_student_floorfix01/summary.json'))
    args=p.parse_args();root=args.source_dir.resolve();output=args.output_dir.resolve()
    if output.exists():raise FileExistsError('Use a NEW recovery output directory; source is never overwritten')
    torch.set_num_threads(1)
    # CPU only, no optimizer construction, no training API call, no CUDA model execution.
    print('[RECOVER] Existing trained weights; CPU single-thread evaluation only; optimizer steps=0',flush=True)
    manifest,logs,model=verify_completed_training(root)
    data=CompetitionData(args.dataset,args.teacher_summary)
    if data.binding!=manifest['binding']:raise ValueError('Recovered data/Teacher binding differs from training')
    _,text,clip_sha=load_baseline_without_pickle(args.baseline_summary,data.binding)
    if clip_sha!=manifest['clip_weights_sha256']:raise ValueError('Frozen text source changed')
    if sha(args.comparison_summary)!=manifest['comparison_summary_sha256']:raise ValueError('Previous comparison run changed')
    comparison=comparison_source(args.comparison_summary,data.binding)
    verify_comparison_arrays(data,args.comparison_summary,comparison)
    records=[r for r in data.records if r['split']!='train']
    output.mkdir(parents=True)
    try:
        old=audit_saved_pairs(root,records);write_json(output/'original_gpu_pair_audit.json',old)
        print('[AUDIT] Original saved pair failures: slots=%d, branches=%d; see original_gpu_pair_audit.json'%
            (old['failed_slot_pairs'],old['failed_branch_pairs']),flush=True)
        originals={n:sha(root/n) for n in ('manifest.json','student_model.json','student_weights.npz','training_log.json')}
        for name in originals:shutil.copyfile(root/name,output/name)
        if (root/'target_audit.json').exists():shutil.copyfile(root/'target_audit.json',output/'target_audit.json')
        verify_repeat(model,data,text,records)
        print('[PASS] CPU repeated identical inputs stable. Evaluating all 90 conditions with original checks.',flush=True)
        rows,review=evaluate(model,data,text,output,args.comparison_summary,comparison,'cpu')
        for name,digest in originals.items():
            if sha(root/name)!=digest or sha(output/name)!=digest:raise ValueError('Original model/provenance changed')
        recovery=dict(schema='competition_cpu_eval_recovery_v1',source_dir=str(root),device='cpu',threads=1,
            optimizer_steps=0,weights_unchanged=True,original_files=originals,
            old_pair_audit_sha256=sha(output/'original_gpu_pair_audit.json'),
            recovery_code_sha256=sha(Path(__file__)),checks_relaxed=False,model_architecture_changed=False,
            note='CPU rerun completes evaluation only. Original GPU cause remains subject to saved-pair audit; no automatic quality approval.')
        write_json(output/'recovery.json',recovery)
        write_json(output/'summary.json',dict(status='COMPETITION_COMPLETE_REVIEW_REQUIRED',
            teacher_checkpoint_sha256=TEACHER_SHA,teacher_unchanged=True,teacher_training_performed=False,
            motion_generation_performed=False,manifest_sha256=sha(output/'manifest.json'),
            model_sha256=sha(output/'student_model.json'),student_sha256=sha(output/'student_weights.npz'),
            comparison_sha256=sha(output/'comparison.json'),rows=rows,log=logs,
            diagnostic_failed_cases=review['failed_cases'],missed_expected_flips=review['missed_expected_flips'],
            approved=False,policy=POLICY,evaluation_recovery=recovery,recovery_sha256=sha(output/'recovery.json')))
        print('[DONE — REVIEW REQUIRED] '+str(output/'summary.json'),flush=True)
    except Exception as exc:
        write_json(output/'failure.json',dict(status='RECOVERY_FAILED_NO_TRAINING',error=repr(exc)));raise


if __name__=='__main__':main()
