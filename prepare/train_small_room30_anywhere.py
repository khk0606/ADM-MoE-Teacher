#!/usr/bin/env python3
"""Add anywhere conditions to all rooms; fine-tune saved student; CPU evaluation."""
import argparse
import json
import math
from pathlib import Path
import random
import time
import numpy as np
from small_room30_adm_dataset import sha,inside
from small_room30_student_data import TEACHER_SHA
from small_room30_anywhere_data import AnywhereData as CompetitionData,RUN_POLICY as POLICY,ANYWHERE,TEXTS
from small_room30_anywhere_text import prepare_text
from train_small_room30_student_v2 import group_records,pair_indices,make_batch,write_json,review_tag
from train_small_room30_student_v2_compat import load_baseline_without_pickle,CACHE_FILE

CODE_FILES=('small_room30_anywhere_data.py','small_room30_anywhere_text.py','train_small_room30_anywhere.py','small_room30_anywhere_review.py','small_room30_competition_data.py','small_room30_competition_model.py',
    'train_small_room30_competition.py','small_room30_competition_review.py',
    'small_room30_weight_student_v2.py','small_room30_student_v2_data.py',
    'small_room30_weight_student.py','small_room30_student_data.py','small_room30_adm_dataset.py',
    'small_room30_evaluation_common.py','relation_aware_moe_v2_contract.py',
    'train_small_room30_student_v2.py','train_small_room30_student_v2_compat.py',CACHE_FILE)
GATES=dict(slot_max_error_m=.5,clear_target_gap=.2,selection_probability_slack=.15,
           final_margin_slack=.15,
           floor_ratio_min=.7,floor_ratio_max=1.3,
           warning='Development diagnostic gates; passing never substitutes for user visual approval')


def comparison_source(path,binding):
    report=json.loads(path.read_text());root=path.resolve().parent
    if report.get('status')!='COMPETITION_COMPLETE_REVIEW_REQUIRED':
        raise ValueError('Use the completed competition CPU evaluation summary, not failed GPU run or floorfix')
    if sha(root/'manifest.json')!=report['manifest_sha256']:raise ValueError('Initial manifest changed')
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest['binding']!=binding:raise ValueError('Initial student data/Teacher binding mismatch')
    for name,key in (('student_model.json','model_sha256'),('student_weights.npz','student_sha256')):
        if sha(root/name)!=report[key]:raise ValueError('Initial checkpoint changed: '+name)
    expected=[r for r in binding['records'] if r['split']!='train']
    if [r['record'] for r in report['rows']]!=expected:raise ValueError('Initial evaluation coverage mismatch')
    for row in report['rows']:
        if sha(inside(root,row['file']))!=row['sha256']:raise ValueError('Initial map changed')
    return report


def target_audit(data):
    rows=[]
    for r in data.records:
        if r['generation']!=0:continue
        _,l=data.example(r)
        rows.append(dict(record=r,candidate_ids=data.geometry(r['scene_id'])[3],
            distances=l['distance_matrix'].tolist(),active_contexts=l['purpose_classes'].tolist(),
            relation=l['candidate_relation'].tolist(),history=l['history_scores'].tolist(),
            score=l['score_target'].tolist(),selection=l['selection_target'].tolist(),
            floor_points=int(l['floor_support_mask'].sum())))
    return dict(policy=POLICY,rows=rows,warning='Geometric proxy targets, not human preference observations or trained predictions')


def verify_comparison_arrays(data,path,report):
    """Fail BEFORE training if comparison is from another Teacher or ordering."""
    for row in report['rows']:
        inputs,_=data.example(row['record'])
        with np.load(inside(path.resolve().parent,row['file']),allow_pickle=False) as z:
            for key in ('points','teacher_a'):
                if not np.array_equal(z[key],inputs[key]):
                    raise ValueError('Preflight comparison mismatch: '+key)
            if z['a_w'].shape!=inputs['teacher_a'].shape or not np.isfinite(z['a_w']).all():
                raise ValueError('Invalid comparison final map')


from train_small_room30_competition import save_model,load_model


def train(data,text,output,device,warmup,joint,lr,seed,width,manifest,initial_model):
    import torch
    from small_room30_competition_model import CompetitionStudent,competition_losses
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.device(device).type=='cuda':torch.cuda.manual_seed_all(seed)
    model=load_model(initial_model,device)
    if model.config['width']!=width:raise ValueError('Initial model width mismatch')
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    groups=group_records([r for r in data.records if r['split']=='train'])
    if not groups or any(r['generation'] not in (0,1) for g in groups for r in g):raise ValueError('Only g0/g1 in optimizer')
    rng=random.Random(seed);order=list(groups);logs=[];start=time.monotonic()
    model.train()
    for step in range(1,warmup+joint+1):
        if (step-1)%len(order)==0:rng.shuffle(order)
        group=order[(step-1)%len(order)];inputs,labels=make_batch(data,group,text,device)
        optimizer.zero_grad(set_to_none=True)
        losses=competition_losses(model(**inputs),inputs['teacher_a'],labels,*pair_indices(group))
        objective=losses['auxiliary'] if step<=warmup else losses['total']
        if not torch.isfinite(objective):raise FloatingPointError('Nonfinite loss')
        objective.backward()
        for name,p in model.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():raise FloatingPointError('Nonfinite gradient: '+name)
            if step>warmup and p.grad is None:raise FloatingPointError('Disconnected parameter: '+name)
        torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
        optimizer.step()
        if step==1 or step%25==0 or step in (warmup,warmup+joint):
            log=dict(step=step,phase='perception_and_support_warmup' if step<=warmup else 'joint_competition',
                seconds=time.monotonic()-start,scene=group[0]['scene_id'],generation=group[0]['generation'],
                **{k:float(v.detach().cpu()) for k,v in losses.items()})
            logs.append(log);write_json(output/'training_log.json',logs)
            print('[COMPETITION TRAIN] '+json.dumps(log),flush=True)
    save_model(model,output,manifest)
    return model,logs


def diagnostic(out,labels,teacher):
    import torch
    from small_room30_competition_model import align
    from small_room30_weight_student_v2 import final_candidate_scores
    a=align(out,labels);target_aw=teacher*labels['target_weight'][...,None]
    target=labels['selection_target'][0].cpu().numpy();selection=a['selection'][0].cpu().numpy()
    pred_scores=final_candidate_scores(out['a_w'],teacher,labels['candidate_masks'])[0].cpu().numpy()
    target_scores=final_candidate_scores(target_aw,teacher,labels['candidate_masks'])[0].cpu().numpy()
    slot_error=(a['slot_xy']-labels['candidate_anchors']).norm(dim=-1)[0].cpu().numpy()
    support=labels['floor_support_weight']
    def floor_mean(x):return float((x.clamp_min(0).max(-1).values*support).sum()/support.sum().clamp_min(1e-8))
    floor_target=floor_mean(target_aw);floor_pred=floor_mean(out['a_w'])
    floor_ratio=floor_pred/floor_target if floor_target>1e-8 else None
    clear=abs(float(target[0]-target[1]))>=GATES['clear_target_gap']
    failures=[]
    if slot_error.max()>GATES['slot_max_error_m']:failures.append('candidate_localization')
    if clear and selection.argmax()!=target.argmax():failures.append('wrong_confident_candidate')
    if clear and selection[int(target.argmax())]<float(target.max())-GATES['selection_probability_slack']:
        failures.append('weak_candidate_contrast')
    if clear and pred_scores.argmax()!=target_scores.argmax():failures.append('wrong_final_map_winner')
    winner=int(target.argmax());other=1-winner
    def margin(s):return float((s[winner]-s[other])/max(float(np.max(np.abs(s))),1e-8))
    if clear and margin(pred_scores)<margin(target_scores)-GATES['final_margin_slack']:
        failures.append('weak_final_map_contrast')
    if clear and target_scores.argmax()!=winner:failures.append('teacher_strength_limits_target_selection')
    if floor_ratio is not None and not GATES['floor_ratio_min']<=floor_ratio<=GATES['floor_ratio_max']:
        failures.append('floor_support_mismatch')
    return dict(relation=a['relation_score'][0].cpu().tolist(),history=a['history_score'][0].cpu().tolist(),
        score=a['score'][0].cpu().tolist(),selection=selection.tolist(),selection_target=target.tolist(),
        relation_target=labels['candidate_relation'][0].cpu().tolist(),history_target=labels['history_scores'][0].cpu().tolist(),
        final_scores=pred_scores.tolist(),target_final_scores=target_scores.tolist(),
        slot_error_m=slot_error.tolist(),floor_target_mean=floor_target,floor_predicted_mean=floor_pred,
        floor_predicted_to_target_ratio=floor_ratio,failures=failures,diagnostic_gate_pass=not failures,
        approval='NOT_APPROVED_REQUIRES_VISUAL_REVIEW')


def evaluate(model,data,text,output,comparison_path,comparison,device):
    import torch
    from small_room30_competition_model import competition_losses
    model.eval();maps=output/'maps';maps.mkdir()
    records=[r for r in data.records if r['split']!='train']
    lookup={json.dumps(r['record'],sort_keys=True):r for r in comparison['rows']}
    rows=[];details=[];slot_outputs={}
    with torch.no_grad():
        for i,r in enumerate(records):
            inputs,labels=make_batch(data,[r],text,device);out=model(**inputs)
            losses=competition_losses(out,inputs['teacher_a'],labels)
            arrays={k:out[k][0].cpu().numpy() for k in ('a_w','w','selection','score','relation_score','history_score',
                'slot_xy','support','relation_map','history_map','score_map','context_xy','pi_r','pi_h')}
            arrays.update(points=inputs['points'][0].cpu().numpy(),teacher_a=inputs['teacher_a'][0].cpu().numpy(),
                observed_history=inputs['observed_history'][0].cpu().numpy(),
                training_target_aw=(inputs['teacher_a']*labels['target_weight'][...,None])[0].cpu().numpy())
            if r['prompt_id']==ANYWHERE:
                # No old Student result exists for this new prompt.
                arrays['baseline_aw']=arrays['teacher_a'].copy()
            else:
                old=lookup[json.dumps(r,sort_keys=True)];path=inside(comparison_path.parent,old['file'])
                if sha(path)!=old['sha256']:raise ValueError('Comparison changed during training')
                with np.load(path,allow_pickle=False) as z:
                    if not np.array_equal(z['points'],arrays['points']) or not np.array_equal(z['teacher_a'],arrays['teacher_a']):
                        raise ValueError('Comparison Teacher/point order mismatch')
                    arrays['baseline_aw']=z['a_w'].copy()
            arrays['baseline_is_teacher']=np.asarray(r['prompt_id']==ANYWHERE)
            name='%03d.npz'%i;np.savez_compressed(maps/name,**arrays)
            rows.append(dict(record=r,file='maps/'+name,sha256=sha(maps/name),losses={k:float(v.cpu()) for k,v in losses.items()}))
            detail=dict(record=r,candidate_ids=data.geometry(r['scene_id'])[3],review_group=review_tag(r),
                        **diagnostic(out,labels,inputs['teacher_a']))
            if r['prompt_id']!=ANYWHERE:
                from small_room30_weight_student_v2 import final_candidate_scores
                previous=final_candidate_scores(torch.from_numpy(arrays['baseline_aw'][None]).to(device),
                    inputs['teacher_a'],labels['candidate_masks'])[0].cpu().numpy()
                detail['previous_final_scores']=previous.tolist()
                detail['previous_correct_winner_lost']=bool(previous.argmax()==np.argmax(detail['target_final_scores']) and
                    np.argmax(detail['final_scores'])!=np.argmax(detail['target_final_scores']))
            else:
                detail['previous_final_scores']=None;detail['previous_correct_winner_lost']=None
            details.append(detail)
            slot_outputs[json.dumps(r,sort_keys=True)]={k:arrays[k] for k in ('relation_score','history_score','slot_xy')}
            if (i+1)%10==0 or i+1==len(records):print('[COMPETITION EVAL] %d/%d'%(i+1,len(records)),flush=True)
    paired=[]
    positions={json.dumps(r,sort_keys=True):i for i,r in enumerate(records)}
    for group in group_records(records):
        for kind,pairs in zip(('purpose','history'),pair_indices(group)):
            for i,j in pairs:
                a,b=(details[positions[json.dumps(group[k],sort_keys=True)]] for k in (i,j))
                raw_a=slot_outputs[json.dumps(group[i],sort_keys=True)];raw_b=slot_outputs[json.dumps(group[j],sort_keys=True)]
                # Structural invariant: localization never depends on text/history;
                # Relation never sees history; History never sees text.
                if not np.allclose(raw_a['slot_xy'],raw_b['slot_xy'],atol=1e-5):raise ValueError('Candidate localization condition leakage')
                invariant='relation_score' if kind=='history' else 'history_score'
                if not np.allclose(raw_a[invariant],raw_b[invariant],atol=1e-5):raise ValueError('Branch condition leakage')
                target_flip=int(np.argmax(a['selection_target']))!=int(np.argmax(b['selection_target']))
                predicted_flip=int(np.argmax(a['final_scores']))!=int(np.argmax(b['final_scores']))
                paired.append(dict(kind=kind,left=a['record'],right=b['record'],target_flip=target_flip,
                    predicted_final_map_flip=predicted_flip,missed_expected_flip=bool(target_flip and not predicted_flip)))
    report=dict(status='DIAGNOSTICS_ONLY_NOT_APPROVAL',gates=GATES,rows=details,pairs=paired,
        previous_correct_winners_lost=sum(r['previous_correct_winner_lost'] is True for r in details),
        failed_cases=sum(not r['diagnostic_gate_pass'] for r in details),missed_expected_flips=sum(r['missed_expected_flip'] for r in paired),
        warning='Softmax confidence is not correctness. Check the chosen object, final map and floor together.')
    write_json(output/'comparison.json',report)
    print('[NOT AUTO-APPROVED] %d/%d diagnostic failures; %d missed expected switches.'%
        (report['failed_cases'],len(rows),report['missed_expected_flips']),flush=True)
    return rows,report


def main():
    import torch
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=Path('data/small_room30_adm_lora_v1'))
    p.add_argument('--teacher-summary',type=Path,default=Path('outputs/small_room30_targeted_v5_eval_full01/summary.json'))
    p.add_argument('--baseline-summary',type=Path,default=Path('outputs/small_room30_student_pilot01/summary.json'))
    p.add_argument('--initial-summary',type=Path,default=Path('outputs/small_room30_student_competition_cpu_eval01/summary.json'))
    p.add_argument('--clip-weights',type=Path,default=Path.home()/'.cache/clip/ViT-B-32.pt')
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--device',default='cuda:0');p.add_argument('--joint-steps',type=int,default=2000)
    p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--seed',type=int,default=20260913)
    args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError('Choose a NEW directory; no overwrite')
    if args.joint_steps<1 or not math.isfinite(args.lr) or args.lr<=0:raise ValueError('Invalid training settings')
    torch.set_num_threads(1)
    print('[PREFLIGHT ANYWHERE] Existing1578 + initial competition weights + real added CLIP text',flush=True)
    data=CompetitionData(args.dataset,args.teacher_summary)
    comparison=comparison_source(args.initial_summary,data.legacy_binding)
    verify_comparison_arrays(data,args.initial_summary,comparison)
    text,text_provenance=prepare_text(args.baseline_summary,data.legacy_binding,args.clip_weights)
    initial_model=args.initial_summary.resolve().parent/'student_model.json'
    initial=load_model(initial_model,'cpu');width=initial.config['width'];del initial
    audit=target_audit(data)
    source=Path(__file__).parent
    manifest=dict(schema='small_room30_anywhere_run_v1',binding=data.binding,policy=POLICY,gates=GATES,
        code_sha256={n:sha(source/n) for n in CODE_FILES},prompt_texts=TEXTS,
        initial_summary_sha256=sha(args.initial_summary),initial_model_sha256=sha(initial_model),
        initial_weights_sha256=sha(initial_model.parent/'student_weights.npz'),text_provenance=text_provenance,
        warmup_steps=0,joint_steps=args.joint_steps,seed=args.seed,lr=args.lr,width=width,
        initialization='Fine-tune existing numeric competition checkpoint with all old + anywhere conditions',
        evaluation_device='cpu',evaluation_threads=1,training_examples=300,evaluation_examples=150,
        scope='All 30 rooms are development. g0/g1 training, g2 same-scene noise repeats; no unseen-room claim')
    args.output_dir.mkdir(parents=True)
    np.savez_compressed(args.output_dir/'text_features.npz',**{k:v.numpy() for k,v in text.items()})
    manifest['text_features_sha256']=sha(args.output_dir/'text_features.npz')
    write_json(args.output_dir/'manifest.json',manifest);write_json(args.output_dir/'target_audit.json',audit)
    if args.preflight_only:
        write_json(args.output_dir/'summary.json',dict(status='ANYWHERE_PREFLIGHT_PASS_NO_TRAINING',manifest_sha256=sha(args.output_dir/'manifest.json')))
        print('[PASS] 300 training / 150 evaluation conditions, five actual text vectors; optimizer steps=0',flush=True);return
    try:
        model,logs=train(data,text,args.output_dir,args.device,0,args.joint_steps,args.lr,args.seed,width,manifest,initial_model)
        del model
        model=load_model(args.output_dir/'student_model.json','cpu')
        print('[CPU EVAL] Single-thread, unchanged strict invariance checks; no CUDA evaluation',flush=True)
        rows,review=evaluate(model,data,text,args.output_dir,args.initial_summary,comparison,'cpu')
        if sha(initial_model)!=manifest['initial_model_sha256'] or sha(initial_model.parent/'student_weights.npz')!=manifest['initial_weights_sha256']:
            raise ValueError('Initial checkpoint changed during run')
        write_json(args.output_dir/'summary.json',dict(status='ANYWHERE_COMPLETE_REVIEW_REQUIRED',
            teacher_checkpoint_sha256=TEACHER_SHA,teacher_unchanged=True,teacher_training_performed=False,
            motion_generation_performed=False,manifest_sha256=sha(args.output_dir/'manifest.json'),
            model_sha256=sha(args.output_dir/'student_model.json'),student_sha256=sha(args.output_dir/'student_weights.npz'),
            comparison_sha256=sha(args.output_dir/'comparison.json'),rows=rows,log=logs,
            diagnostic_failed_cases=review['failed_cases'],missed_expected_flips=review['missed_expected_flips'],
            approved=False,policy=POLICY,evaluation_device='cpu'))
        print('[DONE — REVIEW REQUIRED] '+str(args.output_dir/'summary.json'),flush=True)
    except Exception as exc:
        write_json(args.output_dir/'failure.json',dict(status='FAILED',error=repr(exc)));raise


if __name__=='__main__':main()
