#!/usr/bin/env python3
"""Additive corrected student run. Frozen1578; fresh student; no AMDM training."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import time
import numpy as np
from small_room30_adm_dataset import sha, inside
from small_room30_student_data import TEXTS, TEACHER_SHA
from small_room30_student_v2_data import StudentDataV2, TARGET_POLICY

BASELINE_SHA = '989e08abf861754a1fb8741db7e4c8042bbb63f9b87b3a00b792fcb0bb506d68'
CODE_FILES = ('train_small_room30_student_v2.py', 'small_room30_weight_student_v2.py',
    'small_room30_student_v2_data.py', 'small_room30_weight_student.py', 'small_room30_student_data.py',
    'small_room30_adm_dataset.py', 'small_room30_evaluation_common.py', 'relation_aware_moe_v2_contract.py')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def load_baseline(summary_path, binding):
    """Read only, hash-pinned weights_only load. Reuse frozen text, NOT old weights.

    No unsafe pickle fallback and no automatic CLIP download.
    """
    import torch
    root = summary_path.resolve().parent
    report = json.loads(summary_path.read_text())
    if report.get('status') != 'STUDENT_PILOT_COMPLETE_REVIEW_REQUIRED': raise ValueError('Incomplete baseline')
    manifest_path = root/'manifest.json'
    if sha(manifest_path) != report['manifest_sha256']: raise ValueError('Baseline manifest hash mismatch')
    manifest = json.loads(manifest_path.read_text())
    for field in ('teacher_checkpoint_sha256','teacher_summary_sha256','dataset_index_sha256','dataset_manifest_sha256','records'):
        if manifest['binding'][field] != binding[field]: raise ValueError('Baseline binding mismatch: '+field)
    if sha(root/'student.pt') != BASELINE_SHA or report['student_sha256'] != BASELINE_SHA:
        raise ValueError('Expected the audited pilot01 checkpoint; do not substitute another baseline')
    checkpoint = torch.load(root/'student.pt', map_location='cpu', weights_only=True)
    if (checkpoint['schema'] != 'small_room30_weight_student_v1' or checkpoint['prompt_texts'] != TEXTS
            or checkpoint['manifest'] != manifest or checkpoint['completed_steps'] != 600):
        raise ValueError('Baseline checkpoint contract mismatch')
    expected = [r for r in binding['records'] if r['split'] != 'train']
    if [r['record'] for r in report['rows']] != expected: raise ValueError('Baseline evaluation coverage mismatch')
    for row in report['rows']:
        if sha(inside(root,row['file'])) != row['sha256']: raise ValueError('Baseline map hash mismatch')
    text = checkpoint['text_features']
    if set(text) != set(TEXTS): raise ValueError('Missing frozen prompt features')
    for value in text.values():
        if value.shape != (1,512) or not torch.isfinite(value).all(): raise ValueError('Invalid frozen text features')
        if not torch.allclose(value.norm(dim=-1), torch.ones(1), atol=1e-4): raise ValueError('Unnormalized text')
    return report, {k:v.detach() for k,v in text.items()}, checkpoint['clip_weights_sha256']


def group_records(records):
    groups = defaultdict(list)
    for record in records:
        groups[(record['scene_id'],record['generation'])].append(record)
    return list(groups.values())


def pair_indices(records):
    purpose, history = [], []
    for i, a in enumerate(records):
        for j in range(i+1,len(records)):
            b = records[j]
            if (a['scene_id'],a['generation']) != (b['scene_id'],b['generation']): continue
            if a['history_motion_id'] == b['history_motion_id'] and a['prompt_id'] != b['prompt_id']:
                purpose.append((i,j))
            if a['prompt_id'] == b['prompt_id'] and a['history_motion_id'] != b['history_motion_id']:
                history.append((i,j))
    return purpose, history


def target_audit(data):
    """Report feasible proxy contrasts, not visual pass/fail or test-set tuning."""
    rows = []
    for group in group_records([r for r in data.records if r['generation'] == 0]):
        for record in group:
            _, labels = data.example(record)
            fused = .6*labels['candidate_relation']+.4*labels['history_scores']
            rows.append(dict(record=record, eligible=labels['eligible'].tolist(),
                relation_target=labels['candidate_relation'].tolist(), history_target=labels['history_scores'].tolist(),
                fused_candidate_target=fused.tolist(), proxy_weight_winner=int(fused.argmax())))
    return dict(rows=rows, target_policy=TARGET_POLICY,
        warning='Proxy preference is not eventual contact GT. Fused target ranking can differ from final A_w when Teacher strengths differ.',
        fixed_fusion_limit='No hard on/off: fixed-scene history changes any point weight by at most 0.4')


def make_batch(data, records, text_features, device):
    import torch
    examples = [data.example(r) for r in records]
    inputs = {k:torch.from_numpy(np.stack([e[0][k] for e in examples])).to(device) for k in examples[0][0]}
    inputs['text_features'] = torch.cat([text_features[r['prompt_id']] for r in records]).to(device)
    labels = {k:torch.from_numpy(np.stack([e[1][k] for e in examples])).to(device) for k in examples[0][1]}
    return inputs, labels


def run_training(data, text, output, manifest, device, warmup, joint, lr, seed, width):
    import torch
    from small_room30_weight_student_v2 import WeightStudentV2, student_losses_v2
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.device(device).type == 'cuda': torch.cuda.manual_seed_all(seed)
    model = WeightStudentV2(width=width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    groups = group_records([r for r in data.records if r['split']=='train'])
    if not groups or any(r['generation'] not in (0,1) for g in groups for r in g):
        raise ValueError('Only g0/g1 may enter optimizer')
    rng = random.Random(seed); order = list(groups); logs = []; start = time.monotonic()
    model.train()
    for step in range(1,warmup+joint+1):
        if (step-1) % len(order) == 0: rng.shuffle(order)
        group = order[(step-1) % len(order)]
        inputs, labels = make_batch(data,group,text,device)
        prompt_pairs, history_pairs = pair_indices(group)
        optimizer.zero_grad(set_to_none=True)
        out = model(**inputs)
        losses = student_losses_v2(out,inputs['teacher_a'],labels,prompt_pairs,history_pairs)
        objective = losses['auxiliary'] if step <= warmup else losses['total']
        if not torch.isfinite(objective): raise FloatingPointError('Nonfinite objective')
        objective.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError('Nonfinite gradient: '+name)
            if step > warmup and parameter.grad is None: raise FloatingPointError('Disconnected parameter: '+name)
        torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
        optimizer.step()
        if step == 1 or step % 25 == 0 or step in (warmup,warmup+joint):
            log = dict(step=step, phase='localization_warmup' if step<=warmup else 'joint_moe',
                scene=group[0]['scene_id'], generation=group[0]['generation'], batch=len(group),
                seconds=time.monotonic()-start, objective=float(objective.detach().cpu()),
                **{k:float(v.detach().cpu()) for k,v in losses.items()})
            logs.append(log); write_json(output/'training_log.json',logs)
            print('[STUDENT V2] '+json.dumps(log),flush=True)
    torch.save(dict(schema='small_room30_weight_student_v2', config=model.config,
        model={k:v.detach().cpu() for k,v in model.state_dict().items()}, text_features=text,
        prompt_texts=TEXTS, manifest=manifest, completed_steps=warmup+joint,
        clip_weights_sha256=manifest['clip_weights_sha256']), output/'student.pt')
    return model, logs


def review_tag(record):
    short = record['scene_id'].split('_')[-1]; prompt = record['prompt_id']
    if prompt=='sit_watch_v1' and short in ('0105','0203','0205'): return 'user_reported_failure'
    if prompt=='sit_write_v1' and short in ('0302','0304','0305','0401','0403','0404','0501','0502','0503','0504','0601','0602','0603','0605'):
        return 'user_reported_failure'
    return 'preservation_or_additional_check_not_new_user_approval'


def evaluate(model, data, text, output, baseline_path, baseline, device):
    import torch
    from small_room30_weight_student_v2 import candidate_mean, final_candidate_scores, student_losses_v2
    maps = output/'maps'; maps.mkdir()
    records = [r for r in data.records if r['split']!='train']
    lookup = {json.dumps(row['record'],sort_keys=True):row for row in baseline['rows']}
    rows = []; diagnostics = []; start = time.monotonic()
    model.eval()
    with torch.no_grad():
        for i, record in enumerate(records):
            inputs, labels = make_batch(data,[record],text,device); out = model(**inputs)
            losses = student_losses_v2(out,inputs['teacher_a'],labels)
            arrays = {k:out[k][0].cpu().numpy() for k in ('a_w','w','w_r','w_h','pi_r','pi_h','context_xy','purpose_probability')}
            arrays.update(points=inputs['points'][0].cpu().numpy(),teacher_a=inputs['teacher_a'][0].cpu().numpy(),
                observed_history=inputs['observed_history'][0].cpu().numpy())
            target = inputs['teacher_a']*(.6*labels['relation_spatial']+.4*labels['history_spatial'])[...,None]
            arrays['training_target_aw'] = target[0].cpu().numpy()
            base_row = lookup[json.dumps(record,sort_keys=True)]
            base_file = inside(baseline_path.parent,base_row['file'])
            if sha(base_file) != base_row['sha256']: raise ValueError('Baseline changed during training')
            with np.load(base_file,allow_pickle=False) as base:
                if not np.array_equal(base['teacher_a'],arrays['teacher_a']) or not np.array_equal(base['points'],arrays['points']):
                    raise ValueError('Baseline Teacher/point order mismatch')
                base_aw = torch.from_numpy(base['a_w'].copy())[None].to(device)
                baseline_scores = final_candidate_scores(base_aw,inputs['teacher_a'],labels['candidate_masks'])[0].cpu().tolist()
                drift = float(np.abs(base['a_w']-arrays['a_w']).mean())
                arrays['baseline_aw'] = base['a_w'].copy()
            name = '%03d.npz'%i; np.savez_compressed(maps/name,**arrays)
            rows.append(dict(record=record,file='maps/'+name,sha256=sha(maps/name),losses={k:float(v.cpu()) for k,v in losses.items()}))
            diagnostic = dict(record=record,review_group=review_tag(record),
                candidate_ids=data.geometry(record['scene_id'])[3], eligible=labels['eligible'][0].cpu().tolist(),
                w_r=candidate_mean(out['w_r'],labels['candidate_masks'])[0].cpu().tolist(),
                w_h=candidate_mean(out['w_h'],labels['candidate_masks'])[0].cpu().tolist(),
                relation_target=labels['candidate_relation'][0].cpu().tolist(),
                history_target=labels['history_scores'][0].cpu().tolist(),
                final_scores=final_candidate_scores(out['a_w'],inputs['teacher_a'],labels['candidate_masks'])[0].cpu().tolist(),
                target_final_scores=final_candidate_scores(target,inputs['teacher_a'],labels['candidate_masks'])[0].cpu().tolist(),
                baseline_final_scores=baseline_scores, baseline_aw_mean_abs_change=drift,
                purpose_classes=out['purpose_class_logits'][0].sigmoid().cpu().tolist(),
                context_xy=arrays['context_xy'].tolist())
            diagnostics.append(diagnostic)
            if (i+1)%10==0 or i+1==len(records): print('[EVAL V2] %d/%d'%(i+1,len(records)),flush=True)
    pair_rows = []
    for group in group_records(records):
        p_pairs,h_pairs = pair_indices(group)
        indices = [records.index(r) for r in group]
        for kind,pairs in (('purpose',p_pairs),('history',h_pairs)):
            for a,b in pairs:
                left,right = diagnostics[indices[a]],diagnostics[indices[b]]
                pair_rows.append(dict(kind=kind,left=left['record'],right=right['record'],
                    predicted_winner_changes=int(np.argmax(left['final_scores']))!=int(np.argmax(right['final_scores'])),
                    target_winner_changes=int(np.argmax(left['target_final_scores']))!=int(np.argmax(right['target_final_scores'])),
                    final_score_delta=(np.asarray(right['final_scores'])-left['final_scores']).tolist(),
                    target_score_delta=(np.asarray(right['target_final_scores'])-left['target_final_scores']).tolist()))
    write_json(output/'comparison.json',dict(scope='Same-scene g2 development only; no automatic visual approval',
        score_definition='Mean any-joint A_w weighted by positive Teacher any-joint support inside each annotation candidate; diagnostics only',
        rows=diagnostics,pairs=pair_rows,evaluation_seconds=time.monotonic()-start))
    lines = ['# Student v2 vs pilot01 — manual review required','',
        'Teacher1578 fixed. g2 is same-scene development evaluation, NOT unseen-room generalization.',
        'Scores are diagnostics; no new automatic approval. Check all scenes, not just reported failures.','',
        '| Scene / purpose / history | Review group | Baseline scores | V2 scores | Proxy target scores |',
        '|---|---|---|---|---|']
    for row in diagnostics:
        r=row['record']; fmt=lambda v:', '.join('%.3f'%x for x in v)
        lines.append('| %s / %s / %s | %s | %s | %s | %s |'%(r['scene_id'],r['prompt_id'],r['history_motion_id'],
            row['review_group'],fmt(row['baseline_final_scores']),fmt(row['final_scores']),fmt(row['target_final_scores'])))
    (output/'REVIEW.md').write_text('\n'.join(lines)+'\n')
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=Path('data/small_room30_adm_lora_v1'))
    p.add_argument('--teacher-summary',type=Path,default=Path('outputs/small_room30_targeted_v5_eval_full01/summary.json'))
    p.add_argument('--baseline-summary',type=Path,default=Path('outputs/small_room30_student_pilot01/summary.json'))
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--warmup-steps',type=int,default=300)
    p.add_argument('--joint-steps',type=int,default=2400)
    p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--width',type=int,default=64)
    p.add_argument('--seed',type=int,default=20260911)
    args=p.parse_args()
    if args.warmup_steps<0 or args.joint_steps<1 or args.width<8 or not math.isfinite(args.lr) or args.lr<=0:
        raise ValueError('Invalid training configuration')
    if args.output_dir.exists(): raise FileExistsError('Choose a NEW output directory; no overwrite/resume')
    print('[PREFLIGHT V2] Full saved1578 cache, source data, corrected labels, immutable pilot01',flush=True)
    data=StudentDataV2(args.dataset,args.teacher_summary)
    baseline,text,clip_sha=load_baseline(args.baseline_summary,data.binding)
    audit=target_audit(data)
    args.output_dir.mkdir(parents=True)
    manifest=dict(binding=data.binding,code_sha256={n:sha(Path(__file__).parent/n) for n in CODE_FILES},
        seed=args.seed,warmup_steps=args.warmup_steps,joint_steps=args.joint_steps,lr=args.lr,width=args.width,
        formula='A_w = A_raw * (0.6*w_R + 0.4*w_H)',
        objective='auxiliary + 2*relation + 2*history + 4*map + 2*paired_contrast + .05*batch_router_KL',
        baseline_sha256=BASELINE_SHA,clip_weights_sha256=clip_sha,prompt_texts=TEXTS,
        initialization='Fresh v2 student; only frozen CLIP prompt features reused from hash-pinned pilot01',
        scope='30-room development; g0/g1 optimizer only, g2 final comparison only; no AMDM or Teacher training')
    write_json(args.output_dir/'manifest.json',manifest); write_json(args.output_dir/'target_audit.json',audit)
    if args.preflight_only:
        write_json(args.output_dir/'summary.json',dict(status='DATA_PREFLIGHT_PASS_NO_TRAINING',manifest_sha256=sha(args.output_dir/'manifest.json')))
        print('[PASS V2] Preflight only; no optimizer or training.',flush=True); return
    try:
        model,logs=run_training(data,text,args.output_dir,manifest,args.device,args.warmup_steps,args.joint_steps,args.lr,args.seed,args.width)
        rows=evaluate(model,data,text,args.output_dir,args.baseline_summary,baseline,args.device)
        write_json(args.output_dir/'summary.json',dict(status='STUDENT_PILOT_COMPLETE_REVIEW_REQUIRED',
            student_schema='small_room30_weight_student_v2',teacher_unchanged=True,teacher_checkpoint_sha256=TEACHER_SHA,
            teacher_training_performed=False,motion_generation_performed=False,
            student_sha256=sha(args.output_dir/'student.pt'),manifest_sha256=sha(args.output_dir/'manifest.json'),
            clip_weights_sha256=clip_sha,comparison_sha256=sha(args.output_dir/'comparison.json'),rows=rows,log=logs,
            limitations=['Same-scene development only','Geometric history proxy, not human preference GT',
                'One instance per context category supported; not open-vocabulary object discovery',
                'Fixed convex fusion cannot guarantee binary suppression; A=0 remains zero']))
        print('[DONE V2] '+str(args.output_dir/'summary.json')+' — visual review required',flush=True)
    except Exception as exc:
        write_json(args.output_dir/'failure.json',dict(status='FAILED',error=repr(exc))); raise


if __name__=='__main__': main()
