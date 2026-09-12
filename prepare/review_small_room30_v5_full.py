"""Read-only240-case comparison,16 known1570 anchors, and review ZIP collection."""
import argparse
import json
from pathlib import Path
import zipfile
from small_room30_adm_dataset import SmallRoom30ADM,sha
from small_room30_training_contract import write_json
from small_room30_evaluation_common import validate_report,load_case,panel
from small_room30_v5_full import RUN_SHA,CP_SHA,FILES,comparison,original_case,verify_reviewed_path
from small_room30_targeted_v5 import reference,instance_support,presence

REPO=Path(__file__).resolve().parents[1]


def review(data,summary,training_path):
    if sha(training_path)!=RUN_SHA:raise ValueError('Wrong reviewed v5 run')
    training=json.loads(Path(training_path).read_text())
    report,manifest=validate_report(summary)
    if (manifest['mode']!='full' or manifest['cases']!=panel(data,'full') or
            manifest['checkpoint_sha256']!=CP_SHA or manifest['checkpoint_step']!=1578 or
            manifest['training_summary_sha256']!=RUN_SHA or manifest['training_binding']!=training['binding']):
        raise ValueError('Need exact saved1578 full240 evaluation')
    rows=[];anchors=[];replayed=[]
    initial=Path(training_path).parent/'source_canary'/'summary.json'
    baseline=json.loads(initial.read_text())
    if baseline['source_checkpoint_sha256']!=training['binding']['source_checkpoint']['sha256']:
        raise ValueError('1570 anchor differs')
    known={r['case']['filename'] for r in baseline['rows']}
    if len(known)!=16:raise ValueError('Need16 known1570 paths')
    for row in report['rows']:
        c=row['case'];new=load_case(Path(summary).parent/'cases'/c['filename'])
        old=original_case(data,training['evaluation_summary'],c)
        rows.append(comparison(data,c,old,new))
        error=verify_reviewed_path(training_path,c,new['adapted_raw'])
        if error is not None:replayed.append(dict(case=c,max_abs_error=error))
        if c['filename'] in known:
            raw=reference(initial,c);masks,names,_=instance_support(data,c['index'])
            before=presence(raw,masks,names);after=presence(new['adapted_raw'],masks,names)
            anchors.append(dict(case=c,starting1570=before,selected1578=after,
                lost_1570_targets=[a['id'] for b,a in zip(before['instances'],after['instances']) if b['present'] and not a['present']]))
    if len(replayed)!=16 or len(anchors)!=16:raise ValueError('Missing reviewed paths')
    return dict(status='FULL1578_EVALUATION_REVIEW_REQUIRED',total_cases=240,rooms=30,rows=rows,
        lost_original1560=[r for r in rows if r['lost_original_targets']],known1570_comparison=anchors,
        reviewed1578_reproduction=replayed,reference_limits='1560 full240;1570 known16 only; no full1570 baseline',
        user_visual_acceptance='0205/g0 and0403/g2 approved for saved1578 on2026-09-10',
        criteria='Visibility diagnostics only; no automatic GT-fit rejection or retraining',
        training_started=False,teacher_checkpoint_authorized=False)


def collect(summary,training_path,destination):
    summary=Path(summary).resolve();training_path=Path(training_path).resolve()
    root=summary.parent
    if REPO/'outputs' not in root.parents or root.name!='small_room30_targeted_v5_eval_full01':
        raise ValueError('Only standard full evaluation output may be packaged')
    files={p for p in root.rglob('*') if p.is_file() and p.suffix in ('.json','.npz')}
    files.add(training_path)
    for folder in ('source_canary','proposal_12_try_1'):
        files.update(p for p in (training_path.parent/folder).glob('*') if p.suffix in ('.json','.npz'))
    log=root.with_name(root.name+'.console.log')
    if log.exists():files.add(log)
    files.update(REPO/n for n in FILES)
    files.add(REPO/'docs/guides/teacher-evaluation.md')
    files.update((REPO/'scripts/small_room30').glob('*v5_full.sh'))
    if any(p.is_symlink() or REPO not in p.resolve().parents for p in files):raise ValueError('Unsafe evidence path')
    if Path(destination).exists():
        with zipfile.ZipFile(destination) as z:
            if z.testzip() is not None or set(z.namelist())!={str(p.relative_to(REPO)) for p in files}:
                raise ValueError('Existing review ZIP differs; preserved')
            if any(z.read(str(p.relative_to(REPO)))!=p.read_bytes() for p in files):raise ValueError('Existing review ZIP differs; preserved')
        print('[ZIP VERIFIED] '+str(destination),flush=True);return
    with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):z.write(p,str(p.relative_to(REPO)))
    with zipfile.ZipFile(destination) as z:
        if z.testzip() is not None:raise ValueError('ZIP CRC failure')
    print('[REVIEW ZIP] %s sha256=%s'%(destination,sha(destination)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary',type=Path,default=Path('outputs/small_room30_targeted_v5_eval_full01/summary.json'))
    p.add_argument('--training-summary',type=Path,default=Path('outputs/small_room30_targeted_v5_run01/summary.json'))
    p.add_argument('--dataset',type=Path,default=Path('data/small_room30_adm_lora_v1'))
    p.add_argument('--zip',type=Path,default=Path('small_room30_targeted_v5_full_review01.zip'))
    a=p.parse_args()
    result=review(SmallRoom30ADM(a.dataset),a.summary,a.training_summary)
    out=a.summary.parent/'presence_review.json'
    if out.exists():
        if json.loads(out.read_text())!=result:raise ValueError('Existing comparison differs')
    else:write_json(out,result)
    collect(a.summary,a.training_summary,a.zip)
    print('[COMPLETE] Full240 evaluation and comparison complete. No training or automatic approval.',flush=True)


if __name__=='__main__':main()
