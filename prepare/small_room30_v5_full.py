"""Evaluation-only contract for the user's visually accepted saved1578 adapter."""
import json
from pathlib import Path
import numpy as np
from small_room30_adm_dataset import sha
from small_room30_targeted_v5 import source_contract, FILES as TRAIN_FILES, instance_support, presence
from small_room30_evaluation_common import load_case, case_metrics, seeds

RUN_SHA='422fab37049f978087f039aaa223446ac1fefbe23563e6f4817a6c67de8e0588'
CP_SHA='1b0b4da93e376e007c8e489bf390586d7108ecd46cc1cd523d9787b6a23e814a'
MAP_SHA='a273710f46d34f53889e8495391dc171b58f833c2ebb0ff35a00dd5292eb195d'
FILES=['prepare/'+n+'.py' for n in ('small_room30_v5_full','evaluate_small_room30_v5_full',
    'review_small_room30_v5_full','test_small_room30_v5_full')]


def validate_run(data,path,repo):
    path=Path(path)
    if sha(path)!=RUN_SHA:raise ValueError('Need reviewed v5 run01 ending1578')
    report=json.loads(path.read_text())
    _,_,bound,_=source_contract(data,Path(report['source_summary']),
        Path(report['evaluation_summary']),Path(report['diagnosis_summary']),repo)
    cp=report.get('last_checkpoint')
    if (report['status']!='TARGETED_V5_FINISHED_REVIEW_REQUIRED' or report['binding']!=bound
            or not cp or cp['step']!=1578 or report['completed_step']!=1578
            or cp['sha256']!=CP_SHA or sha(cp['path'])!=CP_SHA
            or report.get('teacher_checkpoint_authorized') is not False):
        raise ValueError('Wrong saved checkpoint/provenance')
    receipt=report['transactions'][-1]['attempts'][-1]
    maps=path.parent/'proposal_12_try_1'/'summary.json'
    if sha(maps)!=MAP_SHA or receipt['canary_sha256']!=MAP_SHA or not receipt['accepted']:
        raise ValueError('Reviewed1578 maps changed')
    panel=json.loads(maps.read_text())
    if panel['adapter_digest']!=cp['adapter_digest']:raise ValueError('Map/adapter identity mismatch')
    for row in panel['rows']:
        file=maps.parent/row['case']['filename']
        if sha(file)!=row['sha256']:raise ValueError('Reviewed map hash mismatch')
    return report,{},bound


def original_case(data,summary,case):
    summary=Path(summary);report=json.loads(summary.read_text())
    row=next(r for r in report['rows'] if r['case']==case)
    path=summary.parent/'cases'/case['filename']
    if sha(path)!=row['sha256'] or row['seeds']!=seeds(case):raise ValueError('Original map/hash/seeds changed')
    arrays=load_case(path)
    if row['metrics']!=case_metrics(arrays):raise ValueError('Original metrics changed')
    from small_room30_evaluation_common import conditioning_arrays,supervision
    points,_=conditioning_arrays(data,case['index']);gt,masks,ids=supervision(data,case['index'])
    for key,value in [('points',points),('gt',gt),('active_masks',masks),('motion_ids',np.asarray(ids))]:
        if not np.array_equal(arrays[key],value):raise ValueError('Original/data pairing differs: '+key)
    return arrays


def verify_reviewed_path(training_path,case,raw):
    root=Path(training_path).parent/'proposal_12_try_1'
    report=json.loads((root/'summary.json').read_text())
    rows=[r for r in report['rows'] if r['case']==case]
    if not rows:return None
    path=root/case['filename']
    if sha(path)!=rows[0]['sha256']:raise ValueError('Reviewed map changed')
    with np.load(path,allow_pickle=False) as z:expected=z['adapted_raw']
    if not np.allclose(raw,expected,atol=1e-5,rtol=1e-5):
        raise ValueError('1578 exact-seed replay differs: '+case['filename'])
    return float(np.abs(raw-expected).max())


def comparison(data,case,old,new):
    for key in ('points','gt','active_masks','motion_ids','base_raw'):
        if not np.array_equal(old[key],new[key]):raise ValueError('Paired evaluation differs: '+key)
    masks,names,_=instance_support(data,case['index'])
    before,after=[presence(a['adapted_raw'],masks,names) for a in (old,new)]
    lost=[a['id'] for b,a in zip(before['instances'],after['instances']) if b['present'] and not a['present']]
    return dict(case=case,original1560=before,selected1578=after,lost_original_targets=lost)
