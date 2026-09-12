#!/usr/bin/env python3
"""Furniture presence is primary; GT fit and background are separate warnings."""
import argparse
from pathlib import Path
import numpy as np
from small_room30_adm_dataset import SmallRoom30ADM,sha
from small_room30_evaluation_common import validate_report,load_case,panel
from small_room30_targeted import score
from small_room30_training_contract import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dataset',type=Path,default=Path('data/small_room30_adm_lora_v1'))
    a=p.parse_args()
    if a.output.exists():raise ValueError('Refusing to overwrite review')
    data=SmallRoom30ADM(a.dataset);before,bm=validate_report(a.baseline);after,am=validate_report(a.summary)
    if (bm['checkpoint_sha256']!=am['training_binding']['source_checkpoint']['sha256']
            or bm['cases']!=am['cases'] or am['cases']!=panel(data,'full')):
        raise ValueError('Need paired full evaluation against actual warm start')
    rows=[]
    for old,new in zip(before['rows'],after['rows']):
        case=new['case']
        if old['seeds']!=new['seeds']:raise ValueError('Seeds differ')
        maps=[load_case(path.parent/'cases'/case['filename']) for path in (a.baseline,a.summary)]
        gt=data.sample(case['index'])['target']
        if not np.array_equal(maps[0]['points'],maps[1]['points']) or any(not np.array_equal(m['gt'],gt) for m in maps):
            raise ValueError('Point/GT mismatch')
        scores=[score(data,case,m['adapted_raw']) for m in maps]
        rows.append(dict(case=case,before=scores[0],after=scores[1],
            presence_regressed=scores[0]['all_furniture_present'] and not scores[1]['all_furniture_present']))
    actions={}
    for action in ('sit','lie','write_board'):
        group=[r for r in rows if r['case']['action']==action]
        actions[action]=dict(cases=len(group),before_present=sum(r['before']['all_furniture_present'] for r in group),
            after_present=sum(r['after']['all_furniture_present'] for r in group),
            lost_cases=[r['case']['filename'] for r in group if r['presence_regressed']],
            missing_cases=[r['case']['filename'] for r in group if not r['after']['all_furniture_present']],
            mean_mae_delta=float(np.mean([r['after']['mae']-r['before']['mae'] for r in group])),
            mean_background_delta=float(np.mean([(r['after']['background_point_hit_fraction'] or 0.)-
                (r['before']['background_point_hit_fraction'] or 0.) for r in group])))
        print('[FURNITURE %s] %d -> %d / %d; lost=%d'%(action,actions[action]['before_present'],
            actions[action]['after_present'],len(group),len(actions[action]['lost_cases'])),flush=True)
    result=dict(status='SIT_FURNITURE_PRESENCE_PASS' if not actions['sit']['missing_cases'] else 'SIT_FURNITURE_PRESENCE_NOT_PASS',
        rows=rows,by_action=actions,baseline_sha256=sha(a.baseline),summary_sha256=sha(a.summary),
        teacher_checkpoint_authorized=False,note='Development presence proxy. Background/MAE are separate warnings; visual review required.')
    write_json(a.output,result);print('[%s] %s'%(result['status'],a.output),flush=True)


if __name__=='__main__':main()
