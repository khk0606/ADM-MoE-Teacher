"""Read-only portable data/loader preflight; does not run training."""
import argparse
import json
from pathlib import Path
import numpy as np
from small_room30_adm_dataset import SmallRoom30ADM,FORWARD_KEYS,inside,sha

def validate(root,stats=None,torch_batch=False,allow_research=False):
    d=SmallRoom30ADM(root);idx=d.index;src=idx['source_contact_index']
    source_index=json.loads(inside(d.root,src).read_text())
    by_id={r['id']:r for r in source_index['motions']}
    counts={};groups={}
    for i,row in enumerate(d.samples):
        sample=d.sample(i);counts[row['action']]=counts.get(row['action'],0)+1
        if row['action']=='write_board'and 'sit' in row['text'].lower():raise ValueError('Board writing mislabelled Sit')
        sd=inside(d.root,src).parent/'scenes'/row['scene_id']
        with np.load(sd/'points.npz',allow_pickle=False)as z:
            original=z['points']
            if not np.array_equal(sample['xyz'],original[:,:3]):raise ValueError('Coordinate/order changed')
            if not np.allclose(sample['feat'],original[:,3:],atol=1e-7,rtol=0):raise ValueError('RGB normalized twice')
        maps=[]
        for mid in row['source_motion_ids']:
            if by_id[mid]['action']!=row['action']:raise ValueError('Action union contaminated')
            with np.load(inside(d.root,src).parent/by_id[mid]['file'],allow_pickle=False)as z:maps.append(z['affordance'].copy())
        if not np.array_equal(sample['target'],np.maximum.reduce(maps)):raise ValueError('Teacher union is incomplete')
        expected_ids={r['id']for r in source_index['motions']if r['scene_id']==row['scene_id']and r['action']==row['action']}
        if set(row['source_motion_ids'])!=expected_ids:raise ValueError('Dropped action candidate')
        key=(row['scene_id'],row['action'])
        if key in groups and not np.array_equal(groups[key],sample['target']):raise ValueError('Purpose changed base Teacher target')
        groups[key]=sample['target']
    if counts!={'sit':65,'lie':5,'write_board':10}:raise ValueError('Prompt/action coverage changed')
    lies=[r for r in source_index['motions']if r['action']=='lie']
    if any(r['source_frames_used']!=301 or r['frames_20fps']!=201 or r['motion_phase']!='standing_approach_to_lie'for r in lies):raise ValueError('Restored Lie was cropped')
    # Identity statistics test only checks algebra; it is never written as training stats.
    x,kwargs=d.batch([0,1],np.zeros((1,6),np.float32),np.ones((1,6),np.float32),allow_research=True)
    if set(kwargs)!=set(FORWARD_KEYS)or x.shape!=(2,8192,6):raise ValueError('ADM forward contract mismatch')
    stats_result='NOT_PROVIDED: normalized training batch still needs pretrained ADM statistics'
    if stats:
        with np.load(stats,allow_pickle=False)as z:
            mean=z['mean'];std=z['std'];x,kwargs=d.batch([0,1],mean,std,allow_research=allow_research)
        gt=np.stack([d.sample(i)['target']for i in [0,1]])
        if not np.allclose(x*std[None]+mean[None],gt,atol=2e-6):raise ValueError('Normalization roundtrip failed')
        stats_result=dict(status='NORMALIZATION_PASS',sha256=sha(Path(stats)))
    if torch_batch:
        if not stats:raise ValueError('--torch-batch requires --stats')
        tx,kw=d.torch_batch([0,1],stats,allow_research=allow_research)
        assert tuple(tx.shape)==(2,8192,6)and set(kw)==set(FORWARD_KEYS)
    return dict(status='EXPORT_AND_NUMPY_LOADER_PASS',scenes=30,source_motions=75,
        teacher_scene_action_targets=len(groups),prompt_samples=len(d),prompt_action_counts=counts,
        lie_frames_30fps=301,lie_frames_20fps=201,adm_input_shape=[8192,6],
        coordinate='AMDM Z-up XZY, metres',rgb='stored 0..255; forward 0..1 exactly once',
        statistics=stats_result,torch_batch_checked=torch_batch,model_forward_tested=False,training_started=False,
        split='all 30 scenes are development data; no independent held-out claim',teacher_checkpoint_authorized=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('data/small_room30_adm_lora_v1'))
    p.add_argument('--stats',type=Path);p.add_argument('--torch-batch',action='store_true');p.add_argument('--allow-research',action='store_true')
    a=p.parse_args();print(json.dumps(validate(a.root,a.stats,a.torch_batch,a.allow_research),indent=2))
