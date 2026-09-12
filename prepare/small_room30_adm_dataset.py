"""Manifest-based small-room ADM input; deliberately separate from v10.x gates.

NumPy validation/inspection works without Torch. No optimizer or training is run.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

FORWARD_KEYS = ('c_pc_xyz', 'c_pc_feat', 'c_text')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def inside(root, relative):
    path=(root/relative).resolve()
    if root.resolve() not in path.parents:raise ValueError('Unsafe dataset path')
    return path

class SmallRoom30ADM:
    def __init__(self, root, *, verify=True):
        self.root=Path(root).resolve()
        if verify:
            for line in (self.root/'SHA256SUMS.txt').read_text().splitlines():
                digest,rel=line.split(maxsplit=1)
                if sha(inside(self.root,rel))!=digest:raise ValueError('Export hash mismatch: '+rel)
        self.index=json.loads((self.root/'index.json').read_text())
        if self.index['schema']!='small_room30_adm_lora_v1':raise ValueError('Wrong schema')
        self.samples=self.index['samples']
        if len(self.samples)!=80 or len(self.index['scenes'])!=30:raise ValueError('Incomplete dataset')
        if len({s['id']for s in self.samples})!=80:raise ValueError('Duplicate sample ID')
        self.scenes={s['id']:s for s in self.index['scenes']}

    def __len__(self):return len(self.samples)

    def sample(self, idx):
        row=self.samples[idx];scene=self.scenes[row['scene_id']]
        with np.load(inside(self.root,scene['points_file']),allow_pickle=False)as z:
            if set(z.files)!={'points'}:raise ValueError('Labels entered point features')
            points=z['points'].astype(np.float32)
        with np.load(inside(self.root,row['target_file']),allow_pickle=False)as z:
            if set(z.files)!={'affordance'}:raise ValueError('Unexpected target keys')
            target=z['affordance'].astype(np.float32)
        if points.shape!=(8192,6)or target.shape!=(8192,6):raise ValueError('Expected 8192 x 6')
        if not np.isfinite(points).all()or not np.isfinite(target).all():raise ValueError('Nonfinite data')
        if np.any((points[:,3:]<0)|(points[:,3:]>255))or np.any((target<0)|(target>1)):raise ValueError('Invalid RGB/GT range')
        return dict(id=row['id'],scene_id=row['scene_id'],action=row['action'],text=row['text'],
                    xyz=points[:,:3].copy(),feat=points[:,3:].copy()/255.,target=target)

    def batch(self, indices, mean, std, *, allow_research=False):
        if not allow_research:raise ValueError('Research data: pass allow_research=True explicitly; this is not Teacher approval')
        mean=np.asarray(mean,np.float32);std=np.asarray(std,np.float32)
        if mean.shape!=(1,6)or std.shape!=(1,6)or not np.isfinite(mean).all()or not np.isfinite(std).all()or np.any(std<=0):
            raise ValueError('Use pretrained ADM contact statistics, exact [1,6], positive std')
        if not indices:raise ValueError('Empty batch')
        rows=[self.sample(i)for i in indices]
        gt=np.stack([r['target']for r in rows])
        # Matches Teacher-v10 _build_batch / _kwargs, not its hard-coded two-room policy.
        return ((gt-mean[None])/std[None]).astype(np.float32),dict(
            c_pc_xyz=np.ascontiguousarray(np.stack([r['xyz']for r in rows])),
            c_pc_feat=np.ascontiguousarray(np.stack([r['feat']for r in rows])),
            c_text=[r['text']for r in rows])

    def torch_batch(self, indices, stats_file, device='cpu', *, allow_research=False):
        import torch
        with np.load(stats_file,allow_pickle=False)as z:
            x,kwargs=self.batch(indices,z['mean'],z['std'],allow_research=allow_research)
        return torch.from_numpy(x).to(device),{
            k:(v if k=='c_text'else torch.from_numpy(v).to(device))for k,v in kwargs.items()}
