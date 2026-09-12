#!/usr/bin/env python3
"""Visually compare saved v5 attempts, including rejected ones. No Torch/model load."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path,PurePosixPath
import socket
import threading
import time
import zipfile
import numpy as np
from view_small_room30_evaluation import CHANNELS,panels


def digest(blob):return hashlib.sha256(blob).hexdigest()


class Store:
    def __init__(self,summary,archive=None,prefix=''):
        self.root=Path(summary).resolve().parent
        self.archive=zipfile.ZipFile(archive) if archive else None
        self.prefix=prefix

    def read(self,name):
        p=PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts or '\\' in name:raise ValueError('Unsafe evidence path')
        if self.archive:return self.archive.read(self.prefix+name)
        path=(self.root/name).resolve()
        if self.root not in path.parents:raise ValueError('Evidence escapes output folder')
        return path.read_bytes()

    def json(self,name):return json.loads(self.read(name))

    def close(self):
        if self.archive:self.archive.close()


def npz(blob):
    with np.load(io.BytesIO(blob),allow_pickle=False) as z:return {k:z[k] for k in z.files}


class Evidence:
    def __init__(self,run,base):
        self.run=run;self.base=base;self.cache={};self.stages={}
        report=run.json('summary.json');bound=report['binding']
        if report.get('status') not in ('TARGETED_V5_STOPPED_REJECTED','TARGETED_V5_FINISHED_REVIEW_REQUIRED'):
            raise ValueError('Need completed/stopped v5 evidence, not a running experiment')
        if bound['policy']['version']!='small_room30_targeted_v5':raise ValueError('Not v5 evidence')
        if digest(base.read('summary.json'))!=bound['evaluation_sha256']:raise ValueError('Original1560 summary mismatch')
        base_report=base.json('summary.json');manifest=base.json('manifest.json')
        if (digest(base.read('manifest.json'))!=bound['evaluation_manifest_sha256']
                or manifest['checkpoint_sha256']!=bound['original_source_checkpoint']['sha256']):
            raise ValueError('Original1560 manifest/checkpoint mismatch')
        self.initial=run.json('source_canary/summary.json')
        if (self.initial['adapter_digest']!=bound['source_checkpoint']['adapter_digest']
                or self.initial['source_checkpoint_sha256']!=bound['source_checkpoint']['sha256']):
            raise ValueError('Starting1570 snapshot differs')
        self.initial_rows={r['case']['filename']:r for r in self.initial['rows']}
        self.base_rows={r['case']['filename']:r for r in base_report['rows']}
        self.cases=[r['case'] for r in bound['failures']+bound['guards']]
        if len(self.cases)!=16 or len({c['filename'] for c in self.cases})!=16:raise ValueError('Expected16 distinct saved paths')
        for case in self.cases:
            name=case['filename']
            if PurePosixPath(name).name!=name or not name.endswith('.npz'):raise ValueError('Unsafe case name')
            if self.base_rows[name]['case']!=case:raise ValueError('Source case metadata differs')
        self.stages['Starting LoRA 1570']=dict(folder=None,receipt=None,rows=None)
        self.default_stage='Starting LoRA 1570'
        checkpoints={c['adapter_digest']:c for c in report['checkpoints']}
        for transaction in report['transactions']:
            for receipt in transaction['attempts']:
                if not receipt.get('canary_summary'):continue
                proposal=receipt['proposal'];attempt=receipt['attempt']
                folder='proposal_%02d_try_%d/'%(proposal,attempt)
                blob=run.read(folder+'summary.json')
                if digest(blob)!=receipt['canary_sha256']:raise ValueError('Candidate summary hash mismatch')
                candidate=json.loads(blob)
                if candidate['adapter_digest']!=receipt['candidate_adapter_digest']:raise ValueError('Candidate weight identity differs')
                if [r['case'] for r in candidate['rows']]!=self.cases:raise ValueError('Candidate path panel differs')
                if candidate['source_checkpoint_sha256']!=bound['source_checkpoint']['sha256']:raise ValueError('Candidate reference differs')
                saved=run.json(folder+'acceptance.json')
                if saved!={k:v for k,v in receipt.items() if k!='projection'}:raise ValueError('Acceptance receipt differs')
                label='P%d / %s'%(proposal,'full LR' if receipt['lr_scale']==1. else 'quarter LR')
                if receipt['accepted']:
                    cp=checkpoints.get(candidate['adapter_digest'])
                    if not cp:raise ValueError('Accepted candidate has no matching checkpoint receipt')
                    label+=' / saved step %s'%cp['step']
                    self.default_stage=label
                else:label+=' / unsaved candidate'
                self.stages[label]=dict(folder=folder,receipt=receipt,rows={r['case']['filename']:r for r in candidate['rows']})
        if len(self.stages)==1:raise ValueError('No saved candidate maps')

    def arrays(self,stage,case,reference='Starting LoRA 1570'):
        name=case['filename'];key=('base',name)
        if key not in self.cache:
            blob=self.base.read('cases/'+name)
            if digest(blob)!=self.base_rows[name]['sha256']:raise ValueError('Original map hash mismatch')
            a=npz(blob)
            for k in ('points','gt','base_raw','adapted_raw'):
                if a[k].shape!=(8192,6) or not np.isfinite(a[k]).all():raise ValueError('Invalid original arrays')
            self.cache[key]=a
        original=self.cache[key];selected=self.stages[stage];key=(stage,name)
        initial_key=('starting1570',name)
        if initial_key not in self.cache:
            blob=self.run.read('source_canary/'+name)
            if digest(blob)!=self.initial_rows[name]['sha256']:raise ValueError('1570 snapshot hash mismatch')
            initial=npz(blob)
            if any(v.shape!=(8192,6) or not np.isfinite(v).all() for v in initial.values()):raise ValueError('Bad1570 snapshot')
            if not np.allclose(initial['adapted_raw'],initial['source_raw'],atol=1e-5,rtol=1e-5):raise ValueError('1570 not reproduced')
            self.cache[initial_key]=initial['source_raw']
        anchor=self.cache[initial_key]
        if selected['folder'] is None:raw=anchor
        else:
            if key not in self.cache:
                row=selected['rows'][name];blob=self.run.read(selected['folder']+name)
                if digest(blob)!=row['sha256']:raise ValueError('Candidate map hash mismatch')
                a=npz(blob)
                if set(a)!=set(('source_raw','adapted_raw')):raise ValueError('Wrong candidate array keys')
                if any(v.shape!=(8192,6) or not np.isfinite(v).all() for v in a.values()):raise ValueError('Invalid candidate arrays')
                if not np.array_equal(a['source_raw'],anchor):raise ValueError('Candidate/source pairing mismatch')
                self.cache[key]=a['adapted_raw']
            raw=self.cache[key]
        return dict(points=original['points'],gt=original['gt'],adapted_raw=raw,
            base_raw=anchor if reference=='Starting LoRA 1570' else original['base_raw'])

    def verify_all(self):
        for stage in self.stages:
            for case in self.cases:self.arrays(stage,case)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-summary',type=Path,default=Path('outputs/small_room30_targeted_v5_run01/summary.json'))
    p.add_argument('--baseline-summary',type=Path,default=Path('outputs/small_room30_onpolicy_eval_full01/summary.json'))
    p.add_argument('--review-zip',type=Path,help='Optional local review ZIP instead of the remote run folder')
    p.add_argument('--baseline-zip',type=Path,help='Optional full_review01 ZIP instead of the starting1570 folder')
    p.add_argument('--verify-only',action='store_true')
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8092)
    a=p.parse_args()
    run=Store(a.run_summary,a.review_zip,'outputs/small_room30_targeted_v5_run01/')
    base=Store(a.baseline_summary,a.baseline_zip,'outputs/small_room30_onpolicy_eval_full01/')
    try:
        evidence=Evidence(run,base)
        print('[VIEW] Verifying saved maps only; no training or model inference.',flush=True)
        evidence.verify_all()
        print('[VERIFIED] %d saved paths; %d selections including starting1570.'%(len(evidence.cases),len(evidence.stages)),flush=True)
        if a.verify_only:return
        with socket.socket() as probe:probe.bind((a.host,a.port))
        import viser
        def timeout():
            print('[STOP] Viser startup exceeded45s; only this viewer will exit.',flush=True);os._exit(2)
        timer=threading.Timer(45.,timeout);timer.daemon=True;timer.start()
        try:server=viser.ViserServer(host=a.host,port=a.port)
        finally:timer.cancel()
        if hasattr(server,'get_port') and server.get_port()!=a.port:
            server.stop();raise RuntimeError('Viser selected another port')
        server.scene.set_up_direction('+z')
        server.gui.add_markdown('## Saved v5 candidates — visual review\n\n'
            'Repair scope: chair100 in0205/g0 and0403/g2 only; starting from saved1570. You judge target visibility; other paths are preservation checks.\n\n'
            '**16 saved paths only:2 target paths +14 preservation paths. Not a full30-room v5 evaluation.**\n\n'
            'Select saved checkpoints OR unsaved proposal maps. Viewing does not resume training or change approval. '
            'Fixed colors[0,1], original8192 points, no smoothing or map editing.')
        stage=server.gui.add_dropdown('Candidate',options=tuple(evidence.stages),initial_value=evidence.default_stage)
        scenes=sorted({c['scene_id'] for c in evidence.cases})
        scene=server.gui.add_dropdown('Scene',options=tuple(scenes),initial_value=evidence.cases[0]['scene_id'])
        def choices():return {c['sample_id']+' / g'+str(c['generation']):c for c in evidence.cases if c['scene_id']==scene.value}
        selected=server.gui.add_dropdown('Prompt / generation',options=tuple(choices()))
        reference=server.gui.add_dropdown('Reference',options=('Starting LoRA 1570','Frozen Base'),initial_value='Starting LoRA 1570')
        channel=server.gui.add_dropdown('Body channel',options=CHANNELS,initial_value='any_joint')
        size=server.gui.add_slider('Point size (m)',min=.005,max=.08,step=.005,initial_value=.025)
        blend=server.gui.add_slider('RGB blend',min=0.,max=.5,step=.05,initial_value=0.)
        show=server.gui.add_checkbox('Show automatic gate details',initial_value=False)
        details=server.gui.add_markdown('');lock=threading.RLock()
        def render(_=None):
            with lock:
                case=choices().get(selected.value)
                if case is None:return
                arrays=evidence.arrays(stage.value,case,reference.value)
                xyz=arrays['points'][:,:3].copy()
                xyz[:,:2]-=(xyz[:,:2].min(axis=0)+xyz[:,:2].max(axis=0))/2
                extent=max(float(np.ptp(xyz[:,:2],axis=0).max()),1.)+1.2
                titles=('Input RGB','Contact GT',reference.value,'Selected LoRA candidate','Candidate - reference','|Candidate - GT|')
                for i,colors in enumerate(panels(arrays,channel.value,float(blend.value))):
                    offset=np.array([(i%3-1)*extent,(.5-i//3)*extent,0],dtype=np.float32)
                    server.scene.add_point_cloud('/panel%d/points'%i,points=xyz+offset,colors=colors,point_size=float(size.value))
                    server.scene.add_label('/panel%d/title'%i,titles[i],position=offset+np.array([0,extent*.43,max(float(xyz[:,2].max()),0)+.15]))
                meta=evidence.stages[stage.value];receipt=meta['receipt']
                text='**%s**\n\n%s / g%s'%(stage.value,case['sample_id'],case['generation'])
                if show.value and receipt:
                    text+='\n\nHistorical automatic gate: '+('accepted' if receipt['accepted'] else 'rejected')
                    text+='\n\n'+('\n\n'.join(receipt.get('reasons',[])) or 'No rejection reasons.')
                    text+='\n\nWhole target instance / any_joint fractions (score>=0.3; proxy, not visual approval):\n\n'
                    text+='\n\n'.join('%s: %.2f%%'%(v['id'],100*v['hit_fraction']) for v in meta['rows'][case['filename']]['after_presence']['instances'])
                details.content=text
        def change_scene(_):
            with lock:
                opts=choices();selected.options=tuple(opts);selected.value=next(iter(opts));render()
        scene.on_update(change_scene)
        for control in (stage,selected,reference,channel,size,blend,show):control.on_update(render)
        @server.on_client_connect
        def connect(client):
            client.camera.position=(0.,-14.,23.);client.camera.look_at=(0.,0.,0.);client.camera.up_direction=(0.,0.,1.)
        render()
        print('[READY] http://127.0.0.1:%d — reuse the existing VS Code forwarded port.'%a.port,flush=True)
        try:
            while True:time.sleep(1)
        except KeyboardInterrupt:server.stop()
    finally:run.close();base.close()


if __name__=='__main__':main()
