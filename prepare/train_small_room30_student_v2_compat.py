#!/usr/bin/env python3
"""V2 compatibility entrypoint: never deserialize a checkpoint on the GPU host.

Only the baseline reader is replaced. The model, optimizer, labels, evaluation,
Teacher contract and original v2 source/checksums remain byte-identical.
"""
import json
from pathlib import Path
import numpy as np
import train_small_room30_student_v2 as v2
from small_room30_adm_dataset import sha, inside
from small_room30_student_data import TEXTS

CACHE_FILE='small_room30_student_v2_text_cache.json'
CACHE_SHA='cce3a0a45a0e4a75d7e20361af3fef0217078b4180b01af8f10d92c9902cda71'


def load_baseline_without_pickle(summary_path,binding):
    import torch
    root=summary_path.resolve().parent
    report=json.loads(summary_path.read_text())
    if report.get('status')!='STUDENT_PILOT_COMPLETE_REVIEW_REQUIRED':raise ValueError('Incomplete baseline')
    manifest_path=root/'manifest.json'
    if sha(manifest_path)!=report['manifest_sha256']:raise ValueError('Baseline manifest hash mismatch')
    manifest=json.loads(manifest_path.read_text())
    for field in ('teacher_checkpoint_sha256','teacher_summary_sha256','dataset_index_sha256','dataset_manifest_sha256','records'):
        if manifest['binding'][field]!=binding[field]:raise ValueError('Baseline binding mismatch: '+field)
    # Hash the original file, but NEVER torch.load/pickle.load it. The numeric
    # cache is pinned separately to this exact audited checkpoint and manifest.
    if sha(root/'student.pt')!=v2.BASELINE_SHA or report['student_sha256']!=v2.BASELINE_SHA:
        raise ValueError('Expected audited pilot01 checkpoint')
    path=Path(__file__).parent/CACHE_FILE
    if sha(path)!=CACHE_SHA:raise ValueError('Frozen text cache hash mismatch')
    cache=json.loads(path.read_text())
    if (cache['schema']!='small_room30_student_v2_frozen_text_cache_v1'
            or cache['source_checkpoint_sha256']!=v2.BASELINE_SHA
            or cache['source_manifest_sha256']!=report['manifest_sha256']
            or cache['prompt_texts']!=TEXTS or cache['completed_steps']!=600
            or cache['clip_weights_sha256']!=report['clip_weights_sha256']):
        raise ValueError('Frozen text cache provenance mismatch')
    expected=[r for r in binding['records'] if r['split']!='train']
    if [r['record'] for r in report['rows']]!=expected:raise ValueError('Baseline evaluation coverage mismatch')
    for row in report['rows']:
        if sha(inside(root,row['file']))!=row['sha256']:raise ValueError('Baseline map hash mismatch')
    if set(cache['text_features'])!=set(TEXTS):raise ValueError('Missing frozen prompt features')
    text={}
    for key,value in cache['text_features'].items():
        array=np.asarray(value,dtype=np.float32)
        if array.shape!=(1,512) or not np.isfinite(array).all():raise ValueError('Invalid frozen text features')
        if not np.allclose(np.linalg.norm(array,axis=-1),1,atol=1e-4):raise ValueError('Unnormalized text')
        text[key]=torch.from_numpy(array.copy())
    print('[COMPAT] Verified frozen numeric text cache; no torch.load, no checkpoint deserialization.',flush=True)
    return report,text,cache['clip_weights_sha256']


def main():
    # Explicit, scoped dependency replacement; record compatibility code and
    # cache hashes in the same manifest used by the unchanged v2 entrypoint.
    old_reader,old_files=v2.load_baseline,v2.CODE_FILES
    v2.load_baseline=load_baseline_without_pickle
    v2.CODE_FILES=old_files+('train_small_room30_student_v2_compat.py',CACHE_FILE)
    try:v2.main()
    finally:v2.load_baseline,v2.CODE_FILES=old_reader,old_files


if __name__=='__main__':main()
