"""Collect bounded v5 evidence without loading a model or collecting checkpoint weights."""
import argparse
import json
from pathlib import Path
import zipfile
from small_room30_adm_dataset import sha
from small_room30_targeted_v5 import POLICY, FILES

REPO=Path(__file__).resolve().parents[1]


def collect(run_path,ready_path,destination):
    run_path=Path(run_path).resolve();ready_path=Path(ready_path).resolve()
    run=json.loads(run_path.read_text())
    if (run.get('binding',{}).get('policy')!=POLICY or run.get('status') not in
            ('TARGETED_V5_STOPPED_REJECTED','TARGETED_V5_FINISHED_REVIEW_REQUIRED')):
        raise ValueError('Need this completed/stopped v5 run')
    if sha(ready_path)!=run['readiness_sha256']:raise ValueError('Readiness differs')
    files=set()
    for folder in (run_path.parent,ready_path.parent):
        if REPO/'outputs' not in folder.parents or not folder.name.startswith('small_room30_targeted_v5_'):
            raise ValueError('Collect only exact v5 output folders')
        for p in folder.rglob('*'):
            if p.is_symlink():raise ValueError('Symlink in evidence')
            if p.is_file() and p.suffix in ('.json','.jsonl','.npz','.log'):files.add(p)
        log=folder.with_name(folder.name+'.console.log')
        if log.is_file() and not log.is_symlink():files.add(log)
    files.update(REPO/n for n in FILES)
    files.update((REPO/'scripts/small_room30').glob('*targeted_v5.sh'))
    files.add(REPO/'scripts/small_room30/view_candidates_v5.sh')
    files.add(REPO/'docs/guides/teacher-training.md')
    with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):z.write(p,str(p.relative_to(REPO)))
    with zipfile.ZipFile(destination) as z:
        if z.testzip() is not None:raise ValueError('ZIP CRC failure')
    print('[REVIEW ZIP] %s sha256=%s; keep checkpoint weights on server'%(destination,sha(destination)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-summary',type=Path,default=Path('outputs/small_room30_targeted_v5_run01/summary.json'))
    p.add_argument('--readiness',type=Path,default=Path('outputs/small_room30_targeted_v5_ready01/summary.json'))
    p.add_argument('--zip',type=Path,required=True)
    a=p.parse_args()
    if Path.cwd().resolve()!=REPO:raise ValueError('Run from AMDM root')
    collect(a.run_summary,a.readiness,a.zip)


if __name__=='__main__':main()
