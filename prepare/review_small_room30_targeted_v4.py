"""Read-only whole-instance full-eval comparison and portable evidence collection."""
import argparse
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from small_room30_adm_dataset import SmallRoom30ADM, sha
from small_room30_evaluation_common import validate_report, load_case
from small_room30_targeted_v4 import POLICY, instance_support, presence, FILES
from small_room30_training_contract import write_json


def compare(data, source_path, candidate_path, run):
    source, sm = validate_report(source_path); candidate, cm = validate_report(candidate_path)
    if sm['mode'] != 'full' or cm['mode'] != 'full' or sm['cases'] != cm['cases']:
        raise ValueError('Need paired full240 evaluation')
    cp = run['last_checkpoint']
    if cm['checkpoint_sha256'] != cp['sha256'] or cm['training_binding'] != run['binding']:
        raise ValueError('Candidate is not this saved v4 checkpoint')
    if sha(source_path) != run['binding']['evaluation_sha256']:
        raise ValueError('Original1560 full evaluation differs')
    rows, lost = [], []
    for sr, cr in zip(source['rows'], candidate['rows']):
        c = sr['case']
        if cr['case'] != c: raise ValueError('Case pairing differs')
        a = load_case(Path(source_path).parent/'cases'/c['filename'])
        b = load_case(Path(candidate_path).parent/'cases'/c['filename'])
        if not np.array_equal(a['points'], b['points']) or not np.array_equal(a['gt'], b['gt']):
            raise ValueError('Geometry/GT pairing changed')
        masks, names, _ = instance_support(data, c['index'])
        before, after = [presence(x['adapted_raw'], masks, names) for x in (a, b)]
        for x, y in zip(before['instances'], after['instances']):
            if x['present'] and not y['present']: lost.append(dict(case=c, target=x['id']))
        rows.append(dict(case=c, before=before, after=after))
    targets = [r for r in rows if r['case']['filename'] in POLICY['target_paths']]
    if len(rows) != 240 or len(targets) != 4: raise ValueError('Incomplete full evaluation')
    return dict(status='FULL_V4_VISUAL_REVIEW_REQUIRED', rows=rows,
        exact_target_paths=targets, lost_previous_targets=lost,
        recovered_paths=[r['case']['filename'] for r in targets if r['after']['all_furniture_present']],
        numerical_presence_proxy_only=True, policy=POLICY, teacher_checkpoint_authorized=False)


def collect(run_path, ready_path, destination, full_path=None):
    run_path, ready_path = Path(run_path).resolve(), Path(ready_path).resolve()
    files = set()
    folders = [run_path.parent, ready_path.parent]
    if full_path is not None: folders.append(Path(full_path).resolve().parent)
    for folder in folders:
        if REPO/'outputs' not in folder.parents or not folder.name.startswith('small_room30_targeted_v4_'):
            raise ValueError('Collect only exact v4 output folders under AMDM/outputs')
        for p in folder.rglob('*'):
            if p.is_symlink(): raise ValueError('Symlink in review folder')
            if p.is_file() and p.suffix in ('.json', '.jsonl', '.npz', '.log'):
                files.add(p)
        console = folder.with_name(folder.name + '.console.log')
        if console.is_file() and not console.is_symlink(): files.add(console)
    if run_path not in files or ready_path not in files: raise ValueError('Missing run/readiness summary')
    files.update(REPO/n for n in FILES)
    files.update((REPO/'scripts/small_room30').glob('*targeted_v4.sh'))
    files.add(REPO/'scripts/small_room30/view_candidates_v4.sh')
    files.add(REPO/'SMALL_ROOM30_TARGETED_V4.md')
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files): z.write(p, str(p.relative_to(REPO)))
    with zipfile.ZipFile(destination) as z:
        if z.testzip() is not None: raise ValueError('Review ZIP CRC failure')
    print('[REVIEW ZIP] %s sha256=%s; checkpoint files excluded, keep them on server' % (destination, sha(destination)), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-summary', type=Path, default=Path('outputs/small_room30_targeted_v4_run01/summary.json'))
    p.add_argument('--readiness', type=Path, default=Path('outputs/small_room30_targeted_v4_ready01/summary.json'))
    p.add_argument('--dataset', type=Path, default=Path('data/small_room30_adm_lora_v1'))
    p.add_argument('--full-summary', type=Path)
    p.add_argument('--zip', type=Path)
    a = p.parse_args()
    if Path.cwd().resolve() != REPO: raise ValueError('Run from AMDM root')
    run = json.loads(a.run_summary.read_text())
    if run.get('binding', {}).get('policy') != POLICY: raise ValueError('Not this v4 run')
    if run.get('status') not in ('TARGETED_V4_STOPPED_REJECTED', 'TARGETED_V4_FINISHED_REVIEW_REQUIRED'):
        raise ValueError('Run must be finished/stopped before collection')
    if sha(a.readiness) != run['readiness_sha256']: raise ValueError('Readiness differs')
    if a.full_summary:
        report = compare(SmallRoom30ADM(a.dataset), Path(run['evaluation_summary']), a.full_summary, run)
        out = a.full_summary.parent/'presence_review.json'
        if out.exists():
            if json.loads(out.read_text()) != report: raise ValueError('Existing review differs')
        else: write_json(out, report)
        print('[FULL PRESENCE] repaired=%d/4; previously present targets lost=%d; visual judgment required' %
              (len(report['recovered_paths']), len(report['lost_previous_targets'])), flush=True)
    if a.zip: collect(a.run_summary, a.readiness, a.zip, a.full_summary)
    if not a.zip and not a.full_summary: p.error('Choose --zip and/or --full-summary')


if __name__ == '__main__': main()
