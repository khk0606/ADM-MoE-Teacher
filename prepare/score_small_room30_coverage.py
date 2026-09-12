#!/usr/bin/env python3
"""Read saved maps only: score both furniture surfaces and background flooding."""
import argparse
from pathlib import Path
import numpy as np
from small_room30_adm_dataset import SmallRoom30ADM, sha
from small_room30_training_contract import write_json
from small_room30_evaluation_common import validate_report, load_case
from small_room30_coverage import POLICY, surface_support, surface_metrics


def score(data, path):
    report, manifest = validate_report(path)
    bound = manifest['training_binding']
    if 'original_binding' in bound: bound = bound['original_binding']
    if (bound['dataset_index_sha256'] != sha(data.root / 'index.json')
        or bound['dataset_checksums_sha256'] != sha(data.root / 'SHA256SUMS.txt')):
        raise ValueError('Evaluation dataset differs from surface annotations')
    rows = []
    for row in report['rows']:
        case = row['case']
        if case['action'] != 'sit': continue
        index = case['index']
        if data.samples[index]['id'] != case['sample_id']: raise ValueError('Case index mismatch')
        arrays = load_case(path.parent / 'cases' / case['filename'])
        sample = data.sample(index)
        if not np.array_equal(sample['target'], arrays['gt']) or not np.array_equal(sample['xyz'], arrays['points'][:, :3]):
            raise ValueError('Case GT/point ordering differs from supervision')
        masks, names, _ = surface_support(data, index)
        if len(names) != 2: raise ValueError('Expected two Sit targets')
        rows.append(dict(case=case, base=surface_metrics(arrays['base_raw'], arrays['gt'], masks, names),
            trained=surface_metrics(arrays['adapted_raw'], arrays['gt'], masks, names)))
    if not rows: raise ValueError('No Sit cases')
    return rows, manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary', type=Path, required=True)
    p.add_argument('--baseline', type=Path)
    p.add_argument('--dataset', type=Path, default=Path('data/small_room30_adm_lora_v1'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise ValueError('Refusing to overwrite coverage score')
    data = SmallRoom30ADM(a.dataset)
    rows, manifest = score(data, a.summary)
    comparisons = []
    if a.baseline:
        old, old_manifest = score(data, a.baseline)
        original = manifest['training_binding'].get('source_checkpoint')
        if original is None or old_manifest['checkpoint_sha256'] != original['sha256']:
            raise ValueError('Baseline is not the actual warm-start checkpoint')
        if [r['case'] for r in old] != [r['case'] for r in rows]:
            raise ValueError('Baseline and new panel must match, including seeds/generations')
        for before, after in zip(old, rows):
            comparisons.append(dict(case=after['case'],
                before_pass=before['trained']['coverage_pass'], after_pass=after['trained']['coverage_pass'],
                before_present=before['trained']['all_furniture_present'],
                after_present=after['trained']['all_furniture_present'],
                regressed=before['trained']['coverage_pass'] and not after['trained']['coverage_pass']))
    passed = sum(r['trained']['coverage_pass'] for r in rows)
    result = dict(status='DEVELOPMENT_SIT_COVERAGE_PASS' if passed == len(rows) else 'DEVELOPMENT_SIT_COVERAGE_NOT_PASS',
        policy=POLICY, summary_sha256=sha(a.summary), baseline_sha256=sha(a.baseline) if a.baseline else None,
        scope=manifest['scope'], covered_cases=passed, total_sit_cases=len(rows), rows=rows,
        comparison=comparisons, teacher_checkpoint_authorized=False,
        lie_write_quality_approved=False, generalization_tested=False)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(a.output, result)
    for r in rows:
        hits = ', '.join('%s=%.3f' % (x['id'], x['hit_fraction']) for x in r['trained']['instances'])
        print('[%s] %s: %s; background=%s' % (
            'PASS' if r['trained']['coverage_pass'] else 'FAIL', r['case']['sample_id'], hits,
            r['trained']['background_point_hit_fraction']))
    print('[%s] %d/%d Sit cases; %s' % (result['status'], passed, len(rows), a.output))
    print('[NOTICE] Successful execution is not a coverage pass; inspect status. No Teacher approval.')


if __name__ == '__main__': main()
