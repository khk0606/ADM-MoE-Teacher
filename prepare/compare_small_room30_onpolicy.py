#!/usr/bin/env python3
"""Compare paired generated maps against the actual step-1500 source, all actions."""
import argparse
from pathlib import Path
from small_room30_adm_dataset import sha
from small_room30_evaluation_common import validate_report
from small_room30_training_contract import write_json


def compare_reports(before, after):
    if [r['case'] for r in before['rows']] != [r['case'] for r in after['rows']]:
        raise ValueError('Panels differ; cannot make a paired comparison')
    rows = []
    for old, new in zip(before['rows'], after['rows']):
        if old['seeds'] != new['seeds']: raise ValueError('Paired seeds differ')
        a = old['metrics']['adapted_raw']['mae_raw']
        b = new['metrics']['adapted_raw']['mae_raw']
        rows.append(dict(case=new['case'], source_mae=a, pilot_mae=b, delta=b-a,
                         mae_worsened=b > a + 1e-6))
    return dict(status='PAIRED_DEVELOPMENT_COMPARISON_COMPLETE', rows=rows,
        lie_write_worsened=[r for r in rows if r['case']['action'] != 'sit' and r['mae_worsened']],
        note='MAE increases are regression warnings, not a standalone quality gate. Review furniture_coverage.json and Viser.',
        teacher_checkpoint_authorized=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--summary', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise ValueError('Refusing to overwrite comparison')
    old, old_manifest = validate_report(a.baseline)
    new, manifest = validate_report(a.summary)
    if old_manifest['checkpoint_sha256'] != manifest['training_binding']['source_checkpoint']['sha256']:
        raise ValueError('Not the actual warm-start source')
    result = compare_reports(old, new)
    result.update(baseline_sha256=sha(a.baseline), summary_sha256=sha(a.summary))
    write_json(a.output, result)
    for row in result['rows']:
        print('[%s] %s MAE %.6f -> %.6f' % ('WORSENED' if row['mae_worsened'] else 'NOT_WORSE',
            row['case']['sample_id'], row['source_mae'], row['pilot_mae']), flush=True)
    print('[NOTICE] Compare actual furniture presence; no automatic continuation or Teacher approval.',flush=True)


if __name__ == '__main__': main()
