"""Read-only current progress and best nominal candidates; not validation."""
import argparse
import json
import study as s
e = s.e

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('initial', 'coarse', 'polish'), default='initial')
    parser.add_argument('--wave', type=float, help='Show only this wave in the per-wave progress section')
    args = parser.parse_args()
    bank = s.manifest() if args.stage == 'initial' else s.read(e.DATA / f'{args.stage}_manifest.json')
    expected = {e.task_id(task) for task in bank['tasks']}
    rows = []
    for tid in expected:
        path = e.OUT / 'assessments' / f'{tid}.json'
        if path.exists():
            try:
                rows.append(s.read(path))
            except json.JSONDecodeError:
                pass
    groups = {}
    for wave in e.WAVES:
        if args.wave is not None and wave != args.wave:
            continue
        group = [r for r in rows if r['wave'] == wave]
        feasible = sorted((r for r in group if r['feasible']), key=lambda r: -r['D'])
        groups[f'{wave:g}'] = dict(completed=len(group), feasible=len(feasible),
            best=dict(D=feasible[0]['D'], amplitudes=feasible[0]['amplitudes']) if feasible else None)
    print(json.dumps(dict(stage=args.stage, completed=len(rows), total=len(expected),
        reused=sum(r['evaluation_source']=='prior_evidence_reused' for r in rows),
        waves=groups, note='Nominal six-second results, not final validation'), indent=2))
