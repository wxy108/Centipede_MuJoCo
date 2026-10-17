"""Verify added irreversible chassis-contact rejection against saved full runs."""
import argparse
import concurrent.futures
import json
import search as s
e=s.e


def check(wave):
    manifest=json.loads(s.MANIFEST.read_text())
    task=next(t for t in manifest['references'] if t['wave']==wave)
    raw,path=s.load_reference(task)
    return check_record(raw,path)


def check_record(raw,path):
    task=raw['task'];wave=task['wave'];assessment=s.assess(raw)
    proof=s.precheck(dict(task,stage='contact_pruning_verification'))
    if assessment['feasible']:assert proof is None
    if proof:
        assert not assessment['feasible']
        if 'chassis_contact_irreversible' in proof['gates']:
            assert raw['metrics']['chassis_contact_fraction']>.03
            p=proof['chassis_contact_proof']
            assert p['bad_step_lower_bound']/p['max_final_fine_steps']>.03
    return dict(wave=wave,full_record=str(path),full_failed=assessment['failed'],
        full_chassis_contact_fraction=raw['metrics']['chassis_contact_fraction'],
        proof=proof,consistent=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--exercise-only',action='store_true');args=parser.parse_args()
    if args.exercise_only:
        tid='flexible_flat_w8_4ce333d1c03fff'
        path=e.TRIALS/'broad'/tid/f'{tid}.json'
        result=check_record(json.loads(path.read_text()),path)
        assert result['proof'] and 'chassis_contact_irreversible' in result['proof']['gates']
        (e.DATA/'chassis_contact_pruning_branch_exercised.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(dict(full_chassis_contact_fraction=result['full_chassis_contact_fraction'],
            checked_until=result['proof']['checked_until_s'],proof=result['proof']['chassis_contact_proof'],consistent=True),indent=2))
        raise SystemExit(0)
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
        rows=list(pool.map(check,(2.5,3.,8.)))
    (e.DATA/'chassis_contact_pruning_verification.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps([dict(wave=r['wave'],full_failed=r['full_failed'],
        prefix_failed=list(r['proof']['gates']) if r['proof'] else [],
        checked_until=r['proof']['checked_until_s'] if r['proof'] else 2.5,
        consistent=True) for r in rows],indent=2))
