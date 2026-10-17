"""Check exact pruning decisions against full, independently recorded runs."""
import concurrent.futures
import json
import search as s
e=s.e

def check(record):
    raw=json.loads(open(record,encoding='utf-8').read());full=s.assess(raw)
    task=dict(raw['task'],stage='pruning_verification');proof=s.precheck(task)
    if full['feasible']:assert proof is None,(record,proof)
    if proof:assert not full['feasible'],(record,proof)
    return dict(record=record,full_failed=full['failed'],pruning_failed=proof['failure_evidence'] if proof else [],
        checked_until=proof['checked_until_s'] if proof else 2.5,decision_consistent=True)

if __name__=='__main__':
    manifest=json.loads(s.MANIFEST.read_text());paths=[]
    for w in (2.5,3.,9.):
        task=next(t for t in manifest['references'] if t['wave']==w);tid=e.task_id(task)
        paths.append(str(e.TRIALS/'reference'/tid/f'{tid}.json'))
    paths.append(str(e.TRIALS/'broad'/'flexible_flat_w0p5_ccbe51b8448a10'/'flexible_flat_w0p5_ccbe51b8448a10.json'))
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(check,paths))
    (e.DATA/'exact_pruning_verification.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows,indent=2))
