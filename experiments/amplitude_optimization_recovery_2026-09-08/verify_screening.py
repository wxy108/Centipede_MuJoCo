import concurrent.futures
from pathlib import Path
import core as c
import screening

def check(r):
    t=dict(r['task'],stage='screening_verification',trace=False)
    screening.install(True)
    try:
        raw=c.e.run_one(t)
    except screening.PrefixRejected as ex:
        assert not r['feasible'],(r['id'],ex.proof)
        actual_failed=set(r['failed']);names={v['gate'].replace('upright_deficit','upright') for v in ex.proof['violations']}
        assert names<=actual_failed,(names,actual_failed)
        return dict(id=r['id'],passed=True,kind='conservative_rejection_agrees_with_full_evaluator',proof=ex.proof)
    saved=c.read(r['record'])
    for k,v in saved['metrics'].items():assert c.np.array_equal(c.np.asarray(v),c.np.asarray(raw['metrics'][k]),equal_nan=True),(r['id'],k)
    return dict(id=r['id'],passed=True,kind='all_metrics_bitwise_equal',source=r['record'],instrumented=raw['record_path'])

if __name__=='__main__':
    rows=c.nominal();chosen=[]
    for w in [1.5,2.,3.5,7.]:chosen.append(max([r for r in rows if r['wave']==w and r['feasible']],key=lambda r:r['D']))
    chosen += sorted([r for r in rows if not r['feasible'] and r['full']],key=lambda r:r['D'],reverse=True)[:2]
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as pool:results=list(pool.map(check,chosen))
    c.write(c.DATA/'screening_verification.json',dict(passed=True,checks=results,screening_sha256=c.sha(c.ROOT/'screening.py')))
    print(results)
