"""Compare instrumented fine-step evidence with fresh uninstrumented evaluations."""
import concurrent.futures
import core as c
from prepare import vec

def check(old):
    t=c.task(old['wave'],vec(old),'verification_plain',scale=old['task']['dt_scale'])
    raw=c.e.run_one(t);a=c.policy.assess(raw)
    if old['full']:
        okay=old['metrics']==raw['metrics'] and old['D']==a['D'] and old['feasible']==a['feasible']
        return dict(id=old['id'],dt_scale=t['dt_scale'],passed=okay,all_metrics_exact=okay,reference=old['record'],plain=raw['record_path'])
    shared=set(old['failed'])&set(a['failed'])
    return dict(id=old['id'],dt_scale=t['dt_scale'],passed=bool(shared) and not a['feasible'],
        prefix_gates=old['failed'],full_failed=a['failed'],shared_violations=sorted(shared),reference=old['record'],plain=raw['record_path'])

def main():
    rows=c.records();plan=[]
    for scale,w in [(.5,2.),(.25,.5)]:
        good=[r for r in rows if r['task']['dt_scale']==scale and r['wave']==w and r['full'] and r['feasible']]
        plan.append(max(good,key=lambda r:r['D']))
    bad=next(r for r in rows if r['task']['dt_scale']==.5 and r.get('proven_infeasible') and r['wave']==1.5)
    plan.append(bad);c.write(c.DATA/'screening_verification_plan.json',plan)
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:checks=list(pool.map(check,plan))
    result=dict(passed=all(r['passed'] for r in checks),screening_sha256=c.sha(c.ROOT/'screening.py'),checks=checks)
    c.write(c.DATA/'screening_verification.json',result);print(result,flush=True);assert result['passed']

if __name__=='__main__':main()
