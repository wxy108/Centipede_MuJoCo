"""Full long-horizon and half-step validation with explicit ranked fallbacks."""
import argparse
import core as c
np=c.np
def vec(r):return [r['amplitudes'][k] for k in c.KEYS]
def pair_tasks(row,stage):return [c.task(row['wave'],vec(row),stage,duration=d,scale=dt) for d,dt in [(12.,1.),(6.,.5)]]
def assess(row,index):
    tasks=pair_tasks(row,'validation');rr=[index.get(c.identity(t)) for t in tasks]
    if not all(rr):return dict(id=row['id'],pending=True,qualified=False)
    half=rr[1];delta=abs(half['D']-row['D'])/max(abs(row['D']),1e-12) if half['full'] else None
    reasons=[]
    for label,r in zip(['long_12s','half_step_6s'],rr):
        if not r['feasible']:reasons.append(dict(condition=label,failed=r['failed']))
    if delta is None or delta>.1:reasons.append(dict(condition='half_step_displacement',relative_difference=delta,limit=.1))
    return dict(id=row['id'],pending=False,qualified=not reasons,reasons=reasons,half_step_relative_difference=delta,
        nominal=row,long=rr[0],half=rr[1])
def update_validation_index():
    rows=c.records();index={r['id']:r for r in rows};assessments={}
    for r in c.nominal():
        if r['feasible']:
            v=assess(r,index)
            if not v['pending']:assessments[r['id']]=v
    c.write(c.DATA/'validation_index.json',assessments);return assessments

def early(workers):
    plan=c.DATA/'warm_validation_plan.json'
    if plan.exists():selected=c.read(plan)
    else:
        selected=[]
        for w in c.WAVES:
            group=[r for r in c.read(c.DATA/'warm_fresh_results.json') if r['wave']==w and r['feasible']]
            if group:selected.append(max(group,key=lambda r:r['D']))
        c.write(plan,selected)
    c.bank([t for r in selected for t in pair_tasks(r,'warm_validation')],'warm_validation',4 if workers is None else workers)
    update_validation_index()

def final(workers):
    plan=c.DATA/'final_validation_plan.json'
    if plan.exists():ranked=c.read(plan)
    else:
        rows=c.nominal();ranked={f'{w:g}':sorted([r for r in rows if r['wave']==w and r['feasible']],key=lambda r:r['D'],reverse=True) for w in c.WAVES}
        c.write(plan,ranked)
    checked=update_validation_index();selection={}
    for round_ in range(5):
        saved=c.DATA/'banks'/f'final_validation_{round_:02d}.json'
        if saved.exists():
            c.bank(c.read(saved),f'final_validation_{round_:02d}',workers or 12)
            checked=update_validation_index()
            continue
        tasks=[]
        for w,rr in ranked.items():
            if w in selection:continue
            for r in rr:
                old=checked.get(r['id'])
                if old:
                    if old['qualified']:selection[w]=old;break
                    continue
                tasks+=pair_tasks(r,'final_validation');break
        if not tasks:break
        c.bank(tasks,f'final_validation_{round_:02d}',workers or 12);checked=update_validation_index()
    for w,rr in ranked.items():
        for r in rr:
            if checked.get(r['id'],{}).get('qualified'):selection[w]=checked[r['id']];break
    # A known previous configuration receives a fresh validation fallback if needed.
    baseline=c.read(c.ROOT.parent/'mujoco_displacement_first_amplitude_2026-09-03/outputs/data/interim_validated_selection.json')
    tasks=[]
    for w in ranked:
        if w in selection:continue
        old=baseline['waves'][w]['selected']
        if old and old['qualified']:
            raw=c.read(old['nominal']['record']);x=[raw['task']['amplitudes'][k] for k in c.KEYS]
            row=next(r for r in c.nominal() if r['id']==c.identity(c.task(float(w),x,'fallback')))
            tasks+=pair_tasks(row,'fallback_validation')
    if tasks:c.bank(tasks,'fallback_validation',workers or 12);checked=update_validation_index()
    for w,rr in ranked.items():
        for r in rr:
            if checked.get(r['id'],{}).get('qualified'):selection[w]=checked[r['id']];break
    result={w:dict(selected=selection.get(w),qualified=bool(w in selection),nominal_feasible_count=len(rr),
        candidates_tested=sum(r['id'] in checked for r in rr),higher_untested=sum(r['id'] not in checked and (w not in selection or r['D']>selection[w]['nominal']['D']) for r in rr)) for w,rr in ranked.items()}
    c.write(c.DATA/'validated_selection.json',dict(waves=result,global_optimum_proved=False,validation_completed_for_declared_candidates=True))
    print({w:None if not v['qualified'] else v['selected']['nominal']['D'] for w,v in result.items()},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['early','final']);p.add_argument('--workers',type=int);args=p.parse_args()
    (early if args.stage=='early' else final)(args.workers)
