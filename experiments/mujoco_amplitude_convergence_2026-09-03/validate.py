"""Same nominal/long/half-step hard gates, descending-D candidate selection."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import study as s
e=s.e


def make_task(base,duration,dt,stage='validation'):
    result={k:v for k,v in base.items() if k not in
            ('source_record','source_sha256','reuse_record','reuse_sha256')}
    result.update(stage=stage,duration=duration,dt_scale=dt,trace=False)
    s.check_task(result)
    path=s.reference_path(result) if stage=='reference' else s.validation_record(result)
    if path.exists():
        raw=s.read(path)
        assert e.task_id(raw['task'])==e.task_id(result)
        assert raw['model_sha256']==s.manifest()['plant_sha256']
        result.update(reuse_record=str(path),reuse_sha256=s.sha(path))
    return result


def run_one(t):
    if t.get('reuse_record'):
        assert s.sha(t['reuse_record'])==t['reuse_sha256']
        raw=s.read(t['reuse_record'])
        assert raw.get('metrics')
        return raw
    return e.run_one(t)


def run_tasks(tasks,workers):
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(run_one,t) for t in tasks]
        for i,f in enumerate(concurrent.futures.as_completed(futures),1):
            raw=f.result()
            s.event('validation','running',done=i,total=len(tasks),id=raw['id'])


def validate(label,workers):
    frozen=s.DATA/f'{label}_validation_manifest.json'
    if frozen.exists():m=s.read(frozen)
    else:
        rows=s.collect()
        candidates={f'{w:g}':sorted((r for r in rows if r['wave']==w and r['feasible']),
                    key=lambda r:(-r['D'],r['id'])) for w in e.WAVES}
        references=[]
        for w in e.WAVES:
            if not candidates[f'{w:g}']:continue
            base=next(t for t in s.manifest()['references'] if t['wave']==w)
            references.extend(make_task(base,d,dt,'reference') for d,dt in ((12.,1.),(6.,.5)))
        m=dict(label=label,nominal_candidates=candidates,references=references,
            source_condition_count=len(rows),policy_sha256=s.gates.POLICY_SHA,
            global_optimum_proved=False,selection='descending nominal D; all faster candidates checked')
        s.write(frozen,m)
    run_tasks(m['references'],workers)
    s._references.clear()
    result_path=s.DATA/f'{label}_validated_selection.json'
    if result_path.exists():result=s.read(result_path)
    else:
        result=dict(label=label,validation_complete=False,global_optimum_proved=False,
            canonical_baseline_replaced=False,source_condition_count=m['source_condition_count'],
            policy_sha256=s.gates.POLICY_SHA,
            waves={f'{w:g}':dict(status='pending',selected=None,candidates=[]) for w in e.WAVES})
    round_no=0
    while True:
        jobs=[];pending=[]
        for key,st in result['waves'].items():
            if st['selected']:continue
            n=len(st['candidates']);ranked=m['nominal_candidates'][key]
            if n>=len(ranked):
                st['status']='unresolved';continue
            row=ranked[n];base=s.read(row['record'])['task']
            checks=[make_task(base,d,dt) for d,dt in ((12.,1.),(6.,.5))]
            jobs.extend(checks);pending.append((key,row,checks))
        if not jobs:break
        batch=s.DATA/f'{label}_validation_round_{round_no:03d}.json'
        # Resumable batch index follows completed per-wave candidate counts.
        while batch.exists() and s.read(batch)['tasks']!=jobs:
            round_no+=1;batch=s.DATA/f'{label}_validation_round_{round_no:03d}.json'
        s.write(batch,dict(tasks=jobs,frozen_before_execution=True))
        run_tasks(jobs,workers)
        for key,row,tasks in pending:
            checks=[s.assess(s.read(t.get('reuse_record') or s.validation_record(t))) for t in tasks]
            difference=abs(checks[1]['D']-row['D'])/max(abs(row['D']),1e-9)
            passing=all(r['feasible'] for r in checks) and difference<=.1
            item=dict(nominal=row,checks=checks,halfstep_D_relative_difference=difference,qualified=passing)
            st=result['waves'][key];st['candidates'].append(item)
            if passing:st.update(selected=item,status='qualified')
        s.write(result_path,result);round_no+=1
    all_rows=s.collect();trace_tasks=[]
    for key,st in result['waves'].items():
        full=sorted((r for r in all_rows if r['wave']==float(key) and r['full']),key=lambda r:(len(r['failed']),-r['D']))
        st['diagnostic']=None if st['selected'] else full[0] if full else None
        row=st['selected']['nominal'] if st['selected'] else st['diagnostic']
        if row:
            t=dict(s.read(row['record'])['task'],stage=label+'_delivery',trace=True)
            trace_tasks.append(t)
    result.update(validation_complete=True,traces=trace_tasks)
    s.write(result_path,result)
    # Trace cold replays are checked against the ranked metrics, not merely rendered.
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        raws=list(pool.map(e.run_one,trace_tasks))
    for raw in raws:
        st=result['waves'][f"{raw['task']['wave']:g}"]
        row=st['selected']['nominal'] if st['selected'] else st['diagnostic']
        for key,value in row['metrics'].items():
            assert s.np.array_equal(s.np.asarray(value),s.np.asarray(raw['metrics'][key]),equal_nan=True),key
        st['trace_record']=raw['record_path']
        st['trace_metrics_exact']=True
    s.write(result_path,result);s.verify_hashes()
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label',default='final',choices=('interim','final'))
    p.add_argument('--workers',type=int,default=20);a=p.parse_args()
    s.freeze();assert s.read(s.DATA/'verification.json')['passed']
    print(json.dumps({k:v['status'] for k,v in validate(a.label,a.workers)['waves'].items()}))
