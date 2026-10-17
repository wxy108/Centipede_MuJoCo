"""Independent cold full-run finalists and matching reference conditions."""
import argparse
import json
import refine as ref
s,e=ref.s,ref.e


def freeze(top=3):
    path=e.DATA/'validation_manifest.json'
    if path.exists():return json.loads(path.read_text())
    rows=ref.collect(['broad','refine','polish','boundary']);manifest=json.loads(s.MANIFEST.read_text())
    tasks=[];references=[];selected={};all_candidates={}
    for wave in e.WAVES:
        ranked_all=sorted([r for r in rows if r['wave']==wave and r['feasible']],key=lambda r:(-r['D'],sum(r['amplitudes'].values())))
        all_candidates[f'{wave:g}']=ranked_all
        ranked=ranked_all[:top]
        selected[f'{wave:g}']=ranked
        if not ranked:continue
        base=next(t for t in manifest['references'] if t['wave']==wave)
        references.extend(dict(base,duration=duration,dt_scale=dt,trace=False) for duration,dt in ((12.,1.),(6.,.5)))
        for row in ranked:
            task=json.loads(open(row['record'],encoding='utf-8').read())['task']
            tasks.extend(dict(task,stage='validation',duration=duration,dt_scale=dt,trace=False) for duration,dt in ((12.,1.),(6.,.5)))
    data=dict(selection='descending nominal displacement; validate batches of three until one qualifies or all nominal feasible candidates are exhausted',
        top_per_wave=top,candidates=selected,all_nominal_candidates=all_candidates,tasks=tasks,references=references,
        dt_displacement_relative_tolerance=.10,global_optimum_proved=False)
    path.write_text(json.dumps(data,indent=2));return data


def evaluate(manifest):
    waves={};traces=[];allrows=ref.collect(['broad','refine','polish','boundary'])
    for wave in e.WAVES:
        candidates=manifest['candidates'][f'{wave:g}'];results=[]
        for row in candidates:
            base=json.loads(open(row['record'],encoding='utf-8').read())['task'];checks=[]
            for duration,dt in ((12.,1.),(6.,.5)):
                task=dict(base,stage='validation',duration=duration,dt_scale=dt,trace=False)
                tid=e.task_id(task);path=e.TRIALS/'validation'/tid/f'{tid}.json'
                checks.append(s.assess(json.loads(path.read_text())))
            half=next(r for r in checks if json.loads(open(r['record'],encoding='utf-8').read())['task']['dt_scale']==.5)
            delta=abs(half['D']-row['D'])/max(abs(row['D']),1e-9)
            passed=all(r['feasible'] for r in checks) and delta<=manifest['dt_displacement_relative_tolerance']
            results.append(dict(nominal=row,checks=checks,halfstep_D_relative_difference=delta,qualified=passed))
        passed=sorted([r for r in results if r['qualified']],key=lambda r:-r['nominal']['D'])
        chosen=passed[0] if passed else None
        diagnostic=None
        if chosen is None:
            full=[r for r in allrows if r['wave']==wave and r['full']]
            ranked=sorted(full,key=lambda r:(len(r['failed']),-r['D']))
            diagnostic=ranked[0] if ranked else None
        waves[f'{wave:g}']=dict(status='qualified' if chosen else 'unresolved',selected=chosen,candidates=results,diagnostic=diagnostic)
        nominal=chosen['nominal'] if chosen else diagnostic
        if nominal:
            task=json.loads(open(nominal['record'],encoding='utf-8').read())['task']
            traces.append(dict(task,stage='delivery' if chosen else 'delivery_diagnostic',trace=True))
            base=next(t for t in json.loads(s.MANIFEST.read_text())['references'] if t['wave']==wave)
            traces.append(dict(base,stage='delivery_reference',trace=True))
    result=dict(scope='amplitude-only on flexible primary model; rigid remains control',waves=waves,validation_complete=False,
        global_optimum_proved=False,baseline_replaced=False,traces=traces,
        meaning='highest nominal displacement in the registered evaluated bank passing nominal, 12 s, half timestep and timestep displacement agreement; higher ranked nominal feasible candidates have been checked first')
    (e.DATA/'validated_selection.json').write_text(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=12);args=p.parse_args()
    import subprocess
    import sys
    print(json.dumps(dict(stage='bounded coupled boundary search',status='running before finalist selection')),flush=True)
    subprocess.run([sys.executable,str(e.ROOT/'boundary.py'),'--workers',str(args.workers)],cwd=e.ROOT,check=True)
    m=freeze();print(json.dumps(dict(finalist_tasks=len(m['tasks']),reference_tasks=len(m['references']))),flush=True)
    e.run_tasks(m['references'],args.workers)
    e.run_tasks(m['tasks'],args.workers)
    result=evaluate(m)
    round_number=1
    while True:
        tasks=[];added={}
        for wave in e.WAVES:
            key=f'{wave:g}'
            if result['waves'][key]['status']=='qualified':continue
            cursor=len(m['candidates'][key]);batch=m['all_nominal_candidates'][key][cursor:cursor+m['top_per_wave']]
            if not batch:continue
            added[key]=[r['id'] for r in batch];m['candidates'][key].extend(batch)
            for row in batch:
                base=json.loads(open(row['record'],encoding='utf-8').read())['task']
                tasks.extend(dict(base,stage='validation',duration=duration,dt_scale=dt,trace=False) for duration,dt in ((12.,1.),(6.,.5)))
        if not tasks:break
        round_number+=1
        m['tasks'].extend(tasks)
        (e.DATA/f'validation_round_{round_number:02d}.json').write_text(json.dumps(dict(round=round_number,added=added,tasks=tasks),indent=2))
        print(json.dumps(dict(validation_round=round_number,added=added,tasks=len(tasks))),flush=True)
        e.run_tasks(tasks,args.workers);result=evaluate(m)
    (e.DATA/'validation_effective_manifest.json').write_text(json.dumps(m,indent=2))
    result['validation_complete']=True
    (e.DATA/'validated_selection.json').write_text(json.dumps(result,indent=2))
    e.run_tasks(result['traces'],args.workers)
    print(json.dumps({w:r['status'] for w,r in result['waves'].items()}),flush=True)
