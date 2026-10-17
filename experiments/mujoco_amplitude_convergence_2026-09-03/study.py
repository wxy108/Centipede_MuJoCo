"""Frozen-physics, reproducible amplitude search. No controller/model edits."""
from __future__ import annotations
import argparse
import concurrent.futures
import copy
import datetime
import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys
import time

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'mujoco_compliant_relaxed_amplitude_2026-09-02'
sys.path.insert(0, str(OLD))
import engine as e
import gates
np = e.np
OUT, DATA, TRIALS = ROOT/'outputs', ROOT/'outputs/data', ROOT/'outputs/trials'
# Redirect output namespaces only. The actual evaluator is imported unchanged.
e.OUT, e.DATA, e.TRIALS = OUT, DATA, TRIALS
e.backend.TRIALS = TRIALS
spec = importlib.util.spec_from_file_location('convergence_frozen_study', OLD/'study.py')
prior = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = prior
spec.loader.exec_module(prior)
sys.path.insert(0, str(ROOT))
KEYS = ('body', 'leg_yaw', 'leg_pitch')
LOW, HIGH = np.zeros(3), np.array([36.,90.,60.])
MANIFEST = DATA/'amplitude_manifest.json'
_manifest = None
_old_rows = None
_references = {}
read, sha = prior.read, prior.sha


def write(path, value):
    path = Path(path)
    assert path.resolve().is_relative_to(ROOT), path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def manifest():
    global _manifest
    if _manifest is None:
        _manifest = read(MANIFEST)
    assert _manifest['gate_policy_sha256'] == gates.POLICY_SHA
    return _manifest


def event(stage, status, **kwargs):
    item = dict(time=datetime.datetime.now().astimezone().isoformat(),
                pid=os.getpid(), stage=stage, status=status, **kwargs)
    write(DATA/'live_status.json', item)
    print(json.dumps(item), flush=True)


def old_rows():
    global _old_rows
    if _old_rows is None:
        _old_rows = {}
        for name in ('initial', 'coarse', 'polish'):
            for row in read(OLD/'outputs/data'/f'{name}_summary.json')['assessments']:
                assert row['id'] not in _old_rows
                _old_rows[row['id']] = row
        assert len(_old_rows) == 5036
    return _old_rows


def reference_path(task):
    base = next(t for t in manifest()['references'] if t['wave']==task['wave'])
    base = dict(base, duration=task['duration'], dt_scale=task['dt_scale'])
    tid = e.task_id(base)
    for root in (TRIALS, OLD/'outputs/trials', e.PARENT/'outputs/trials'):
        path = root/'reference'/tid/f'{tid}.json'
        if path.exists():
            return path
    return TRIALS/'reference'/tid/f'{tid}.json'


def load_reference(task):
    path = reference_path(task)
    if str(path) not in _references:
        raw = read(path)
        assert raw['model_sha256'] == manifest()['plant_sha256']
        _references[str(path)] = raw
    return _references[str(path)], path


prior.manifest = manifest
prior.reference_path = reference_path
prior.load_reference = load_reference
prior.write = write


def assess(raw):
    return prior.assess(raw)


def check_task(task):
    m = manifest()
    assert task['topology']=='flexible' and task['terrain']=='flat'
    assert task['config']==m['PD_profiles_frozen'][f"{task['wave']:g}"]['config']
    x = np.array([task['amplitudes'][k] for k in KEYS])
    assert np.all(np.isfinite(x)) and np.all(x>=LOW) and np.all(x<=HIGH)
    assert task['duration'] in (6.,12.) and task['dt_scale'] in (.5,1.)
    assert set(task['amplitudes'])==set(KEYS)


def task(wave, x, origin, stage='search'):
    base = next(t for t in manifest()['references'] if t['wave']==wave)
    result = dict(base, stage=stage, trace=False,
                  amplitudes=dict(zip(KEYS, map(float,x))), origin=origin)
    check_task(result)
    return result


def screen(t):
    check_task(t)
    tid = e.task_id(t)
    if tid in old_rows():
        row = old_rows()[tid]
        assert sha(row['record']) == row['raw_record_sha256']
        current = assess(read(row['record']))
        assert current['feasible']==row['feasible'] and current['D']==row['D']
        return dict(row, evaluation_source='verified_previous_round')
    row = prior.screen(t)
    assert Path(row['record']).is_relative_to(ROOT)
    return row


def collect():
    rows = dict(old_rows())
    for path in sorted((OUT/'assessments').glob('*.json')):
        row = read(path)
        assert row['gate_policy_sha256']==gates.POLICY_SHA
        rows[row['id']] = row
    return list(rows.values())


def validation_record(t):
    tid=e.task_id(t)
    own=TRIALS/t['stage']/tid/f'{tid}.json'
    if own.exists():return own
    for root in (OLD/'outputs/trials',e.PARENT/'outputs/trials'):
        for stage in ('validation','reference','delivery','search'):
            path=root/stage/tid/f'{tid}.json'
            if path.exists() and read(path).get('metrics'):return path
    return own


def compact(rows):
    result = {}
    for w in e.WAVES:
        group = [r for r in rows if r['wave']==w]
        passing = sorted((r for r in group if r['feasible']), key=lambda r:(-r['D'],r['id']))
        result[f'{w:g}'] = dict(count=len(group), feasible=len(passing),
            best=passing[0] if passing else None)
    return result


def run_bank(tasks, name, workers):
    path = DATA/'banks'/f'{name}.json'
    if path.exists():
        saved = read(path)
        assert saved['tasks']==tasks, name
    else:
        write(path, dict(name=name, tasks=tasks, count=len(tasks), frozen_before_execution=True))
    done_path = DATA/'bank_results'/f'{name}.json'
    if done_path.exists():
        data=read(done_path)
        assert data['bank_sha256']==sha(path)
        return data['assessments']
    start=time.monotonic()
    rows=[]
    event(name, 'running', total=len(tasks))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(screen,t):t for t in tasks}
        for i,f in enumerate(concurrent.futures.as_completed(futures),1):
            try:
                row=f.result()
            except Exception as exc:
                event(name,'failed',task=futures[f],error=repr(exc),completed=len(rows))
                raise
            rows.append(row)
            if i%20==0 or i==len(tasks):
                event(name,'running',done=i,total=len(tasks),elapsed_s=time.monotonic()-start)
    rows.sort(key=lambda r:r['id'])
    write(done_path,dict(bank_sha256=sha(path),assessments=rows,waves=compact(rows)))
    event(name,'complete',count=len(rows),elapsed_s=time.monotonic()-start)
    return rows


def verify_hashes():
    m=manifest()
    for item in m['frozen_files']:
        assert sha(item['path'])==item['sha256'], item['path']
    assert sha(m['plant'])==m['plant_sha256']
    for profile in m['PD_profiles_frozen'].values():
        assert sha(profile['source'])==profile['sha256']
    return True


def freeze():
    if MANIFEST.exists():
        verify_hashes()
        return manifest()
    old=read(OLD/'outputs/data/amplitude_manifest.json')
    assert read(OLD/'outputs/data/final_delivery_audit.json')['passed']
    assets=read(OLD/'outputs/data/reporting_and_asset_sources.json')
    sources=old['source_files']+assets['assets']
    sources += [dict(path=str(OLD/name),sha256=sha(OLD/name)) for name in
        ('study.py','engine.py','gates.py','validate.py','render_compliant_comparison.py')]
    sources += [dict(path=str(OLD/'outputs/data'/name),sha256=sha(OLD/'outputs/data'/name))
                for name in ('amplitude_manifest.json','validated_selection.json',
                             'initial_summary.json','coarse_summary.json','polish_summary.json')]
    sources += [dict(path=str(p),sha256=sha(p)) for p in sorted(ROOT.glob('*.py'))]
    sources += [dict(path=str(ROOT/'PROTOCOL.md'),sha256=sha(ROOT/'PROTOCOL.md'))]
    unique={v['path']:v for v in sources}
    for source in unique.values():
        assert sha(source['path'])==source['sha256']
    data={k:copy.deepcopy(old[k]) for k in ('plant','plant_sha256','PD_profiles_frozen',
         'references','gate_policy','gate_policy_sha256','frequency_hz','body_leg_wave_coupled',
         'lower_deg','upper_deg','waves')}
    data.update(scope='Centipede Model only; frozen compliant local-Z tip, flat, erasure OFF',
        created=datetime.datetime.now().astimezone().isoformat(),frozen_files=list(unique.values()),
        previous_condition_count=5036,controller_unchanged=True,global_optimum_proved=False,
        canonical_baseline_replaced=False,clean_adoption=False,
        sobol_seeds=[9302026,19302026],sobol_power=8,body_profile_deg=list(range(0,37,3)),
        profile_sobol_power=3,three_d_starts=3,three_d_max_polls=16,three_d_min_step=.0025,
        profile_max_polls=8,profile_min_step=.01,initial_step=.04,improvement_mm=1e-6)
    write(MANIFEST,data)
    return manifest()


def audit():
    verify_hashes()
    count=0
    for row in old_rows().values():
        assert sha(row['record'])==row['raw_record_sha256'],row['record']
        raw=read(row['record'])
        check_task(raw['task'])
        current=assess(raw)
        assert current['D']==row['D'] and current['feasible']==row['feasible']
        assert set(current['failed'])==set(row['failed'])
        count+=1
    result=dict(passed=True,old_conditions_verified=count,
        frozen_hashes_verified=True,gates_exact=True,only_three_amplitudes_varied=True)
    write(DATA/'setup_audit.json',result)
    return result


def cold_check(w):
    old=read(OLD/'outputs/data/validated_selection.json')['waves'][f'{w:g}']['selected']['nominal']
    raw=read(old['record'])
    t=dict(raw['task'],stage='verification_cold',trace=False)
    current=e.run_one(t)
    for k,value in raw['metrics'].items():
        assert np.array_equal(np.asarray(value),np.asarray(current['metrics'][k]),equal_nan=True),k
    assert assess(current)['feasible']
    prefix=prior.prefix(dict(t,stage='verification_prefix'))
    assert prefix is None, prefix
    return dict(wave=w,passed=True,exact_all_metrics=True,record=current['record_path'])


def verify(workers):
    path=DATA/'verification.json'
    if path.exists():
        assert read(path)['passed']
        return read(path)
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(workers,2)) as pool:
        checks=list(pool.map(cold_check,[2.,3.5]))
    result=dict(passed=True,checks=checks,policy_sha256=gates.POLICY_SHA)
    write(path,result)
    return result


def deduplicate(tasks):
    result={}
    for t in tasks:
        result.setdefault(e.task_id(t),t)
    return list(result.values())


def incumbents():
    return [r['selected']['nominal'] for r in read(OLD/'outputs/data/validated_selection.json')['waves'].values()
            if r['selected']]


def transfer_bank():
    tasks=[]
    anchors=incumbents()
    for w in e.WAVES:
        for r in anchors:
            tasks.append(task(w,[r['amplitudes'][k] for k in KEYS],
                dict(kind='cross_wave_incumbent',source_wave=r['wave'],source_id=r['id'])))
        own=next((r for r in anchors if r['wave']==w),None)
        if own:
            x=np.array([own['amplitudes'][k] for k in KEYS])
            for dim in range(3):
                for sign in (-1,1):
                    y=np.clip(x+sign*.02*HIGH*np.eye(3)[dim],LOW,HIGH)
                    tasks.append(task(w,y,dict(kind='incumbent_axis_challenge',source_id=own['id'])))
    return deduplicate(tasks)


def sobol_bank(seed):
    from scipy.stats import qmc
    tasks=[]
    for w in e.WAVES:
        wave_seed=seed+int(w*2)
        x=qmc.scale(qmc.Sobol(3,scramble=True,seed=wave_seed).random_base2(manifest()['sobol_power']),LOW,HIGH)
        for i,point in enumerate(x):
            tasks.append(task(w,point,dict(kind='independent_sobol',seed=seed,wave_seed=wave_seed,index=i)))
    return tasks


def profile_bank():
    from scipy.stats import qmc
    saved=DATA/'banks/profile_seed.json'
    if saved.exists():
        return read(saved)['tasks']
    rows=collect();tasks=[];anchors=incumbents()
    for w in e.WAVES:
        feasible=sorted((r for r in rows if r['wave']==w and r['feasible']),key=lambda r:-r['D'])
        neighbors=sorted(anchors,key=lambda r:abs(r['wave']-w))[:3]
        for node in manifest()['body_profile_deg']:
            seed=29302026+int(w*100)+node
            points=qmc.scale(qmc.Sobol(2,scramble=True,seed=seed).random_base2(manifest()['profile_sobol_power']),LOW[1:],HIGH[1:]).tolist()
            seed_rows=list(neighbors)
            if feasible:
                seed_rows += [feasible[0],min(feasible,key=lambda r:abs(r['amplitudes']['body']-node))]
            points += [[r['amplitudes'][k] for k in KEYS[1:]] for r in seed_rows]
            for i,legs in enumerate(points):
                tasks.append(task(w,[node,*legs],dict(kind='profile_seed',node=node,seed=seed,index=i)))
    return deduplicate(tasks)


def choose_starts(kind):
    path=DATA/f'{kind}_starts.json'
    if path.exists():
        return read(path)
    rows=collect();starts=[]
    for w in e.WAVES:
        feasible=sorted((r for r in rows if r['wave']==w and r['feasible']),key=lambda r:(-r['D'],r['id']))
        if kind=='profile':
            for node in manifest()['body_profile_deg']:
                matching=[r for r in feasible if abs(r['amplitudes']['body']-node)<1e-10]
                starts.append(dict(key=f'w{w:g}_body{node}',wave=w,node=node,
                    center=matching[0] if matching else None,step=.04,history=[],
                    stop=None if matching else 'no_feasible_seed_unresolved'))
        else:
            chosen=[]
            # Prefer independent-bank bests if spatially distinct, then other basins.
            bank_bests=[]
            for seed in manifest()['sobol_seeds']:
                bank=read(DATA/'bank_results'/f'sobol_{seed}.json')['assessments']
                group=sorted((r for r in bank if r['wave']==w and r['feasible']),key=lambda r:-r['D'])
                if group:bank_bests.append(group[0])
            priority=feasible[:1]+bank_bests+feasible[1:]
            for row in priority:
                x=np.array([row['amplitudes'][k] for k in KEYS])
                if any(np.linalg.norm((x-np.array([r['amplitudes'][k] for k in KEYS]))/HIGH)<.05 for r in chosen):
                    continue
                chosen.append(row)
                if len(chosen)==manifest()['three_d_starts']:break
            for i,row in enumerate(chosen):
                starts.append(dict(key=f'w{w:g}_start{i}',wave=w,center=row,step=.04,history=[],stop=None))
            if not chosen:
                starts.append(dict(key=f'w{w:g}_none',wave=w,center=None,step=.04,history=[],stop='no_feasible_seed_unresolved'))
    write(path,starts)
    return starts


def refine(kind,workers):
    states=choose_starts(kind)
    dims=(1,2) if kind=='profile' else (0,1,2)
    maxpoll=manifest()['profile_max_polls' if kind=='profile' else 'three_d_max_polls']
    minimum=manifest()['profile_min_step' if kind=='profile' else 'three_d_min_step']
    for poll in range(maxpoll):
        after=DATA/f'{kind}_state_{poll:02d}.json'
        if after.exists():
            states=read(after);continue
        tasks=[];ids={}
        for st in states:
            if st['stop']:continue
            x=np.array([st['center']['amplitudes'][k] for k in KEYS])
            directions=list(itertools.product((-1.,1.),repeat=len(dims)))
            directions += [tuple(sign*np.eye(len(dims))[j]) for j in range(len(dims)) for sign in (-1,1)]
            ids[st['key']]=[]
            for direction in directions:
                y=x.copy();y[list(dims)]+=st['step']*HIGH[list(dims)]*direction
                y=np.clip(y,LOW,HIGH)
                t=task(st['wave'],y,dict(kind=kind+'_poll',poll=poll,start=st['key'],center=st['center']['id'],step=st['step']))
                tasks.append(t);ids[st['key']].append(e.task_id(t))
        if not tasks:break
        result=run_bank(deduplicate(tasks),f'{kind}_poll_{poll:02d}',workers)
        lookup={r['id']:r for r in result}
        for st in states:
            if st['stop']:continue
            old=st['center'];candidates=[lookup[tid] for tid in ids[st['key']] if lookup[tid]['feasible']]
            best=max([old,*candidates],key=lambda r:r['D'])
            improvement=best['D']-old['D']
            entry=dict(poll=poll,step=st['step'],before_id=old['id'],before_D=old['D'],after_id=best['id'],after_D=best['D'],improvement_mm=improvement)
            if improvement>manifest()['improvement_mm']:
                st['center']=best
            elif st['step']<=minimum+1e-12:
                st['stop']='poll_stationary_at_declared_resolution_not_global_proof'
            else:
                st['step']=max(minimum,st['step']/2)
            st['history'].append(entry)
        write(after,states)
    for st in states:
        if not st['stop']:st['stop']='poll_budget_reached_unresolved'
    write(DATA/f'{kind}_convergence.json',states)
    return states


def summary():
    rows=collect()
    result=dict(total_distinct=len(rows),old_conditions=5036,new_conditions=len(rows)-5036,
        waves=compact(rows),global_optimum_proved=False,validation_pending=True)
    write(DATA/'search_summary.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('freeze','audit','verify','transfer','global','profile','local','summary'))
    p.add_argument('--workers',type=int,default=20);args=p.parse_args()
    freeze()
    if args.command=='freeze':print(json.dumps(dict(frozen=True,plant=manifest()['plant'],gate_sha=gates.POLICY_SHA)))
    elif args.command=='audit':print(json.dumps(audit()))
    elif args.command=='verify':print(json.dumps(verify(args.workers)))
    elif args.command=='summary':print(json.dumps(summary()))
    else:
        assert read(DATA/'setup_audit.json')['passed'] and read(DATA/'verification.json')['passed']
        verify_hashes()
        if args.command=='transfer':run_bank(transfer_bank(),'transfer',args.workers)
        elif args.command=='global':
            for seed in manifest()['sobol_seeds']:run_bank(sobol_bank(seed),f'sobol_{seed}',args.workers)
        elif args.command=='profile':
            run_bank(profile_bank(),'profile_seed',args.workers);refine('profile',args.workers)
        elif args.command=='local':refine('three_d',args.workers)
        summary();verify_hashes()
