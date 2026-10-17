"""Amplitude-only search under a forward-displacement-first viability policy."""
from __future__ import annotations
import argparse,concurrent.futures,copy,datetime,hashlib,importlib.util,itertools,json,os,sys,time
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parent
STOPPED=ROOT.parent/'mujoco_amplitude_convergence_2026-09-03'
OLD=ROOT.parent/'mujoco_compliant_relaxed_amplitude_2026-09-02'
RL=ROOT.parents[2]/'Centipede_IsaacLab_RL'
sys.path.insert(0,str(OLD))
import engine as e
import policy as gates
np=e.np
OUT,DATA,TRIALS=ROOT/'outputs',ROOT/'outputs/data',ROOT/'outputs/trials'
e.OUT,e.DATA,e.TRIALS=OUT,DATA,TRIALS;e.backend.TRIALS=TRIALS
KEYS=('body','leg_yaw','leg_pitch');LOW=np.zeros(3);HIGH=np.array([36.,90.,60.])
MANIFEST=DATA/'amplitude_manifest.json';_manifest=None;_sources=None;_rows=None;_references={}
_engine_run_one=e.run_one


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()
def write(path,value):
    path=Path(path);assert path.resolve().is_relative_to(ROOT),path
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');tmp.replace(path)
def event(stage,status,**kwargs):
    row=dict(time=datetime.datetime.now().astimezone().isoformat(),pid=os.getpid(),stage=stage,status=status,**kwargs)
    write(DATA/'live_status.json',row);print(json.dumps(row),flush=True)
def manifest():
    global _manifest
    if _manifest is None:_manifest=read(MANIFEST)
    assert _manifest['gate_policy_sha256']==gates.POLICY_SHA
    return _manifest


def source_rows():
    """Evidence index; incomplete old prefixes are kept but never ranked."""
    result={}
    for name in ('initial','coarse','polish'):
        summary=read(OLD/'outputs/data'/f'{name}_summary.json')
        for row in summary['assessments']:result[row['id']]=row
    for path in sorted((STOPPED/'outputs/assessments').glob('*.json')):
        row=read(path);result[row['id']]=row
    return result


def sources():
    global _sources
    if _sources is None:
        _sources={r['id']:r for r in manifest()['source_records']}
    return _sources


def check_task(t):
    m=manifest();assert t['topology']=='flexible' and t['terrain']=='flat'
    assert t['config']==m['PD_profiles_frozen'][f"{t['wave']:g}"]['config']
    assert t['duration'] in (6.,12.) and t['dt_scale'] in (.5,1.)
    assert set(t['amplitudes'])==set(KEYS)
    x=np.array([t['amplitudes'][k] for k in KEYS]);assert np.all(np.isfinite(x)) and np.all(x>=LOW) and np.all(x<=HIGH)


def task(wave,x,origin,stage='speed_search'):
    base=next(t for t in manifest()['references'] if t['wave']==wave)
    result=dict(base,stage=stage,trace=False,amplitudes=dict(zip(KEYS,map(float,x))),origin=origin)
    check_task(result);return result


def assess(raw):
    result=gates.assess(raw);t=raw['task']
    result.update(id=raw['id'],wave=t['wave'],amplitudes=t['amplitudes'],record=raw['record_path'],
        gate_policy_sha256=gates.POLICY_SHA)
    return result


def physics_run_one(t):
    raw=_engine_run_one(t);raw['old_relaxed_gates']=dict(raw.get('gates',{}))
    now=gates.assess(raw);raw['gates']=now['gates'];raw['absolute_feasible']=now['feasible']
    raw['speed_first_feasible']=now['feasible'];raw['failed']=now['failed'];raw['gate_policy_sha256']=gates.POLICY_SHA
    Path(raw['record_path']).write_text(json.dumps(raw,indent=2),encoding='utf-8');return raw


e.run_one=physics_run_one


def source_raw(tid):
    item=sources().get(tid)
    if not item:return None
    assert sha(item['record'])==item['sha256'],item['record']
    raw=read(item['record']);assert raw['id']==tid and raw['model_sha256']==manifest()['plant_sha256']
    return raw


def screen(t):
    check_task(t);tid=e.task_id(t);cache=OUT/'assessments'/f'{tid}.json'
    if cache.exists():
        row=read(cache);assert row['gate_policy_sha256']==gates.POLICY_SHA;return row
    own=TRIALS/t['stage']/tid/f'{tid}.json';raw=None;kind=None
    if own.exists():
        candidate=read(own)
        if candidate.get('metrics'):raw=candidate;kind='own_full_cache'
    if raw is None:
        candidate=source_raw(tid)
        if candidate and candidate.get('metrics'):raw=candidate;kind='verified_external_full'
    if raw is None:
        raw=e.run_one(t);kind='new_full_cold_rollout'
    row=assess(raw);row.update(evaluation_source=kind,raw_record_sha256=sha(raw['record_path']))
    write(cache,row);return row


def collect(refresh=False):
    global _rows
    if _rows is not None and not refresh:return list(_rows.values())
    rows={}
    for tid in sources():
        raw=source_raw(tid)
        if raw.get('metrics'):rows[tid]=dict(assess(raw),evaluation_source='verified_external_full',raw_record_sha256=sha(raw['record_path']))
    for path in sorted((OUT/'assessments').glob('*.json')):
        row=read(path);assert row['gate_policy_sha256']==gates.POLICY_SHA;rows[row['id']]=row
    _rows=rows;return list(rows.values())


def compact(rows):
    result={}
    for w in e.WAVES:
        group=[r for r in rows if r['wave']==w];ok=sorted((r for r in group if r['feasible']),key=lambda r:(-r['D'],r['id']))
        full=sorted((r for r in group if r['full']),key=lambda r:(len(r['failed']),-r['D']))
        result[f'{w:g}']=dict(full=len(group),feasible=len(ok),best=ok[0] if ok else None,diagnostic=full[0] if full else None)
    return result


def run_bank(tasks,name,workers):
    global _rows
    tasks=deduplicate(tasks);bank=DATA/'banks'/f'{name}.json'
    if bank.exists():assert read(bank)['tasks']==tasks,name
    else:write(bank,dict(name=name,count=len(tasks),tasks=tasks,frozen_before_execution=True))
    done=DATA/'bank_results'/f'{name}.json'
    if done.exists():
        saved=read(done);assert saved['bank_sha256']==sha(bank);return saved['assessments']
    start=time.monotonic();rows=[];event(name,'running',total=len(tasks))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(screen,t):t for t in tasks}
        for i,f in enumerate(concurrent.futures.as_completed(futures),1):
            try:rows.append(f.result())
            except Exception as exc:
                event(name,'failed',completed=len(rows),task=futures[f],error=repr(exc));raise
            if i%20==0 or i==len(tasks):event(name,'running',done=i,total=len(tasks),elapsed_s=time.monotonic()-start)
    rows.sort(key=lambda r:r['id']);write(done,dict(bank_sha256=sha(bank),assessments=rows,waves=compact(rows)))
    _rows=None;event(name,'complete',count=len(rows),elapsed_s=time.monotonic()-start);return rows


def deduplicate(tasks):
    return list({e.task_id(t):t for t in tasks}.values())


def verify_hashes():
    m=manifest();assert sha(m['plant'])==m['plant_sha256']
    for item in m['frozen_files']:assert sha(item['path'])==item['sha256'],item['path']
    for p in m['PD_profiles_frozen'].values():assert sha(p['source'])==p['sha256']
    return True


def freeze():
    if MANIFEST.exists():verify_hashes();return manifest()
    old=read(OLD/'outputs/data/amplitude_manifest.json');rows=source_rows();records=[]
    for tid,row in rows.items():
        assert row['id']==tid and sha(row['record'])==row['raw_record_sha256']
        records.append(dict(id=tid,record=row['record'],sha256=row['raw_record_sha256'],full=bool(row['full'])))
    files=[*sorted(ROOT.glob('*.py')),ROOT/'PROTOCOL.md',
      OLD/'outputs/data/amplitude_manifest.json',OLD/'outputs/data/validated_selection.json',
      STOPPED/'outputs/data/banks/sobol_9302026.json',
      STOPPED/'validate.py',STOPPED/'report.py',
      OLD/'render_compliant_comparison.py',
      RL/'outputs/analysis/p10_lowwave_map_extension_2026-07-22/data/flat_locked_gait_map_v5_1p0_to_4p0_13knot.json',
      RL/'outputs/report/p8_flat_roll_causal_summary_2026-07-21/data/flat_locked_gait_map_v4_13knot.json']
    files += [OLD/'outputs/data'/f'{n}_summary.json' for n in ('initial','coarse','polish')]
    files += [p['source'] for p in old['PD_profiles_frozen'].values()]
    files += [p['path'] for p in old.get('source_files',[])]
    files=list(dict.fromkeys(map(str,files)))
    data={k:copy.deepcopy(old[k]) for k in ('plant','plant_sha256','PD_profiles_frozen','references','frequency_hz','body_leg_wave_coupled','lower_deg','upper_deg','waves')}
    data.update(scope='Centipede Model; flat; compliant local-Z tip; displacement first',
      created=datetime.datetime.now().astimezone().isoformat(),gate_policy=gates.SPEC,
      gate_policy_sha256=gates.POLICY_SHA,source_records=records,
      frozen_files=[dict(path=p,sha256=sha(p)) for p in files],controller_unchanged=True,
      physical_model_unchanged=True,only_three_amplitudes_varied=True,contact_erasure=False,
      canonical_baseline_replaced=False,clean_adoption=False,global_optimum_proved=False,
      second_sobol_seed=39302026,second_sobol_power=7,local_starts=3,
      local_step=.04,local_min_step=.0025,local_max_polls=16,improvement_mm=1e-6,
      body_profile_deg=[],search_note='RL curves are seeds only; no smoothness/monotonicity objective')
    write(MANIFEST,data);return manifest()


def audit():
    verify_hashes();seen=set();full=0;incomplete=0
    for item in manifest()['source_records']:
        assert item['id'] not in seen;seen.add(item['id']);raw=source_raw(item['id']);check_task(raw['task'])
        if raw.get('metrics'):
            row=assess(raw);assert row['D']==raw['metrics']['displacement_mm'];full+=1
        else:incomplete+=1
    result=dict(passed=True,unique_source_conditions=len(seen),full_reclassified=full,
      incomplete_prefixes_not_ranked=incomplete,only_three_amplitudes_vary=True,
      frozen_model_controller_PD=True,robot_step18_19_touched=False)
    write(DATA/'setup_audit.json',result);return result


def cold_check(w):
    old=read(OLD/'outputs/data/validated_selection.json')['waves'][f'{w:g}']['selected']['nominal'];raw=read(old['record'])
    t=dict(raw['task'],stage='verification_cold',trace=False);current=e.run_one(t)
    for k,v in raw['metrics'].items():assert np.array_equal(np.asarray(v),np.asarray(current['metrics'][k]),equal_nan=True),k
    return dict(wave=w,passed=True,exact_all_metrics=True,record=current['record_path'])


def verify(workers):
    path=DATA/'verification.json'
    if path.exists():return read(path)
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:checks=list(pool.map(cold_check,[2.,3.5]))
    result=dict(passed=True,checks=checks,policy_sha256=gates.POLICY_SHA);write(path,result);return result


def first_sobol():
    return [dict(t,stage='speed_search',origin=dict(kind='inherited_sobol_design',source='stopped strict-policy study'))
            for t in read(STOPPED/'outputs/data/banks/sobol_9302026.json')['tasks']]


def second_sobol():
    from scipy.stats import qmc
    tasks=[];seed=manifest()['second_sobol_seed']
    for w in e.WAVES:
        ws=seed+int(2*w);points=qmc.scale(qmc.Sobol(3,scramble=True,seed=ws).random_base2(manifest()['second_sobol_power']),LOW,HIGH)
        tasks += [task(w,x,dict(kind='second_independent_sobol',seed=seed,wave_seed=ws,index=i)) for i,x in enumerate(points)]
    return tasks


def rl_rows():
    p=RL/'outputs/analysis/p10_lowwave_map_extension_2026-07-22/data/flat_locked_gait_map_v5_1p0_to_4p0_13knot.json'
    v5=read(p);rows=list(v5['wave_map'])
    v4=read(RL/'outputs/report/p8_flat_roll_causal_summary_2026-07-21/data/flat_locked_gait_map_v4_13knot.json')
    rows += [r for r in v4['wave_map'] if r['wave']>4]
    return sorted(rows,key=lambda r:r['wave'])


def transfer_bank():
    rows=collect(True);best={w:max((r for r in rows if r['wave']==w and r['feasible']),key=lambda r:r['D'],default=None) for w in e.WAVES}
    incumbents=[r for r in best.values() if r];tasks=[]
    rr=rl_rows();phys=np.array([[np.degrees(.4*r['body_amp']),np.degrees(.6*r['leg_yaw_amp']),np.degrees(.4*1.15)] for r in rr])
    current=np.array([[r['amplitudes'][k] for k in KEYS] for r in incumbents]);scale=np.median(current,axis=0)/np.median(phys,axis=0)
    for w in e.WAVES:
        for r in incumbents:tasks.append(task(w,[r['amplitudes'][k] for k in KEYS],dict(kind='cross_wave_speed_incumbent',source_wave=r['wave'],source_id=r['id'])))
        nearest=min(rr,key=lambda r:abs(r['wave']-w));direct=np.array([np.degrees(.4*nearest['body_amp']),np.degrees(.6*nearest['leg_yaw_amp']),np.degrees(.4*1.15)])
        tasks.append(task(w,np.clip(direct,LOW,HIGH),dict(kind='isaac_command_seed_only',source_wave=nearest['wave'])))
        tasks.append(task(w,np.clip(direct*scale,LOW,HIGH),dict(kind='isaac_shape_scaled_seed_only',source_wave=nearest['wave'])))
        own=best[w]
        if own:
            x=np.array([own['amplitudes'][k] for k in KEYS])
            for dim in range(3):
                for sign in (-1,1):tasks.append(task(w,np.clip(x+sign*.02*HIGH*np.eye(3)[dim],LOW,HIGH),dict(kind='speed_incumbent_axis',source_id=own['id'])))
    return deduplicate(tasks)


def choose_starts():
    path=DATA/'three_d_starts.json'
    if path.exists():return read(path)
    rows=collect(True);starts=[]
    for w in e.WAVES:
        ranked=sorted((r for r in rows if r['wave']==w and r['feasible']),key=lambda r:(-r['D'],r['id']));chosen=[]
        for row in ranked:
            x=np.array([row['amplitudes'][k] for k in KEYS])
            if any(np.linalg.norm((x-np.array([r['amplitudes'][k] for k in KEYS]))/HIGH)<.05 for r in chosen):continue
            chosen.append(row)
            if len(chosen)==manifest()['local_starts']:break
        for i,row in enumerate(chosen):starts.append(dict(key=f'w{w:g}_s{i}',wave=w,center=row,step=manifest()['local_step'],history=[],stop=None))
        if not chosen:starts.append(dict(key=f'w{w:g}_none',wave=w,center=None,step=manifest()['local_step'],history=[],stop='no_feasible_seed_unresolved'))
    write(path,starts);return starts


def refine(workers):
    states=choose_starts();maximum=manifest()['local_max_polls'];minimum=manifest()['local_min_step']
    for poll in range(maximum):
        after=DATA/f'three_d_state_{poll:02d}.json'
        if after.exists():states=read(after);continue
        tasks=[];ids={}
        for st in states:
            if st['stop']:continue
            x=np.array([st['center']['amplitudes'][k] for k in KEYS]);ids[st['key']]=[]
            directions=list(itertools.product((-1.,1.),repeat=3))+[tuple(sign*np.eye(3)[j]) for j in range(3) for sign in (-1,1)]
            for d in directions:
                t=task(st['wave'],np.clip(x+st['step']*HIGH*np.array(d),LOW,HIGH),dict(kind='three_d_poll',poll=poll,start=st['key'],center=st['center']['id'],step=st['step']))
                tasks.append(t);ids[st['key']].append(e.task_id(t))
        if not tasks:break
        got={r['id']:r for r in run_bank(tasks,f'three_d_poll_{poll:02d}',workers)}
        for st in states:
            if st['stop']:continue
            old=st['center'];valid=[got[i] for i in ids[st['key']] if got[i]['feasible']];best=max([old,*valid],key=lambda r:r['D']);gain=best['D']-old['D']
            st['history'].append(dict(poll=poll,step=st['step'],before=old['id'],after=best['id'],gain_mm=gain))
            if gain>manifest()['improvement_mm']:st['center']=best
            elif st['step']<=minimum+1e-12:st['stop']='stationary_at_declared_resolution_not_global_proof'
            else:st['step']=max(minimum,st['step']/2)
        write(after,states)
    for st in states:
        if not st['stop']:st['stop']='poll_budget_reached_unresolved'
    write(DATA/'three_d_convergence.json',states);return states


def validation_record(t):
    tid=e.task_id(t);own=TRIALS/t['stage']/tid/f'{tid}.json'
    if own.exists():return own
    for root in (STOPPED/'outputs/trials',OLD/'outputs/trials'):
        for stage in ('validation','reference','interim_delivery','delivery'):
            path=root/stage/tid/f'{tid}.json'
            if path.exists() and read(path).get('metrics'):return path
    return own
def reference_path(t):
    base=next(x for x in manifest()['references'] if x['wave']==t['wave']);base=dict(base,duration=t['duration'],dt_scale=t['dt_scale']);tid=e.task_id(base)
    for root in (TRIALS,STOPPED/'outputs/trials',OLD/'outputs/trials',e.PARENT/'outputs/trials'):
        path=root/'reference'/tid/f'{tid}.json'
        if path.exists():return path
    return TRIALS/'reference'/tid/f'{tid}.json'


def summary():
    rows=collect(True);result=dict(total_full_distinct=len(rows),waves=compact(rows),gate_policy_sha256=gates.POLICY_SHA,
      global_optimum_proved=False,canonical_baseline_replaced=False);write(DATA/'search_summary.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('freeze','audit','verify','pilot','global1','global2','transfer','local','summary'));p.add_argument('--workers',type=int,default=20);a=p.parse_args();freeze()
    if a.command=='freeze':print(json.dumps(dict(frozen=True,policy=gates.POLICY_SHA)))
    elif a.command=='audit':print(json.dumps(audit()))
    elif a.command=='verify':print(json.dumps(verify(a.workers)))
    elif a.command=='pilot':print(json.dumps(summary()))
    else:
        assert read(DATA/'setup_audit.json')['passed'] and read(DATA/'verification.json')['passed'];verify_hashes()
        if a.command=='global1':run_bank(first_sobol(),'global1_complete',a.workers)
        elif a.command=='global2':run_bank(second_sobol(),'global2_independent',a.workers)
        elif a.command=='transfer':run_bank(transfer_bank(),'transfer',a.workers)
        elif a.command=='local':refine(a.workers)
        print(json.dumps(summary()))
