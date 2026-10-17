from __future__ import annotations
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import concurrent.futures,datetime,hashlib,importlib.util,json,sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs';DATA=OUT/'data';TRIALS=OUT/'trials'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def write(p,v):
    p=Path(p);assert p.resolve().is_relative_to(ROOT);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+f'.{os.getpid()}.tmp');temp.write_text(json.dumps(v,indent=2),encoding='utf-8')
    # Windows readers or scanners can temporarily deny replacement of an open
    # destination. Preserve the atomic write and retry; never truncate it.
    deadline=time.monotonic()+5.
    while True:
        try:temp.replace(p);break
        except PermissionError:
            if time.monotonic()>=deadline:raise
            time.sleep(.05)
def load(p,name):
    sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
EXPERIMENTS=ROOT.parent.parent
e=load(EXPERIMENTS/'mujoco_rigid_control_amplitude_2026-09-02/engine.py','recovery_physics_engine')
policy=load(EXPERIMENTS/'mujoco_displacement_first_amplitude_2026-09-03/policy.py','recovery_policy')
e.TRIALS=TRIALS;np=e.np;mujoco=e.mujoco
KEYS=['body','leg_yaw','leg_pitch'];HIGH=np.array([36.,90.,60.]);WAVES=list(e.WAVES)
_history=None
_guard_installed=False
def history():
    global _history
    if _history is None:_history=read(ROOT.parent/'outputs/data/history.json')
    return _history
def task(w,x,stage,duration=6.,scale=.5,trace=False):
    x=np.asarray(x,float);assert x.shape==(3,) and np.all(x>=0) and np.all(x<=HIGH)
    return dict(topology='flexible',terrain='flat',wave=float(w),config=history()['profiles'][f'{w:g}']['config'],
        amplitudes=dict(zip(KEYS,map(float,x))),duration=float(duration),dt_scale=float(scale),trace=trace,stage=stage)
def identity(t):return e.task_id(t)
def run(t):
    global _guard_installed
    key=identity(t);path=OUT/'assessments'/f'{key}.json'
    if path.exists() and not t['trace']:return read(path)
    assert sha(history()['plant'])==history()['plant_sha256']
    if t['stage']!='warm' and (DATA/'screening_verification.json').exists() and not _guard_installed:
        import screening
        check=read(DATA/'screening_verification.json');assert check['passed'] and sha(ROOT/'screening.py')==check['screening_sha256']
        screening.install(True);_guard_installed=True
    try:
        r=e.run_one(t);a=policy.assess(r)
        row=dict(id=key,task=t,wave=t['wave'],amplitudes=t['amplitudes'],D=a['D'],full=a['full'],feasible=a['feasible'],
            failed=a['failed'],metrics=r['metrics'],record=r['record_path'],raw_sha256=sha(r['record_path']),elapsed_s=r['elapsed_s'])
    except Exception as ex:
        if hasattr(ex,'proof'):
            folder=TRIALS/t['stage']/key;folder.mkdir(parents=True,exist_ok=True);rawpath=folder/'prefix_rejection.json'
            np.savez_compressed(folder/'prefix_state.npz',**ex.state)
            write(rawpath,dict(id=key,task=t,proof=ex.proof,model_sha256=history()['plant_sha256'],metrics=None))
            row=dict(id=key,task=t,wave=t['wave'],amplitudes=t['amplitudes'],D=None,full=False,feasible=False,
                failed=[v['gate'].replace('upright_deficit','upright') for v in ex.proof['violations']],proven_infeasible=True,proof=ex.proof,record=str(rawpath),raw_sha256=sha(rawpath))
        else:row=dict(id=key,task=t,wave=t['wave'],amplitudes=t['amplitudes'],D=None,full=False,feasible=False,failed=['execution_error'],error=repr(ex))
    write(path,row);return row
def records():return [read(p) for p in sorted((OUT/'assessments').glob('*.json'))]
def nominal():return [r for r in records() if r['task']['duration']==6 and r['task']['dt_scale']==.5]
def bank(tasks,name,workers=12):
    for p in read(DATA/'manifest.json')['frozen_inputs']:assert sha(p['path'])==p['sha256'],p['path']
    tasks=list({identity(t):t for t in tasks}.values());path=DATA/'banks'/f'{name}.json'
    if path.exists():assert read(path)==tasks
    else:write(path,tasks)
    version_key=hashlib.sha256(b''.join(p.read_bytes() for p in sorted(ROOT.glob('*.py')))).hexdigest()[:12]
    version=DATA/'banks'/f'{name}_sources_{version_key}.json'
    archive=OUT/'audit/source_versions'/version_key
    archive.mkdir(parents=True,exist_ok=True)
    for source in ROOT.glob('*.py'):
        target=archive/source.name
        if not target.exists():shutil.copy2(source,target)
        assert sha(source)==sha(target)
    if not version.exists():write(version,dict(files=[dict(path=str(p),sha256=sha(p)) for p in sorted(ROOT.glob('*.py'))]))
    done=DATA/'completed'/f'{name}.json'
    if done.exists():return [read(OUT/'assessments'/f'{i}.json') for i in read(done)['ids']]
    start=time.monotonic();rows=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        # Interleave waves so an interrupted batch has broad, balanced coverage.
        groups={w:[t for t in tasks if t['wave']==w] for w in WAVES};ordered=[]
        for j in range(max([len(v) for v in groups.values()],default=0)):
            ordered.extend(groups[w][j] for w in WAVES if j<len(groups[w]))
        pending={};it=iter(ordered)
        def fill():
            while len(pending)<workers*2:
                t=next(it,None)
                if t is None:break
                pending[pool.submit(run,t)]=t
        fill()
        while pending:
            complete,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
            for f in complete:
                t=pending.pop(f);r=f.result();rows.append(r)
                status=dict(time=datetime.datetime.now().astimezone().isoformat(),stage=name,done=len(rows),total=len(tasks),id=r['id'],D=r['D'],feasible=r['feasible'],elapsed_s=time.monotonic()-start)
                write(DATA/'live_status.json',status)
                if len(rows)%10==0 or len(rows)==len(tasks):print(json.dumps(status),flush=True)
            fill()
    write(done,dict(ids=[r['id'] for r in rows],count=len(rows),full=sum(r['full'] for r in rows),feasible=sum(r['feasible'] for r in rows)))
    return rows
