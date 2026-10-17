from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import concurrent.futures, datetime, hashlib, importlib.util, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
EXP=ROOT.parent
DATA=ROOT/'outputs/data'
OUT=ROOT/'outputs'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()
def write(path,value):
    path=Path(path);assert path.resolve().is_relative_to(ROOT)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2),encoding='utf-8')
    deadline=time.monotonic()+5
    while True:
        try:tmp.replace(path);break
        except PermissionError:
            if time.monotonic()>deadline:raise
            time.sleep(.05)
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

e=load(EXP/'mujoco_rigid_control_amplitude_2026-09-02/engine.py','common_pd_engine')
policy=load(EXP/'mujoco_displacement_first_amplitude_2026-09-03/policy.py','common_pd_policy')
np,mujoco=e.np,e.mujoco
e.TRIALS=OUT/'trials'
WAVES=list(e.WAVES);HIGH=np.array([36.,90.,60.]);KEYS=['body','leg_yaw','leg_pitch']
CFG=dict(next(c for c in e.fx.pilot.BANK if c['name']=='p4_leg_k2_yaw_d2_roll_d4'),yaw_k=1.)
PLANTS={
    'current':EXP/'mujoco_tip_passive_gliding_2026-09-01/models/centipede_flat_localz_k2em05_d3p2_r38.xml',
    'august31':EXP/'mujoco_template_locked_head_2026-08-27/models/centipede_flat_locked_head_m29p0.xml',
}
ORIGINAL_CONTROLLER=e.fx.opt.ThreeAmplitudeController
ORIGINAL_APPLY=e.fx.pilot.apply_impedance
ORIGINAL_NEAR=e.fx.near.near_ground
_installed=False
_active_task=None
_pd_readback=None

def task(w,x,stage,*,plant='current',phase='legacy',duration=6.,scale=.5,trace=False,prefix=True,initialize='near'):
    x=np.asarray(x,float)
    assert x.shape==(3,) and np.all(x>=0) and np.all(x<=HIGH)
    assert phase in ('legacy','midpoint') and plant in PLANTS
    return dict(topology='flexible',terrain='flat',wave=float(w),amplitudes=dict(zip(KEYS,map(float,x))),
        config=dict(CFG),plant=plant,phase=phase,duration=float(duration),dt_scale=float(scale),
        trace=trace,prefix=prefix,initialize=initialize,stage=stage)
def identity(t):
    physical={k:v for k,v in t.items() if k not in ('stage','trace','prefix')}
    physical['config']={k:v for k,v in physical['config'].items() if k!='name'}
    digest=hashlib.sha256(json.dumps(physical,sort_keys=True).encode()).hexdigest()[:16]
    return f"{t['plant']}_{t['phase']}_w{t['wave']:g}_{digest}".replace('.','p')

class MidpointController(ORIGINAL_CONTROLLER):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.leg_offsets=self.leg_offsets + .5*(2*np.pi*self.wave/18.)

def checked_apply(model,index,cfg):
    global _pd_readback
    assert cfg==CFG, 'Wave-specific or tuned PD forbidden'
    ORIGINAL_APPLY(model,index,cfg)
    payload=dict(names=index.active_joint_names,kp=index.kp.tolist(),kv=index.kv.tolist())
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest().upper()
    expected=read(DATA/'manifest.json')['active_pd_sha256']
    assert digest==expected,(digest,expected)
    _pd_readback=dict(active_pd_sha256=digest,config=dict(cfg),
       body_roll_k=model.jnt_stiffness[index.roll_joint_ids].tolist(),
       body_roll_d=model.dof_damping[index.roll_dof_addresses].tolist(),
       tip_k=model.jnt_stiffness[index.tip_joint_ids].tolist(),
       tip_d=model.dof_damping[index.tip_dof_addresses].tolist())

def run(t):
    global _installed,_active_task,_pd_readback
    assert t['config']==CFG
    key=identity(t);path=OUT/'assessments'/f'{key}.json'
    if path.exists():
        old=read(path)
        if not t['trace'] or old.get('motion_path') or (not old['full'] and t['prefix']):return old
    manifest=read(DATA/'manifest.json')
    assert sha(PLANTS[t['plant']])==manifest['plants'][t['plant']]['sha256']
    if not _installed:
        import screening
        e.fx.pilot.apply_impedance=checked_apply
        screening.install(True)
        e.task_id=identity
        _installed=True
    import screening
    screening.Guard.enabled=bool(t['prefix'])
    _active_task=t;_pd_readback=None
    e.model_path=lambda topology,terrain:PLANTS[t['plant']]
    e.fx.opt.ThreeAmplitudeController=ORIGINAL_CONTROLLER if t['phase']=='legacy' else MidpointController
    e.fx.near.near_ground=ORIGINAL_NEAR if t['initialize']=='near' else lambda ev:dict(method='original_xml_reset')
    started=time.monotonic()
    try:
        raw=e.run_one(t);a=policy.assess(raw)
        motion=Path(raw['record_path']).parent/'motion.npz'
        row=dict(id=key,task=t,wave=t['wave'],amplitudes=t['amplitudes'],D=a['D'],full=a['full'],feasible=a['feasible'],
            failed=a['failed'],metrics=raw['metrics'],record=raw['record_path'],raw_sha256=sha(raw['record_path']),
            elapsed_s=raw['elapsed_s'],pd_readback=_pd_readback,
            motion_path=str(motion) if motion.exists() else None)
    except Exception as ex:
        if hasattr(ex,'proof'):
            folder=e.TRIALS/t['stage']/key;folder.mkdir(parents=True,exist_ok=True)
            rawpath=folder/'prefix_rejection.json'
            np.savez_compressed(folder/'prefix_state.npz',**ex.state)
            write(rawpath,dict(id=key,task=t,proof=ex.proof,pd_readback=_pd_readback))
            row=dict(id=key,task=t,wave=t['wave'],amplitudes=t['amplitudes'],D=None,full=False,feasible=False,
                failed=[v['gate'].replace('upright_deficit','upright') for v in ex.proof['violations']],
                proof=ex.proof,record=str(rawpath),raw_sha256=sha(rawpath),pd_readback=_pd_readback)
        else:
            import traceback
            row=dict(id=key,task=t,wave=t['wave'],amplitudes=t['amplitudes'],D=None,full=False,feasible=False,
                failed=['execution_error'],error=repr(ex),traceback=traceback.format_exc())
    row['wall_s']=time.monotonic()-started
    write(path,row);return row

def records():return [read(p) for p in sorted((OUT/'assessments').glob('*.json'))]
def nominal(phase='legacy',plant='current'):
    return [r for r in records() if r['task']['phase']==phase and r['task']['plant']==plant and r['task']['duration']==6 and r['task']['dt_scale']==.5 and r['task']['initialize']=='near']
def bank(tasks,name,workers=12):
    manifest=read(DATA/'manifest.json')
    for p,digest in manifest['frozen_inputs'].items():assert sha(p)==digest,p
    tasks=list({identity(t):t for t in tasks}.values())
    path=DATA/'banks'/f'{name}.json'
    if path.exists():assert read(path)==tasks
    else:write(path,tasks)
    done=DATA/'completed'/f'{name}.json'
    if done.exists():return [read(OUT/'assessments'/f'{k}.json') for k in read(done)['ids']]
    ordered=[]
    bywave={w:[t for t in tasks if t['wave']==w] for w in WAVES}
    for i in range(max([len(x) for x in bywave.values()],default=0)):
        ordered.extend(bywave[w][i] for w in WAVES if i<len(bywave[w]))
    rows=[];started=time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        pending={};it=iter(ordered)
        def fill():
            while len(pending)<workers*2:
                t=next(it,None)
                if t is None:break
                pending[pool.submit(run,t)]=t
        fill()
        while pending:
            ready,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in ready:
                t=pending.pop(future);r=future.result();rows.append(r)
                status=dict(time=datetime.datetime.now().astimezone().isoformat(),stage=name,done=len(rows),total=len(tasks),
                    id=r['id'],D=r['D'],failed=r['failed'],elapsed_s=time.monotonic()-started)
                write(DATA/'live_status.json',status)
                if len(rows)%10==0 or len(rows)==len(tasks):print(json.dumps(status),flush=True)
                if 'error' in r:raise RuntimeError(r)
            fill()
    write(done,dict(ids=[r['id'] for r in rows],count=len(rows),full=sum(r['full'] for r in rows),feasible=sum(r['feasible'] for r in rows)))
    return rows
