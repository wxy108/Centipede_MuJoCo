"""Matched contact erasure on frozen displacement-first gait profiles."""
from __future__ import annotations
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse, concurrent.futures, datetime, hashlib, importlib.util, inspect, json, math, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
OLD=ROOT.parent/'mujoco_displacement_first_amplitude_2026-09-03'
OUT=ROOT/'outputs'; DATA=OUT/'data'; TRIALS=OUT/'trials_current_runtime'
MANIFEST=DATA/'manifest.json'
def load(path,name):
    sp=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(sp)
    sys.modules[name]=mod;sp.loader.exec_module(mod);return mod
e=load(ROOT.parent/'mujoco_rigid_control_amplitude_2026-09-02/engine.py','crossover_frozen_engine')
policy=load(OLD/'policy.py','crossover_frozen_policy')
np,mujoco=e.np,e.mujoco
pilot=e.fx.pilot; opt=pilot.opt
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def write(path,value):
    path=Path(path);assert path.resolve().is_relative_to(ROOT)
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8');temp.replace(path)
def now():return datetime.datetime.now().astimezone().isoformat()
def manifest():return read(MANIFEST)
def verify():
    m=manifest()
    for r in m['frozen_files']:assert sha(r['path'])==r['sha256'],r['path']
    return {'passed':True,'count':len(m['frozen_files'])}
def freeze():
    if MANIFEST.exists():return verify()
    old=read(OLD/'outputs/data/amplitude_manifest.json')
    selected=read(OLD/'outputs/data/interim_validated_selection.json')
    profiles={}
    for w,r in selected['waves'].items():
        s=r['selected']
        if s and s['qualified']:
            raw=read(s['nominal']['record'])
            profiles[w]={'wave':float(w),'amplitudes':raw['task']['amplitudes'],'config':raw['task']['config'],
                'source':s['nominal']['record'],'source_sha256':sha(s['nominal']['record']),
                'nominal_D':s['nominal']['D'],'source_metrics':raw['metrics'],'nominal_id':raw['id']}
    paths=[old['plant'],OLD/'outputs/data/amplitude_manifest.json',OLD/'outputs/data/interim_validated_selection.json',
        ROOT/'PROTOCOL.md',ROOT/'study.py',Path(pilot.__file__),Path(e.__file__),OLD/'policy.py']
    paths += [Path(r['path']) for r in old['frozen_files']]
    paths += [Path(r['source']) for r in profiles.values()]
    files=[{'path':str(p),'sha256':sha(p)} for p in dict.fromkeys(paths)]
    value={'created':now(),'scope':'Centipede Model only','plant':old['plant'],'plant_sha256':old['plant_sha256'],
        'profiles':profiles,'excluded_waves':[.5,1,8,9],'exclusion_reason':'no selected qualified nominal profile',
        'frozen_files':files,'noise':'bilateral_pair_strong_burst_external_clock','nu':[0,.25,.5,.75,1],
        'pilot_waves':[2,3,6,7],'pilot_seeds':[0,1,2],'main_seeds':list(range(6)),
        'duration':6.,'eval_start':1.5,'support_window':.5,'gate_policy':policy.SPEC,
        'source_selection_provisional':True,'continuous_global_optimum':False,
        'versions':{'python':sys.version,'numpy':np.__version__,'mujoco':mujoco.__version__}}
    write(MANIFEST,value);return verify()

def schedule(nu,seed,duration=6.):
    rng=np.random.default_rng(9082026+int(seed));events=[]
    q=nu/.75 if nu<=.75 else 1.
    low=.5 if nu<=.75 else 2*nu-1
    count=int(round((duration-1.5)/.5))
    assert abs(1.5+count*.5-duration)<1e-9
    uniforms=rng.random((count,19,3))
    for k in range(count):
        start=1.5+k*.5
        for pair in range(19):
            u,v,z=uniforms[k,pair];a=low+(1-low)*v
            length=.5*a;on=start+z*(.5-length)
            if u<q:events.append({'pair':pair,'window':k,'start':on,'end':on+length,'u':float(u)})
    return {'nu':nu,'q':q,'duration_low_fraction':low,'seed':seed,'duration':duration,
        'expected_erased_load':nu,'opportunities':count*19,'uniform_sha256':hashlib.sha256(uniforms.tobytes()).hexdigest(),
        'events':sorted(events,key=lambda r:(r['start'],r['pair']))}

class Channel:
    def __init__(self,ev,dt,task):
        self.ev=ev;self.m=ev.model;self.d=ev.data;self.ix=ev.index;self.dt=dt;self.task=task
        self.schedule=schedule(task['nu'],task['seed'],task['duration'])
        n=int(round(task['duration']/dt));self.mask=np.zeros((n,19),dtype=bool)
        for r in self.schedule['events']:
            a=max(0,int(math.ceil(r['start']/dt-1e-9)));b=min(n,int(math.ceil(r['end']/dt-1e-9)))
            self.mask[a:b,r['pair']]=True
        self.original_type=self.m.geom_contype.copy();self.original_aff=self.m.geom_conaffinity.copy()
        self.groups=ev.segment_collision_geoms;self.active=np.zeros(19,dtype=bool)
        assert len(self.groups)==19 and len(set(np.concatenate(self.groups)))==sum(map(len,self.groups))
        assert not set(np.concatenate(self.groups)) & set(ev.chassis_geoms)
        self.mask_sha=hashlib.sha256(self.mask.tobytes()).hexdigest();self.command_hash=hashlib.sha256()
        self.transitions=[];self.jumps=[];self.restore_steps=[];self.pose_t=[];self.qpos=[];self.pose_masks=[]
        self.raw_center=[];self.raw_t=[];self.upright=[];self.root_yaw=[];self.pre=np.zeros(6)
        self.restoring=False;self.last_k=-1;self.sample=max(1,int(round(.001/dt)))
        self.video_stride=max(1,int(round((1/60)/dt)))
    def before_step(self,k):
        if abs(float(self.d.time)-k*self.dt)>1e-7:
            raise RuntimeError(f'solver_time_reset at step {k}: {self.d.time}')
        desired=self.mask[k];off=np.flatnonzero(desired & ~self.active);on=np.flatnonzero(~desired & self.active)
        self.restoring=bool(len(on));self.pre[:]=self.d.qvel[:6]
        for p in on:
            gaps=[float(mujoco.mj_geomDistance(self.m,self.d,self.ev.ground,int(g),.1,None)) for g in self.groups[p]]
            self.transitions.append({'time':k*self.dt,'pair':int(p),'min_gap_m':min(gaps)})
            ids=self.groups[p];self.m.geom_contype[ids]=self.original_type[ids];self.m.geom_conaffinity[ids]=self.original_aff[ids]
        for p in off:
            ids=self.groups[p];self.m.geom_contype[ids]=0;self.m.geom_conaffinity[ids]=0
        self.active=desired.copy()
        if k%self.video_stride==0:
            self.pose_t.append(k*self.dt);self.qpos.append(self.d.qpos.copy());self.pose_masks.append(desired.copy())
    def after_step(self,k):
        self.last_k=k;self.command_hash.update(self.d.ctrl.tobytes())
        self.jumps.append(float(np.linalg.norm(self.d.qvel[:3]-self.pre[:3])));self.restore_steps.append(self.restoring)
        if not np.isfinite(self.d.qpos).all() or not np.isfinite(self.d.qvel).all():raise RuntimeError('nonfinite_state')
        if abs(float(self.d.time)-(k+1)*self.dt)>1e-7:raise RuntimeError('solver_time_reset')
        if k%self.sample==0:
            self.raw_t.append(k*self.dt);self.raw_center.append(self.d.xpos[self.ix.chassis_body_ids].mean(axis=0).copy())
            mat=self.d.xmat[int(self.ix.chassis_body_ids[0])].reshape(3,3)
            self.upright.append(float(mat[2,2]));self.root_yaw.append(float(np.arctan2(mat[1,0],mat[0,0])))
    def finish(self):
        good=int(round(1.5/self.dt));j=np.asarray(self.jumps)[good:];r=np.asarray(self.restore_steps,dtype=bool)[good:]
        def stats(a):return {'count':int(a.size),'p95':float(np.quantile(a,.95)) if a.size else None,'max':float(a.max()) if a.size else None}
        return {'channel':{'nu':self.task['nu'],'q':self.schedule['q'],'seed':self.task['seed'],
            'schedule_sha256':digest(self.schedule),'mask_sha256':self.mask_sha,
            'command_sha256':self.command_hash.hexdigest(),'realized_erased_load':float(self.mask[good:].mean()),
            'erased_load_per_pair':self.mask[good:].mean(axis=0).tolist(),'event_count':len(self.schedule['events']),
            'collision_geoms_per_pair':list(map(len,self.groups)),'restoration_count':len(self.transitions),
            'restorations_overlapping_floor':sum(r['min_gap_m']<0 for r in self.transitions),
            'restoration_root_speed_jump_m_s':stats(j[r]),'ordinary_root_speed_jump_m_s':stats(j[~r])}}
    def save(self,folder):
        folder.mkdir(parents=True,exist_ok=True)
        write(folder/'events.json',self.schedule);write(folder/'restorations.json',self.transitions)
        np.savez_compressed(folder/'trajectory.npz',t=np.asarray(self.raw_t),center=np.asarray(self.raw_center),
            upright=np.asarray(self.upright),root_yaw=np.asarray(self.root_yaw),
            pose_t=np.asarray(self.pose_t),qpos=np.asarray(self.qpos),erased_pose=np.asarray(self.pose_masks),
            root_speed_jump=np.asarray(self.jumps),restoration_step=np.asarray(self.restore_steps),
            erased_timestep=self.mask[:self.last_k+1],dt=self.dt)

def condition_id(t):
    return f"w{t['wave']:g}_nu{t['nu']:g}_s{t['seed']}_t{t['duration']:g}_dt{t['dt_scale']:g}".replace('.','p')
def task(w,nu,seed=0,duration=6.,dt_scale=1.,stage='main'):
    return dict(wave=float(w),nu=float(nu),seed=int(seed if 0<nu<1 else 0),duration=float(duration),dt_scale=float(dt_scale),stage=stage)

def run_one(t):
    mid=condition_id(t);folder=TRIALS/mid;record=folder/'result.json'
    if record.exists():return read(record)
    mf=manifest();p=mf['profiles'][f"{t['wave']:g}"]
    assert sha(mf['plant'])==mf['plant_sha256'];assert 0<=t['nu']<=1
    opt.MODEL=Path(mf['plant']);opt.FastEvaluator=e.bounded_evaluator
    current=[]
    def factory(ev,dt):
        channel=Channel(ev,dt,t);current.append(channel);return channel
    source=inspect.getsource(pilot.run_task)
    changes=[('    finite=True\n','    finite=True\n    channel=CHANNEL_FACTORY(ev,dt)\n'),
        ('        prevel[:]=d.qvel\n','        channel.before_step(k)\n        prevel[:]=d.qvel\n'),
        ('        mujoco.mj_step(m,d)\n','        mujoco.mj_step(m,d)\n        channel.after_step(k)\n'),
        ('    if save_trace:\n','    result.update(channel.finish())\n    if save_trace:\n')]
    for old,new in changes:
        assert source.count(old)==1,old;source=source.replace(old,new)
    namespace=dict(vars(pilot),CHANNEL_FACTORY=factory)
    exec(compile(source,str(ROOT/'instrumented_evaluator'),'exec'),namespace)
    cfg=dict(p['config'],name=mid)
    start=time.monotonic()
    try:
        r=namespace['run_task']((cfg,t['wave'],t['duration'],t['dt_scale'],False),
            initialize=e.fx.near.near_ground,case_override=dict(p['amplitudes'],wave=t['wave']),output_dir=folder)
        classified=policy.assess(r)
        r.update(id=mid,task=t,amplitudes=p['amplitudes'],profile_source=p['source'],
            model_path=mf['plant'],completed=True,viable=classified['feasible'],viability_gates=classified['gates'],
            failed_viability=classified['failed'],D=r['metrics']['displacement_mm'],
            retained=r['metrics']['displacement_mm']/p['nominal_D'],upright_survived=classified['gates']['upright'],
            forward_sign=opt.base.FORWARD_SIGN,record_path=str(record))
        if t['nu']==0 and t['duration']==6 and t['dt_scale']==1:
            r['nominal_exact_all_metrics']=all(np.array_equal(np.asarray(v),np.asarray(r['metrics'][k]),equal_nan=True) for k,v in p['source_metrics'].items())
            r['nominal_D_relative_difference']=abs(r['D']-p['nominal_D'])/p['nominal_D']
            r['nominal_current_runtime_accepted']=r['nominal_D_relative_difference']<=.01 and r['viable']
            assert r['nominal_current_runtime_accepted'],'no-loss current runtime baseline failed the disclosed 1% displacement / unchanged viability check'
        ch=current[0];arr=np.asarray(ch.raw_center);tt=np.asarray(ch.raw_t);good=tt>=1.5
        recomputed=float(opt.base.FORWARD_SIGN*(arr[good][-1,0]-arr[good][0,0])*1000)
        assert recomputed==r['D'],(recomputed,r['D'])
        r['D_recomputed_from_raw']=recomputed
    except Exception as exc:
        r={'id':mid,'task':t,'completed':False,'D':None,'error':repr(exc),'elapsed_s':time.monotonic()-start,
            'model_path':mf['plant'],'model_sha256':mf['plant_sha256'],'record_path':str(record)}
    finally:
        if current:current[0].save(folder)
    write(record,r);return r

def tasks_for(stage):
    m=manifest();waves=m['pilot_waves'] if stage=='pilot' else list(map(float,m['profiles']))
    seeds=m['pilot_seeds'] if stage=='pilot' else m['main_seeds']
    return [task(w,nu,s,stage=stage) for nu in m['nu'] for s in ([0] if nu in (0,1) else seeds) for w in waves]
def bank(tasks,stage,workers):
    verify();path=DATA/f'{stage}_tasks.json'
    if path.exists():assert read(path)==tasks
    else:write(path,tasks)
    start=time.monotonic();done=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(run_one,t):t for t in tasks}
        for f in concurrent.futures.as_completed(futures):
            r=f.result();done.append(r)
            status={'time':now(),'stage':stage,'done':len(done),'total':len(tasks),'id':r['id'],
                'D':r['D'],'completed':r['completed'],'viable':r.get('viable'),'error':r.get('error'),'elapsed_s':time.monotonic()-start}
            write(DATA/'live_status.json',status);print(json.dumps(status),flush=True)
            if not r['completed'] and 'no-loss' in r.get('error',''):raise RuntimeError(r['error'])
    write(DATA/f'{stage}_complete.json',{'time':now(),'count':len(done),'completed':sum(r['completed'] for r in done),
        'records':[r['record_path'] for r in done]})
    return done

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['freeze','verify','baseline_check','pilot','main','one'])
    ap.add_argument('--workers',type=int,default=12);ap.add_argument('--wave',type=float,default=2);ap.add_argument('--nu',type=float,default=0);ap.add_argument('--seed',type=int,default=0)
    a=ap.parse_args()
    if a.command=='freeze':print(json.dumps(freeze()))
    elif a.command=='verify':print(json.dumps(verify()))
    elif a.command=='baseline_check':bank([task(w,0,stage='baseline_check') for w in [2,6]],'baseline_check',2)
    elif a.command=='one':print(json.dumps(run_one(task(a.wave,a.nu,a.seed))))
    else:bank(tasks_for(a.command),a.command,a.workers)
