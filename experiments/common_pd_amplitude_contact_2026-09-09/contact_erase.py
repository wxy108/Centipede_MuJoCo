"""Ground-only persistent random erasure on a frozen common-PD gait bank."""
import argparse,hashlib,inspect,json
import physical_search as ps
import core as c
np=c.np
PHYSICAL_DATA=c.DATA
CONTACT_OUT=ps.SOURCE_OUT/'contact_erase'
SELECTION=PHYSICAL_DATA/'validated_selection.json'
BASE_RUN_ONE=c.e.run_one
BASE_ASSESSMENT_RUN=c.run
LAST_CHANNEL=None


def compatibility(types,affinities):
    return ((types[:,None]&affinities[None,:])!=0)|((types[None,:]&affinities[:,None])!=0)


class Channel:
    def __init__(self,ev,dt,folder):
        global LAST_CHANNEL
        LAST_CHANNEL=self
        self.ev=ev;self.dt=dt;self.folder=folder;self.task=c._active_task;self.m=ev.model;self.d=ev.data
        spec=self.task['erasure'];p=spec['probability'];seed=spec['seed']
        self.uniforms=np.random.default_rng(909202600+seed).random(19)
        self.selected=self.uniforms<p;self.groups=ev.segment_collision_geoms
        assert len(self.groups)==19 and len(set(np.concatenate(self.groups)))==sum(map(len,self.groups))
        assert not set(np.concatenate(self.groups))&set(ev.chassis_geoms)
        original_type=self.m.geom_contype.copy();original_aff=self.m.geom_conaffinity.copy()
        original=compatibility(original_type,original_aff)
        colliding=(original_type!=0)|(original_aff!=0)
        assert np.all(original_type[colliding]==1) and np.all(original_aff[colliding]==1)
        self.m.geom_contype[colliding]=1;self.m.geom_conaffinity[colliding]=3
        self.m.geom_contype[ev.ground]=2;self.m.geom_conaffinity[ev.ground]=2
        assert np.array_equal(compatibility(self.m.geom_contype,self.m.geom_conaffinity),original)
        self.base_compatibility=original;self.original_robot_type=self.m.geom_contype.copy();self.original_robot_aff=self.m.geom_conaffinity.copy()
        self.active=False;self.applied=False;self.pose_t=[];self.qpos=[];self.times=[];self.center=[];self.upright=[]
        self.pose_stride=max(1,round(1/30/dt));self.sample_stride=max(1,round(.001/dt));self.start=int(round(1.5/dt))
        self.changed=[]
        self.last_k=-1

    def before(self,k):
        if k==self.start:
            ids=np.concatenate([self.groups[i] for i in np.flatnonzero(self.selected)]) if self.selected.any() else np.array([],dtype=int)
            self.m.geom_conaffinity[ids]=1
            expected=self.base_compatibility.copy();expected[self.ev.ground,ids]=False;expected[ids,self.ev.ground]=False
            assert np.array_equal(compatibility(self.m.geom_contype,self.m.geom_conaffinity),expected)
            self.changed=list(map(int,ids));self.applied=True
        if k%self.pose_stride==0:self.pose_t.append(float(self.d.time));self.qpos.append(self.d.qpos.copy())

    def after(self,k):
        self.last_k=k
        if k%self.sample_stride==0:
            self.times.append(k*self.dt);self.center.append(self.d.xpos[self.ev.index.chassis_body_ids].mean(0).copy())
            self.upright.append(float(self.d.xmat[int(self.ev.index.chassis_body_ids[0])].reshape(3,3)[2,2]))

    def finish(self,result):
        assert self.applied
        t=np.asarray(self.times);center=np.asarray(self.center);good=t>=1.5
        D=float(c.e.fx.opt.base.FORWARD_SIGN*(center[good][-1,0]-center[good][0,0])*1000)
        assert D==result['metrics']['displacement_mm']
        result['channel']=dict(**self.task['erasure'],
            uniforms=self.uniforms.tolist(),uniform_sha256=hashlib.sha256(self.uniforms.tobytes()).hexdigest(),
            erased_pairs=np.flatnonzero(self.selected).tolist(),actual_erased_fraction=float(self.selected.mean()),
            start_s=1.5,end_s=self.task['duration'],restorations=0,
            original_pair_compatibility_exact=True,only_selected_leg_ground_pairs_removed=True,
            changed_geom_ids=self.changed,raw_D_recomputed=D)
        c.write(self.folder/'schedule.json',result['channel'])
        np.savez_compressed(self.folder/'contact_trajectory.npz',t=t,center=center,upright=np.asarray(self.upright),
            pose_t=np.asarray(self.pose_t),qpos=np.asarray(self.qpos),erased_pairs=self.selected)

    def save_partial(self,error):
        c.write(self.folder/'incomplete_contact_outcome.json',dict(task=self.task,error=error,last_completed_step=self.last_k,
            actual_time_s=float(self.d.time),full_window_displacement=None,uniforms=self.uniforms.tolist(),erased_pairs=np.flatnonzero(self.selected).tolist()))
        np.savez_compressed(self.folder/'incomplete_contact_state.npz',t=np.asarray(self.times),center=np.asarray(self.center),
            upright=np.asarray(self.upright),pose_t=np.asarray(self.pose_t),qpos=np.asarray(self.qpos),
            final_qpos=self.d.qpos.copy(),final_qvel=self.d.qvel.copy())


def corrected_record(t):
    r=BASE_RUN_ONE(t);r['contact_erasure']=True
    c.write(r['record_path'],r);return r


def run_contact(t):
    global LAST_CHANNEL
    LAST_CHANNEL=None
    r=BASE_ASSESSMENT_RUN(t)
    if 'error' in r and LAST_CHANNEL is not None:LAST_CHANNEL.save_partial(r['error'])
    if 'error' in r and t['erasure']['probability']>0 and any(
        reason in r['error'] for reason in ('nonfinite simulation state','solver time reset')):
        r['numerical_failure']=r.pop('error')
        r['failed']=['incomplete_numerical_outcome'];r['D']=None
        c.write(c.OUT/'assessments'/f"{r['id']}.json",r)
    return r


def install():
    pilot=c.e.fx.pilot;source=inspect.getsource(pilot.run_task)
    changes=[('    finite=True\n','    finite=True\n    channel=CONTACT_CHANNEL(ev,dt,output_dir)\n'),
        ('        prevel[:]=d.qvel\n','        channel.before(k)\n        prevel[:]=d.qvel\n'),
        ('        mujoco.mj_step(m,d)\n','        mujoco.mj_step(m,d)\n        channel.after(k)\n'),
        ('    if save_trace:\n','    channel.finish(result)\n    if save_trace:\n')]
    for old,new in changes:assert source.count(old)==1;source=source.replace(old,new)
    path=c.ROOT/'contact_run_task.py'
    if path.exists():assert path.read_text()==source
    else:path.write_text(source,encoding='utf-8')
    ns=dict(vars(pilot),CONTACT_CHANNEL=Channel);exec(compile(source,str(path),'exec'),ns)
    pilot.run_task=ns['run_task'];pilot.CONTACT_CHANNEL=Channel
    c.OUT=CONTACT_OUT;c.DATA=c.OUT/'data';c.e.TRIALS=c.OUT/'trials';c.e.run_one=corrected_record;c.run=run_contact


install()


def freeze():
    check=c.read(c.ROOT/'outputs/data/channel_static_check.json')
    assert check['passed'] and check['code_sha256']==c.sha(__file__)
    selected=c.read(SELECTION);assert selected['complete'] and selected['selected']
    manifest=c.read(PHYSICAL_DATA/'manifest.json')
    manifest.update(contact_erasure=True,amplitudes_frozen=True,selection_sha256=c.sha(SELECTION),
        profiles=selected['selected'],unresolved_waves=selected['unresolved'],
        probabilities=[0,.2,.4,.6,.8,1],discovery_seeds=list(range(4)),confirmation_seeds=list(range(100,120)))
    for p in (SELECTION,c.ROOT/'qualify.py',c.ROOT/'contact_erase.py',c.ROOT/'contact_run_task.py',c.ROOT/'ERASURE_PROTOCOL.md'):
        manifest['frozen_inputs'][str(p)]=c.sha(p)
    path=c.DATA/'manifest.json'
    if path.exists():assert c.read(path)==manifest
    else:c.write(path,manifest)


def task(w,p,seed,stage,duration=6.,scale=.5):
    profile=c.read(c.DATA/'manifest.json')['profiles'][str(float(w))]
    assert profile['qualified']
    t=c.task(w,[profile['amplitude'][k] for k in c.KEYS],stage,phase='midpoint',duration=duration,scale=scale,prefix=False)
    t['erasure']=dict(probability=float(p),seed=int(seed if 0<p<1 else 0),kind='persistent_ground_only_bilateral_pair')
    return t


def baseline(workers):
    m=c.read(c.DATA/'manifest.json');rows=c.bank([task(float(w),0,0,'p0_verification') for w in m['profiles']],'p0_verification',workers)
    checks=[]
    for r in rows:
        reference=m['profiles'][str(r['wave'])]['nominal']
        exact=all(np.array_equal(np.asarray(v),np.asarray(r['metrics'][k]),equal_nan=True) for k,v in reference['metrics'].items())
        channel=c.read(r['record'])['channel']
        checks.append(dict(wave=r['wave'],D_difference=r['D']-reference['D'],all_metrics_exact=exact,
            pair_compatibility_exact=channel['original_pair_compatibility_exact'],no_erased_pairs=not channel['erased_pairs'],
            passed=exact and r['feasible'] and not channel['erased_pairs']))
    receipt=dict(passed=all(x['passed'] for x in checks),checks=checks)
    c.write(c.DATA/'p0_verification.json',receipt);assert receipt['passed'],receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','baseline','discovery']);p.add_argument('--workers',type=int,default=12)
    a=p.parse_args()
    if a.action=='freeze':freeze()
    elif a.action=='baseline':baseline(a.workers)
    else:
        assert c.read(c.DATA/'p0_verification.json')['passed']
        m=c.read(c.DATA/'manifest.json')
        tasks=[task(float(w),p,seed,'discovery') for p in m['probabilities'] for seed in ([0] if p in (0,1) else m['discovery_seeds']) for w in m['profiles']]
        c.bank(tasks,'discovery',a.workers)
