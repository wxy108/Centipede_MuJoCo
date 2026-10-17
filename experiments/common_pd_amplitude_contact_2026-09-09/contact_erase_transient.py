"""Separate transient channel, with ground-only loss and clearance-safe restoration."""
import argparse,hashlib
import contact_erase as ce
from plane_clearance import PlaneClearance,separate_leg_geoms
c=ce.c;np=c.np
OUT=ce.ps.SOURCE_OUT/'contact_erase_transient'
KIND='transient_ground_only_bilateral_desired_legwise_safe_restore'

class Channel(ce.Channel):
    def __init__(self,ev,dt,folder):
        super().__init__(ev,dt,folder)
        assert self.task['erasure']['kind']==KIND
        self.legs=separate_leg_geoms(ev);self.leg_ids=np.concatenate(self.legs)
        self.clearance=PlaneClearance(self.m,ev.ground,self.leg_ids)
        self.period=int(round(.5/dt));self.restore_stride=int(round(.001/dt))
        assert abs(self.period*dt-.5)<1e-12 and abs(self.restore_stride*dt-.001)<1e-12
        spec=self.task['erasure'];n=int(np.ceil((self.task['duration']-1.5)/.5))
        self.uniforms=np.random.default_rng(909202600+spec['seed']).random((n,19))
        self.desired=self.uniforms<spec['probability']
        self.selected=np.zeros(38,bool);self.command=np.zeros(38,bool);self.window=-1
        self.erased_steps=np.zeros(38,np.int64);self.desired_steps=np.zeros(38,np.int64)
        self.transitions=[];self.restore_events=[];self.mask_times=[];self.mask_values=[]
        self.min_restore_clearance_m=None

    def matrix_check(self):
        ids=np.concatenate([self.legs[i] for i in np.flatnonzero(self.selected)]) if self.selected.any() else np.array([],int)
        expected=self.base_compatibility.copy();expected[self.ev.ground,ids]=False;expected[ids,self.ev.ground]=False
        assert np.array_equal(ce.compatibility(self.m.geom_contype,self.m.geom_conaffinity),expected)

    def before(self,k):
        if k>=self.start:
            window=(k-self.start)//self.period;changed=False
            if window!=self.window:
                self.window=window;self.command=np.repeat(self.desired[window],2)
                newly=self.command & ~self.selected
                for leg in np.flatnonzero(newly):self.m.geom_conaffinity[self.legs[leg]]=1
                self.selected|=self.command;changed=bool(newly.any());self.applied=True
                self.transitions.append(dict(step=int(k),window=int(window),desired_pairs=np.flatnonzero(self.desired[window]).tolist()))
            pending=self.selected & ~self.command
            if pending.any() and k%self.restore_stride==0:
                # mj_step leaves Cartesian geometry at its forward pass, before integration.
                # Update only kinematics to inspect the current qpos before restoration.
                c.mujoco.mj_kinematics(self.m,self.d)
                for leg in np.flatnonzero(pending):
                    gap=float(self.clearance.values(self.d,self.legs[leg]).min())
                    if gap>=50e-6:
                        self.m.geom_conaffinity[self.legs[leg]]=3;self.selected[leg]=False;changed=True
                        self.restore_events.append(dict(step=int(k),leg=int(leg),clearance_m=gap))
                        self.min_restore_clearance_m=gap if self.min_restore_clearance_m is None else min(gap,self.min_restore_clearance_m)
            if changed:self.matrix_check()
            self.erased_steps+=self.selected;self.desired_steps+=self.command
            if k%self.sample_stride==0:
                self.mask_times.append(k*self.dt);self.mask_values.append(self.selected.copy())
        if k%self.pose_stride==0:self.pose_t.append(float(self.d.time));self.qpos.append(self.d.qpos.copy())

    def finish(self,result):
        assert self.applied
        t=np.asarray(self.times);center=np.asarray(self.center);good=t>=1.5
        D=float(c.e.fx.opt.base.FORWARD_SIGN*(center[good][-1,0]-center[good][0,0])*1000)
        assert D==result['metrics']['displacement_mm']
        n=self.last_k-self.start+1
        result['channel']=dict(**self.task['erasure'],uniforms=self.uniforms.tolist(),
            uniform_sha256=hashlib.sha256(self.uniforms.tobytes()).hexdigest(),
            start_s=1.5,end_s=self.task['duration'],refresh_seconds=.5,restoration_check_seconds=.001,
            restoration_gap_m=50e-6,restorations=len(self.restore_events),min_restore_clearance_m=self.min_restore_clearance_m,
            desired_erased_fraction=float(self.desired_steps.sum()/(n*38)),actual_erased_fraction=float(self.erased_steps.sum()/(n*38)),
            actual_erased_fraction_per_leg=(self.erased_steps/n).tolist(),
            delayed_extra_erased_fraction=float((self.erased_steps-self.desired_steps).sum()/(n*38)),
            desired_pairs_by_window=[np.flatnonzero(v).tolist() for v in self.desired],
            final_erased_legs=np.flatnonzero(self.selected).tolist(),transitions=self.transitions,restore_events=self.restore_events,
            original_pair_compatibility_exact=True,only_selected_leg_ground_pairs_removed=True,raw_D_recomputed=D)
        c.write(self.folder/'schedule.json',result['channel'])
        np.savez_compressed(self.folder/'contact_trajectory.npz',t=t,center=center,upright=np.asarray(self.upright),
            pose_t=np.asarray(self.pose_t),qpos=np.asarray(self.qpos),mask_t=np.asarray(self.mask_times),erased_legs=np.asarray(self.mask_values))

    def save_partial(self,error):
        c.write(self.folder/'incomplete_contact_outcome.json',dict(task=self.task,error=error,last_completed_step=self.last_k,
            actual_time_s=float(self.d.time),full_window_displacement=None,uniforms=self.uniforms.tolist(),
            desired_pairs_by_window=[np.flatnonzero(v).tolist() for v in self.desired],
            final_erased_legs=np.flatnonzero(self.selected).tolist(),restore_events=self.restore_events))
        np.savez_compressed(self.folder/'incomplete_contact_state.npz',t=np.asarray(self.times),center=np.asarray(self.center),
            upright=np.asarray(self.upright),pose_t=np.asarray(self.pose_t),qpos=np.asarray(self.qpos),
            mask_t=np.asarray(self.mask_times),erased_legs=np.asarray(self.mask_values),
            final_qpos=self.d.qpos.copy(),final_qvel=self.d.qvel.copy())

ce.c.e.fx.pilot.run_task.__globals__['CONTACT_CHANNEL']=Channel
ce.c.e.fx.pilot.CONTACT_CHANNEL=Channel
c.OUT=OUT;c.DATA=OUT/'data';c.e.TRIALS=OUT/'trials'

def freeze():
    check=c.read(c.ROOT/'outputs/data/transient_static_check.json');assert check['passed']
    for p,h in check['source_hashes'].items():assert c.sha(p)==h
    selected=c.read(ce.SELECTION);assert selected['complete'] and selected['selected']
    m=c.read(ce.PHYSICAL_DATA/'manifest.json')
    m.update(contact_erasure=True,channel=KIND,amplitudes_frozen=True,selection_sha256=c.sha(ce.SELECTION),
        profiles=selected['selected'],unresolved_waves=selected['unresolved'],probabilities=[0,.2,.4,.6,.8,1],discovery_seeds=list(range(4)),confirmation_seeds=list(range(100,120)))
    for p in [ce.SELECTION,c.ROOT/'qualify.py',c.ROOT/'contact_erase.py',c.ROOT/'contact_run_task.py',c.ROOT/'contact_erase_transient.py',c.ROOT/'plane_clearance.py',c.ROOT/'TRANSIENT_ERASURE_PROTOCOL.md']:
        m['frozen_inputs'][str(p)]=c.sha(p)
    path=c.DATA/'manifest.json'
    if path.exists():assert c.read(path)==m
    else:c.write(path,m)

def task(w,p,seed,stage,duration=6.,scale=.5):
    t=ce.task(w,p,seed,stage,duration,scale);t['erasure']['kind']=KIND;return t

def baseline(workers):
    m=c.read(c.DATA/'manifest.json');rows=c.bank([task(float(w),0,0,'p0_verification') for w in m['profiles']],'p0_verification',workers)
    checks=[]
    for r in rows:
        ref=m['profiles'][str(r['wave'])]['nominal']
        exact=all(np.array_equal(np.asarray(v),np.asarray(r['metrics'][k]),equal_nan=True) for k,v in ref['metrics'].items())
        channel=c.read(r['record'])['channel']
        passed=exact and r['feasible'] and channel['actual_erased_fraction']==0 and channel['restorations']==0
        checks.append(dict(wave=r['wave'],passed=passed,all_metrics_exact=exact))
    receipt=dict(passed=all(r['passed'] for r in checks),checks=checks)
    c.write(c.DATA/'p0_verification.json',receipt);assert receipt['passed']

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','baseline','discovery']);p.add_argument('--workers',type=int,default=12);a=p.parse_args()
    if a.action=='freeze':freeze()
    elif a.action=='baseline':baseline(a.workers)
    else:
        assert c.read(c.DATA/'p0_verification.json')['passed'];m=c.read(c.DATA/'manifest.json')
        c.bank([task(float(w),p,seed,'discovery') for p in m['probabilities'] for seed in ([0] if p in (0,1) else m['discovery_seeds']) for w in m['profiles']],'discovery',a.workers)
