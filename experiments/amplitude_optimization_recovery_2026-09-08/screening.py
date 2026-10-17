"""Conservative prefix rejection using only necessary unchanged viability gates."""
import inspect,math
from pathlib import Path
import core as c
np=c.np

class PrefixRejected(Exception):
    def __init__(self,proof,state):
        super().__init__('irreversible original-policy violation');self.proof=proof;self.state=state

class Guard:
    enabled=True
    def __init__(self,ev,n,dt,trace):
        self.ev=ev;self.n=n;self.dt=dt;self.stride=max(1,round(.001/dt));self.trace=trace
        self.pose_t=[];self.qpos=[];self.pose_stride=max(1,round((1/60)/dt))
        self.weight=float(ev.model.body_mass.sum()*9.81);self.last_k=-1
    def before(self,k):
        d=self.ev.data
        if k%self.stride==0 and abs(float(d.time)-k*self.dt)>1e-7:raise RuntimeError('solver time reset; incomplete outcome')
        if self.trace and k%self.pose_stride==0:self.pose_t.append(float(d.time));self.qpos.append(d.qpos.copy())
    def after(self,k,forces,pen,torque,speed,tip_speed,tip_angle):
        self.last_k=k
        if not np.isfinite(self.ev.data.qpos).all() or not np.isfinite(self.ev.data.qvel).all():raise RuntimeError('nonfinite simulation state')
        if not self.enabled or k%self.stride: return
        d=self.ev.data;ix=self.ev.index
        values=[('torque',float(torque),2e-4)]
        if k*self.dt>=1.5:
            rm=d.xmat[int(ix.chassis_body_ids[0])].reshape(3,3);xyz=d.xpos[ix.chassis_body_ids]
            spine=float(np.linalg.norm(np.diff(xyz,axis=0),axis=1).sum()/max(np.linalg.norm(xyz[-1]-xyz[0]),1e-15))
            values += [('active_speed',float(speed),200.),('tip_speed',float(tip_speed),1000.),('tip_peak',float(np.degrees(tip_angle)),40.),
                ('penetration',float(pen)*1000,.25),('total_force',float(forces.sum()/self.weight),5.),
                ('chassis_force',float(forces[38:59].sum()/self.weight),2.),('foot_force',float(forces[:38].max()/self.weight),2.),
                ('root_roll_peak',abs(float(np.degrees(np.arctan2(rm[2,1],rm[2,2])))),35.),
                ('upright_deficit',1-float(rm[2,2]),1-math.cos(math.radians(50))),('spine',spine,1.2)]
        bad=[dict(gate=k,value=v,limit=l) for k,v,l in values if v>l+max(1e-12,abs(l)*1e-9)]
        if bad:
            proof=dict(label_time_s=k*self.dt,actual_sim_time_s=float(d.time),violations=bad,full_horizon_ranked=False,
                note='Necessary peak/extreme gate exceeded on a sample shared with the unchanged full evaluator; no prefix displacement assigned.')
            state=dict(time=float(d.time),qpos=d.qpos.copy(),qvel=d.qvel.copy())
            raise PrefixRejected(proof,state)
    def finish(self,result,a,folder):
        assert self.last_k==self.n-1
        assert abs(float(self.ev.data.time)-self.n*self.dt)<1e-7
        tt=a['t'];centers=a['xpos'].mean(axis=1);good=tt>=1.5
        value=float(c.e.fx.opt.base.FORWARD_SIGN*(centers[good][-1,0]-centers[good][0,0])*1000)
        assert value==result['metrics']['displacement_mm']
        result['executed_full_horizon']=True;result['raw_D_recomputed']=value
        if self.trace:
            np.savez_compressed(Path(folder)/'motion.npz',pose_t=np.asarray(self.pose_t),qpos=np.asarray(self.qpos),
                t=tt,center=centers,bq=a['bq'],lq=a['lq'],tq=a['tq'],rootmat=a['rootmat'])

def install(enabled=True):
    pilot=c.e.fx.pilot
    if hasattr(pilot,'recovery_original_run_task'):pilot.run_task=pilot.recovery_original_run_task
    else:pilot.recovery_original_run_task=pilot.run_task
    Guard.enabled=enabled;src=inspect.getsource(pilot.recovery_original_run_task)
    replacements=[('    finite=True\n','    finite=True\n    guard=RECOVERY_GUARD(ev,n,dt,save_trace)\n'),
        ('        prevel[:]=d.qvel\n','        guard.before(k)\n        prevel[:]=d.qvel\n'),
        ('        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():\n','        guard.after(k,force[k],pen[k],torque[k],all_speed[k],all_tip_speed[k],tipmax[k])\n        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():\n'),
        ('    if save_trace:\n','    guard.finish(result,a,output_dir)\n    if save_trace:\n')]
    for old,new in replacements:assert src.count(old)==1;src=src.replace(old,new)
    ns=dict(vars(pilot),RECOVERY_GUARD=Guard);exec(compile(src,str(c.ROOT/'screening_instrumented_evaluator'),'exec'),ns);pilot.run_task=ns['run_task']
