"""Read-only numerical/performance experiment, not used for search decisions."""
import time
import core as c
import mujoco.rollout as rollout
np=c.np;mj=c.mujoco

def run(row):
    begin=time.monotonic();t=row['task'];c.e.fx.opt.MODEL=c.history()['plant'];ev=c.e.bounded_evaluator();m,d,ix=ev.model,ev.data,ev.index
    c.e.fx.pilot.apply_impedance(m,ix,t['config']);m.opt.timestep*=t['dt_scale'];dt=float(m.opt.timestep);n=round(t['duration']/dt)
    mj.mj_resetData(m,d);mj.mj_forward(m,d);c.e.fx.near.near_ground(ev)
    spec=mj.mjtState.mjSTATE_FULLPHYSICS;initial=np.empty(mj.mj_stateSize(m,spec));mj.mj_getState(m,d,initial,spec);warm=d.qacc_warmstart.copy()
    probe=mj.MjData(m);ctl=c.e.fx.opt.ThreeAmplitudeController(m,ix,t['wave'],*[t['amplitudes'][k] for k in c.KEYS]);controls=np.empty((n,m.nu))
    for k in range(n):ctl.step(probe);controls[k]=probe.ctrl;probe.time+=dt
    pre=time.monotonic();states,sensors=rollout.rollout(m,d,initial,controls,initial_warmstart=warm);native=time.monotonic()
    assert np.isfinite(states).all() and np.all(np.diff(states[0,:,0])>0)
    centers=[]
    for k in [round(1.5/dt),round(5.999/dt)]:
        mj.mj_setState(m,d,states[0,k-1],spec);mj.mj_kinematics(m,d);centers.append(d.xpos[ix.chassis_body_ids].mean(axis=0).copy())
    D=float(c.e.fx.opt.base.FORWARD_SIGN*(centers[1][0]-centers[0][0])*1000)
    return dict(id=row['id'],D_native=D,D_full=row['D'],difference_mm=D-row['D'],exact_D=D==row['D'],integrator=int(m.opt.integrator),
        controller_precompute_s=pre-begin,native_physics_s=native-pre,total_s=time.monotonic()-begin,old_full_elapsed_s=row['elapsed_s'],warning_counts=c.e.fx.opt.tip.stage1.warning_counts(d),not_used_for_selection=True)

if __name__=='__main__':
    row=max([r for r in c.read(c.DATA/'warm_fresh_results.json') if r['wave']==2 and r['feasible']],key=lambda r:r['D'])
    result=run(row);c.write(c.DATA/'native_prototype.json',result);print(result)
