"""Fixed-amplitude PD-only search. No changes to frozen XML or CPG law."""
from __future__ import annotations
import argparse
import concurrent.futures
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
OLD=ROOT.parent/'mujoco_pd_multimetric_stability_2026-09-01'
spec=importlib.util.spec_from_file_location('near_pd_pilot',OLD/'run_pd_pilot.py')
pilot=importlib.util.module_from_spec(spec);sys.modules[spec.name]=pilot;spec.loader.exec_module(pilot)
np,mujoco,opt=pilot.np,pilot.mujoco,pilot.opt
OUT=ROOT/'outputs'

def near_ground(ev,clearance=25e-6):
    """Rigid root transform only. Keep all relative joint positions and velocities zero."""
    m,d,ix=ev.model,ev.data,ev.index
    original=d.qpos.copy()
    first=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'link_body_1')
    # Inverse of the fixed head-to-link1 rotation makes link1/body horizontal.
    q=np.empty(4);mujoco.mju_negQuat(q,m.body_quat[first])
    d.qpos[3:7]=q
    mujoco.mj_forward(m,d)
    geoms=[g for g in range(m.ngeom) if g!=ev.ground and
           ((m.geom_contype[g]&m.geom_conaffinity[ev.ground]) or
            (m.geom_contype[ev.ground]&m.geom_conaffinity[g]))]
    before=np.asarray([mujoco.mj_geomDistance(m,d,ev.ground,g,1.,None) for g in geoms])
    d.qpos[2]+=clearance-float(before.min())
    mujoco.mj_forward(m,d)
    after=np.asarray([mujoco.mj_geomDistance(m,d,ev.ground,g,1.,None) for g in geoms])
    assert np.array_equal(original[7:],d.qpos[7:])
    assert np.count_nonzero(d.qvel)==0
    assert abs(float(after.min())-clearance)<1e-8
    body_z=d.xpos[ix.chassis_body_ids[1:],2]
    return dict(method='root_only_near_ground',clearance_m=clearance,root_before=original[:7].tolist(),
                root_after=d.qpos[:7].tolist(),minimum_collision_clearance_m=float(after.min()),
                lowest_geom=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(geoms[int(np.argmin(after))])),
                chassis_origin_height_spread_m=float(np.ptp(body_z)),
                all_relative_joint_positions_unchanged=True,initial_velocity_zero=True,
                settling='original unchanged zero-target interval t=0..1s')

def cfg(name,**kwargs):
    c=pilot.candidate(name);c['yaw_k']=1.;c.update(kwargs);return c

PREVIOUS=next(c for c in pilot.BANK if c['name']=='p4_leg_k2_yaw_d2_roll_d4')
PREVIOUS=dict(PREVIOUS,yaw_k=1.)

def amplitude_case(wave,mode):
    if mode=='per-wave':return pilot.wave_case(wave)
    return dict(pilot.wave_case(2.5),wave=wave)

def run_one(task):
    c,w,dur,dt,trace,mode,*flags=task
    if flags and flags[0]:
        proof=precheck(c,w,dur,dt,mode)
        if proof is not None:return proof
    result=pilot.run_task((c,w,dur,dt,trace),initialize=near_ground,
                          case_override=amplitude_case(w,mode),output_dir=OUT/mode)
    return result

def precheck(c,w,duration,dt_scale,mode):
    """Reject only an irreversible per-step hard-gate failure. Never certify here."""
    started=time.monotonic();ev=opt.FastEvaluator();m,d,ix=ev.model,ev.data,ev.index
    pilot.apply_impedance(m,ix,c);m.opt.timestep*=dt_scale
    mujoco.mj_resetData(m,d);mujoco.mj_forward(m,d);init=near_ground(ev)
    case=amplitude_case(w,mode)
    ctl=opt.ThreeAmplitudeController(m,ix,w,case['body'],case['leg_yaw'],case['leg_pitch'])
    step=float(m.opt.timestep)
    while d.time<duration-.5*step:
        ctl.step(d);mujoco.mj_step(m,d)
        limits=[('torque',float(np.max(np.abs(d.actuator_force[ix.active_actuator_ids]))),2e-4)]
        if d.time>=1.5+step-1e-10:
            limits.extend([
                ('body_yaw_speed',float(np.max(np.abs(d.qvel[ix.dof_addresses[ix.body_positions]]))),12.),
                ('body_pitch_speed',float(np.max(np.abs(d.qvel[ix.dof_addresses[ix.pitch_positions]]))),8.),
                ('body_roll_speed',float(np.max(np.abs(d.qvel[ix.roll_dof_addresses]))),5.),
                ('tip_speed',float(np.max(np.abs(d.qvel[ix.tip_dof_addresses]))),1000.),
                ('tip_peak',float(np.degrees(np.max(np.abs(d.qpos[ix.tip_qpos_addresses])))),40.)])
        failed=[(name,val,limit) for name,val,limit in limits if not np.isfinite(val) or val>limit]
        if failed:
            tag=f"{c['name']}_w{w:g}_t{duration:g}_dt{step:g}".replace('.','p')
            result=dict(id=tag,config=c,case=case,duration=duration,dt=step,
                initialization=init,model_sha256=opt.tip.sha256(opt.MODEL),elapsed_s=time.monotonic()-started,
                proven_infeasible=True,precheck=True,checked_until_s=float(d.time),
                failure_evidence=[dict(gate=name,value=val,limit=limit) for name,val,limit in failed],
                gates={name:False for name,_,_ in failed},metrics=None,
                objective='not compared: truncated run',remaining_gates='not evaluated')
            path=OUT/mode;path.mkdir(exist_ok=True,parents=True)
            (path/f'{tag}.json').write_text(json.dumps(result,indent=2));return result
    return None

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--waves',nargs='+',type=float,default=[1.,1.5,2.,2.5,3.,4.,5.,6.,9.])
    p.add_argument('--bank',type=Path)
    p.add_argument('--duration',type=float,default=6.)
    p.add_argument('--dt-scale',type=float,default=1.)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--mode',choices=['common','per-wave'],default='common')
    p.add_argument('--trace',action='store_true')
    p.add_argument('--init-only',action='store_true')
    p.add_argument('--precheck',action='store_true')
    a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if a.init_only:
        ev=opt.FastEvaluator();mujoco.mj_forward(ev.model,ev.data)
        r=near_ground(ev);(OUT/'initialization_geometry.json').write_text(json.dumps(r,indent=2));print(json.dumps(r));return
    bank=json.loads(a.bank.read_text()) if a.bank else [cfg('baseline'),PREVIOUS]
    tasks=[]
    for c in bank:
        for w in a.waves:
            tag=f"{c['name']}_w{w:g}_t{a.duration:g}_dt{.00025*a.dt_scale:g}".replace('.','p')
            path=OUT/a.mode/f'{tag}.json'
            if path.exists() and (not a.trace or path.with_suffix('.npz').exists()):continue
            tasks.append((c,w,a.duration,a.dt_scale,a.trace,a.mode,a.precheck))
    print(f'pending={len(tasks)} workers={a.workers} mode={a.mode}',flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(run_one,t) for t in tasks]
        for i,f in enumerate(concurrent.futures.as_completed(futures),1):
            r=f.result();m=r['metrics']
            print(json.dumps(dict(done=i,total=len(tasks),id=r['id'],seconds=r['elapsed_s'],
                D=m['displacement_mm'] if m else None,checked_until=r.get('checked_until_s',r['duration']),
                failed=[k for k,v in r['gates'].items() if not v])),flush=True)

if __name__=='__main__':main()
