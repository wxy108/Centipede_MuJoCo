from pathlib import Path
import argparse,time,json,os,sys
import numpy as np
import mujoco
import shared as s
from command_cache import CommandCache
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--bank',default='calibration');p.add_argument('--dt',type=float,default=.00025);p.add_argument('--command-dt',type=float,default=.0000625);p.add_argument('--duration',type=float,default=6);p.add_argument('--integrator',default='implicitfast');p.add_argument('--position-implicit-adapter',action='store_true');p.add_argument('--case-index',type=int);a=p.parse_args()
command_stride=round(a.command_dt/a.dt);assert command_stride>=1 and abs(command_stride*a.dt-a.command_dt)<1e-12
folder=s.ROOT/'runs'/a.name;assert not folder.exists();folder.mkdir(parents=True)
s.write(folder/'pid.json',dict(pid=os.getpid(),argv=sys.argv))
rows=s.cases(a.bank);m=mujoco.MjModel.from_xml_path(str(s.ROOT/'assets/common.xml'));m.opt.timestep=a.dt
if a.case_index is not None:rows=[rows[a.case_index]]
m.opt.integrator={'implicitfast':mujoco.mjtIntegrator.mjINT_IMPLICITFAST,'RK4':mujoco.mjtIntegrator.mjINT_RK4,'implicit':mujoco.mjtIntegrator.mjINT_IMPLICIT}[a.integrator]
names=[m.joint(i).name for i in range(1,m.njnt)];body_names=[m.body(i).name for i in range(1,m.nbody)]
ids=np.array([m.actuator(n).id for n in names]);qids=np.array([m.joint(n).qposadr[0] for n in names]);dids=np.array([m.joint(n).dofadr[0] for n in names])
lookup={x['name']:x for x in s.M['joints']};kp=np.array([lookup[n]['kp'] for n in names],np.float32).astype(float);kv=np.array([lookup[n]['kv'] for n in names],np.float32).astype(float)
m.actuator_gainprm[ids,0]=kp;m.actuator_biasprm[ids,1]=-kp;m.actuator_biasprm[ids,2]=-kv
if a.position_implicit_adapter:m.actuator_biasprm[ids,2]-=a.dt*kp
stride=round(.01/a.dt);assert abs(stride*a.dt-.01)<1e-10
traces=[];started=time.monotonic();steps=round(a.duration/a.dt)
for e,row in enumerate(rows):
    cache=CommandCache(names,a.command_dt,[row])
    d=mujoco.MjData(m);d.qpos[:7]=s.M['initial_root_pose_wxyz'];mujoco.mj_forward(m,d)
    ts=[0.];qs=[d.qpos[qids].copy()];bs=[d.xpos[1:].copy()];vs=[d.qvel[dids].copy()];ns=[int(d.ncon)]
    for k in range(steps):
        target,vel=cache.get(k//command_stride);d.ctrl[ids]=target[0]+kv/kp*vel[0]
        mujoco.mj_step(m,d)
        if (k+1)%stride==0:
            mujoco.mj_kinematics(m,d)
            ts.append((k+1)*a.dt);qs.append(d.qpos[qids].copy());bs.append(d.xpos[1:].copy());vs.append(d.qvel[dids].copy());ns.append(int(d.ncon))
        if not np.isfinite(d.qpos).all():raise RuntimeError('Nonfinite simulation state')
    traces.append(dict(q=np.array(qs),body=np.array(bs),v=np.array(vs),ncon=ns,warnings=[int(x.number) for x in d.warning]))
    print(json.dumps(dict(case=e,steps=steps,wall_s=time.monotonic()-started,warnings=[int(x.number) for x in d.warning])),flush=True)
t=np.array(ts);q=np.stack([x['q'] for x in traces],axis=1);body=np.stack([x['body'] for x in traces],axis=1);v=np.stack([x['v'] for x in traces],axis=1)
np.savez_compressed(folder/'trajectory.npz',t=t,q=q,v=v,body=body,joint_names=names,body_names=body_names)
s.write(folder/'results.json',dict(engine='mujoco',version=mujoco.__version__,arguments=vars(a),wall_s=time.monotonic()-started,
    source_hashes={str(x):s.sha(x) for x in [Path(__file__),s.ROOT/'shared.py',s.ROOT/'command_cache.py',s.ROOT/'assets/common.xml',s.ROOT/'COMMON_CONTRACT.json']},
    rows=s.score(t,body,q,rows,names,body_names),contact_counts=[x['ncon'] for x in traces],warnings=[x['warnings'] for x in traces]))
print(json.dumps(s.score(t,body,q,rows,names,body_names)),flush=True)
