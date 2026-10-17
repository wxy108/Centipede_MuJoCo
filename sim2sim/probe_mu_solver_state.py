"""Short solver diagnostic initialized from a recorded pose; not a scored validation run."""
import argparse,time,json
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
import shared as s
from command_cache import CommandCache
p=argparse.ArgumentParser();p.add_argument('name');a=p.parse_args();folder=s.ROOT/'runs'/a.name
path=folder/'trajectory.npz'
if not path.exists():path=folder/'partial_trajectory.npz'
tr={k:v.copy() for k,v in np.load(path).items()};meta=json.loads((folder/'pid.json').read_text());argv=meta['argv'];dt=float(argv[argv.index('--dt')+1]);case=int(argv[argv.index('--case-index')+1]);row=s.cases('holdout')[case]
m=mujoco.MjModel.from_xml_path(str(s.ROOT/'assets/common.xml'));m.opt.timestep=dt;m.geom_solref[:,0]=2*dt;d=mujoco.MjData(m)
names=tr['joint_names'].tolist();bn=tr['body_names'].tolist();qids=[m.joint(n).qposadr[0] for n in names];vids=[m.joint(n).dofadr[0] for n in names];bids=[m.body(n).id for n in bn];aids=[m.actuator(n).id for n in names]
lookup={j['name']:j for j in s.M['joints']};kp=np.array([lookup[n]['kp'] for n in names],np.float32).astype(float);kd=np.array([lookup[n]['kv'] for n in names],np.float32).astype(float);m.actuator_gainprm[aids,0]=kp;m.actuator_biasprm[aids,1]=-kp;m.actuator_biasprm[aids,2]=-(kd+dt*kp)
poses=[]
for k in [-2,-1]:
    d.qpos[:7]=[0,0,0,1,0,0,0];d.qpos[qids]=tr['q'][k,0];mujoco.mj_kinematics(m,d);x=d.xpos[bids].copy();y=tr['body'][k,0];xc=x.mean(0);yc=y.mean(0);u,_,vt=np.linalg.svd((x-xc).T@(y-yc));r=vt.T@np.diag([1,1,np.linalg.det(vt.T@u.T)])@u.T;d.qpos[:7]=np.r_[yc-r@xc,np.roll(Rotation.from_matrix(r).as_quat(),1)];poses.append(d.qpos.copy())
mujoco.mj_differentiatePos(m,d.qvel,float(tr['t'][-1]-tr['t'][-2]),poses[0],poses[1]);d.qpos[:]=poses[1];d.qvel[vids]=tr['v'][-1,0];d.time=float(tr['t'][-1]);cache=CommandCache(names,.0000625,[row]);stats=[];start=time.perf_counter()
for k in range(round(.03/dt)):
    target,velocity=cache.get(round(d.time/.0000625));d.ctrl[aids]=target[0]+kd/kp*velocity[0];mujoco.mj_step(m,d);stats.append([int(d.solver_niter[0]),int(d.ncon),int(d.nefc)])
st=np.array(stats);out=dict(run=a.name,diagnostic_only=True,initial_time_s=float(tr['t'][-1]),duration_s=.03,dt=dt,initial_root_velocity='finite difference of fitted two recorded root poses; joint velocities exact recorded values',solver_iteration_max=int(st[:,0].max()),solver_iteration_p95=float(np.percentile(st[:,0],95)),solver_iteration_mean=float(st[:,0].mean()),configured_iteration_limit=int(m.opt.iterations),at_limit_steps=int(np.sum(st[:,0]>=m.opt.iterations)),contacts_range=[int(st[:,1].min()),int(st[:,1].max())],wall_s=time.perf_counter()-start,warnings=[int(w.number) for w in d.warning]);s.write(folder/'solver_state_diagnostic.json',out);print(json.dumps(out,indent=2))
