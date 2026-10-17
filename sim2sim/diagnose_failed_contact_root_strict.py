"""Solve one saved unresolved contact equation; no full-gait continuation."""
import json,time
import numpy as np,mujoco
from scipy.optimize import root,least_squares
import shared as s
from command_cache import CommandCache
f=s.ROOT/'runs/mu_desaxceAA_cmd250_dt125_case03_000';err=json.loads((f/'error.json').read_text());state={k:v for k,v in np.load(f/'nonconverged_state.npz').items()};h=.000125
m=mujoco.MjModel.from_xml_path(str(s.ROOT/'assets/common.xml'));m.opt.tolerance=1e-20;m.opt.ls_tolerance=1e-5;m.opt.iterations=500;m.opt.timestep=h;m.opt.integrator=mujoco.mjtIntegrator.mjINT_EULER;m.geom_solref[:,0]=2*h
names=[m.joint(i).name for i in range(1,m.njnt)];lookup={x['name']:x for x in s.M['joints']};di=np.array([m.joint(n).dofadr[0] for n in names]);ai=np.array([m.actuator(n).id for n in names]);kp=np.array([lookup[n]['kp'] for n in names],np.float32).astype(float);kd=np.array([lookup[n]['kv'] for n in names],np.float32).astype(float);m.actuator_gainprm[ai,0]=kp;m.actuator_biasprm[ai,1]=-kp;m.actuator_biasprm[ai,2]=-(kd+h*kp);m.dof_armature[di]=h*kd+h*h*kp;mujoco.mj_setConst(m,mujoco.MjData(m))
d=mujoco.MjData(m);d.qpos[:]=state['qpos'];mujoco.mj_integratePos(m,d.qpos,state['qvel'],-h);d.qvel[:]=state['qvel']-h*state['qacc'];d.qacc_warmstart[:]=state['qacc'];d.time=err['time_s']-h
cache=CommandCache(names,.00025,[s.cases('holdout')[3]]);q,v=cache.get((err['step']-1)//2);d.ctrl[ai]=q[0]+kd/kp*v[0];mujoco.mj_step1(m,d);mujoco.mj_fwdActuation(m,d);mujoco.mj_fwdAcceleration(m,d)
con=[c for c in d.contact if c.efc_address>=0 and c.dim==3];normal=np.array([c.efc_address for c in con]);tan=normal[:,None]+[1,2];mu=np.array([c.friction[0] for c in con]);base=d.efc_aref.copy();ja=np.zeros(d.nefc);mujoco.mj_mulJacVec(m,d,ja,d.qacc_smooth);initial=h*mu*np.linalg.norm((ja-base)[tan],axis=1);rows=[];solutions=[];calls=0
def fun(x):
    global calls
    calls+=1;d.efc_aref[:]=base;d.efc_aref[normal]-=x/h;mujoco.mj_fwdConstraint(m,d);mujoco.mj_mulJacVec(m,d,ja,d.qacc);alpha=ja-base+d.efc_R*d.efc_force;out=x-h*mu*np.linalg.norm(alpha[tan],axis=1);d.qacc_warmstart[:]=d.qacc;return out.copy()
for method in ['hybr','lm','least_squares']:
    calls=0;start=time.monotonic()
    if method=='least_squares':result=least_squares(fun,initial,bounds=(0,np.inf),xtol=1e-12,ftol=1e-12,gtol=1e-12,diff_step=1e-4,max_nfev=1000)
    else:result=root(fun,initial,method=method,options={'eps':1e-8,**({'maxfev':2000,'xtol':1e-9} if method=='hybr' else {'maxiter':2000,'ftol':1e-12,'xtol':1e-12})})
    residual=fun(result.x);row=dict(method=method,contacts=len(con),success_flag=bool(result.success),max_residual_m_s=float(np.max(np.abs(residual))),min_correction_m_s=float(result.x.min()),calls=calls,wall_s=time.monotonic()-start,message=str(result.message));rows.append(row);solutions.append(result.x.copy());print(json.dumps(row),flush=True)
s.write(s.ROOT/'audit/desaxce_failed_frame_strict_inner.json',dict(rows=rows,source_run=f.name,source_sha=s.sha(__file__),reconstruction='Reverse the final semi-implicit Euler position update using post-step velocity; reverse velocity using saved acceleration. This recovers the pre-step physical state, but does not claim identical solver warmstart.'))
np.savez_compressed(s.ROOT/'audit/desaxce_failed_frame_strict_roots.npz',initial=initial,solutions=solutions,normal_rows=normal)

