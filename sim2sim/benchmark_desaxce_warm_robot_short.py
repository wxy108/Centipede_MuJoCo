import json,time
import numpy as np,mujoco
import shared as s
from desaxce_contact import step as cold
from desaxce_contact_warm import step as warm
m=mujoco.MjModel.from_xml_path(str(s.ROOT/'assets/common.xml'));dt=.0000625;m.opt.timestep=dt;m.opt.integrator=mujoco.mjtIntegrator.mjINT_EULER;m.geom_solref[:,0]=2*dt
names=[m.joint(i).name for i in range(1,m.njnt)];lookup={x['name']:x for x in s.M['joints']};dids=np.array([m.joint(n).dofadr[0] for n in names]);ids=np.array([m.actuator(n).id for n in names]);kp=np.array([lookup[n]['kp'] for n in names],np.float32).astype(float);kd=np.array([lookup[n]['kv'] for n in names],np.float32).astype(float);m.actuator_gainprm[ids,0]=kp;m.actuator_biasprm[ids,1]=-kp;m.actuator_biasprm[ids,2]=-(kd+dt*kp);m.dof_armature[dids]=dt*kd+dt*dt*kp;mujoco.mj_setConst(m,mujoco.MjData(m));allq=[];rows=[]
for label,fn in [('cold',cold),('warm',warm)]:
    d=mujoco.MjData(m);d.qpos[:7]=s.M['initial_root_pose_wxyz'];d.time=1.125;mujoco.mj_forward(m,d);qs=[];rs=[];start=time.monotonic()
    for k in range(160):
        q,v=s.targets(names,1.125+k*dt,dt,[s.cases('holdout')[3]]);d.ctrl[ids]=q[0]+kd/kp*v[0];rs.append(fn(m,d));qs.append(d.qpos.copy())
    rows.append(dict(label=label,steps=160,wall_s=time.monotonic()-start,unconverged_steps=sum(not r['converged'] for r in rs),max_residual_m_s=max(r['residual_m_s'] for r in rs),mean_iterations=np.mean([r['iterations'] for r in rs])));allq.append(np.array(qs))
out=dict(rows=rows,max_recorded_qpos_difference=float(np.max(np.abs(allq[0]-allq[1]))),source_sha=s.sha(__file__),interpretation='Short full-robot active-ramp stress test, not gait acceptance. Compares only nonlinear-solver initialization/warmstart.')
s.write(s.ROOT/'audit/desaxce_warm_robot_short.json',out);print(json.dumps(out,indent=2))

