import json
import numpy as np,mujoco
import shared as s
from desaxce_contact import step
rows=[]
# Verify manual split pipeline on a coupled, full robot with active gait commands.
m=mujoco.MjModel.from_xml_path(str(s.ROOT/'assets/common.xml'));dt=.0000625;m.opt.timestep=dt;m.opt.integrator=mujoco.mjtIntegrator.mjINT_EULER;m.geom_solref[:,0]=2*dt
names=[m.joint(i).name for i in range(1,m.njnt)];lookup={x['name']:x for x in s.M['joints']};dids=np.array([m.joint(n).dofadr[0] for n in names]);ids=np.array([m.actuator(n).id for n in names]);kp=np.array([lookup[n]['kp'] for n in names],np.float32).astype(float);kd=np.array([lookup[n]['kv'] for n in names],np.float32).astype(float);m.actuator_gainprm[ids,0]=kp;m.actuator_biasprm[ids,1]=-kp;m.actuator_biasprm[ids,2]=-(kd+dt*kp);m.dof_armature[dids]=dt*kd+dt*dt*kp;mujoco.mj_setConst(m,mujoco.MjData(m))
a=mujoco.MjData(m);b=mujoco.MjData(m)
for d in [a,b]:d.qpos[:7]=s.M['initial_root_pose_wxyz'];d.time=1.125;mujoco.mj_forward(m,d)
mx=np.zeros(3)
for k in range(1000):
    q,v=s.targets(names,1.125+k*dt,dt,[s.cases('holdout')[3]])
    for d in [a,b]:d.ctrl[ids]=q[0]+kd/kp*v[0]
    mujoco.mj_step(m,a);step(m,b,0)
    mx=np.maximum(mx,[np.max(np.abs(a.qpos-b.qpos)),np.max(np.abs(a.qvel-b.qvel)),np.max(np.abs(a.qfrc_constraint-b.qfrc_constraint))])
rows.append(dict(test='full_robot_manual_zero_equivalence',steps=1000,max_qpos_error=float(mx[0]),max_qvel_error=float(mx[1]),max_constraint_force_error=float(mx[2]),passed=bool(np.max(mx)<1e-12)))
for name,kind in [('tgs_drop_external_each_iter_cpu_000','drop'),('tgs_friction_angle45_default_cpu_000','force45'),('oblique_slide_pgs_cpu_dt62_000','free_slide')]:
    f=s.ROOT/'runs'/name;meta=json.loads((f/'results.json').read_text());tr={k:v for k,v in np.load(f/'trajectory.npz').items()};names=tr['joint_names'].tolist();m=mujoco.MjModel.from_xml_path(str(f/'probe.xml'));dt=float(meta['dt']);m.opt.integrator=mujoco.mjtIntegrator.mjINT_EULER;m.geom_solref[:,0]=2*dt;d=mujoco.MjData(m);ids=[m.joint(n).qposadr[0] for n in names];d.qpos[ids]=tr['q'][0];d.qvel[ids]=tr['v'][0]
    if kind=='force45':d.qfrc_applied[ids]=meta['applied_force_N']
    qs=[d.qpos[ids].copy()];vs=[d.qvel[ids].copy()];fail=0;maxres=0.
    for k in range(len(tr['t'])-1):
        r=step(m,d);maxres=max(maxres,r['residual_m_s']);fail+=not r['converged'];qs.append(d.qpos[ids].copy());vs.append(d.qvel[ids].copy())
    q=np.array(qs);v=np.array(vs);ix=names.index('x');iz=names.index('z');r=dict(test=kind,native_run=name,final_x_um=float(q[-1,ix]*1e6),native_final_x_um=float(tr['q'][-1,ix]*1e6),max_normal_velocity_mm_s=float(np.max(np.abs(v[:,iz]))*1000),unconverged_steps=fail,max_correction_residual_m_s=maxres)
    if 'y' in names:r.update(final_y_um=float(q[-1,names.index('y')]*1e6),native_final_y_um=float(tr['q'][-1,names.index('y')]*1e6))
    rows.append(r);np.savez_compressed(f/'desaxce20.npz',t=tr['t'],q=q,v=v,joint_names=names)
s.write(s.ROOT/'audit/desaxce_mechanical_suite.json',dict(rows=rows,source_sha=s.sha(__file__),adapter_sha=s.sha(s.ROOT/'desaxce_contact.py')));print(json.dumps(rows,indent=2))
