import json
import numpy as np,mujoco
import shared as s
from desaxce_contact import step
f=s.ROOT/'runs/prescribed_slide_tgs_cpu_dt62_000';tr={k:v for k,v in np.load(f/'trajectory.npz').items()};names=tr['joint_names'].tolist();rows=[]
for label,dt,maxit in [('native_mj_step',.0000625,None),('manual_zero',.0000625,0),('desaxce20',.0000625,20),('desaxce40',.0000625,40),('desaxce20_fine',.000015625,20)]:
    m=mujoco.MjModel.from_xml_path(str(f/'probe.xml'));m.opt.integrator=mujoco.mjtIntegrator.mjINT_EULER;m.opt.timestep=dt;m.geom_solref[:,0]=2*dt;d=mujoco.MjData(m);ids=[m.joint(n).qposadr[0] for n in names];d.qpos[ids]=tr['q'][0];d.qvel[ids]=tr['v'][0];ix=m.joint('x').qposadr[0];iy=m.joint('y').qposadr[0];iz=names.index('z');ts=[0.];qs=[d.qpos[ids].copy()];vs=[d.qvel[ids].copy()];maxres=0.;fail=0;mi=0
    for k in range(round(.025/dt)):
        d.qpos[ix]=float(np.float32(k*dt*.03));d.qpos[iy]=0.;d.qvel[ix]=np.float32(.03);d.qvel[iy]=0.
        if maxit is None:mujoco.mj_step(m,d)
        else:
            r=step(m,d,maxit);maxres=max(maxres,r['residual_m_s']);mi=max(mi,r['iterations']);fail+=not r['converged']
        ts.append((k+1)*dt);qs.append(d.qpos[ids].copy());vs.append(d.qvel[ids].copy())
    q=np.array(qs);v=np.array(vs);active=np.array(ts)>=.005;rows.append(dict(label=label,mean_height_um=float((q[active,iz]-.00045).mean()*1e6),max_height_um=float((q[active,iz]-.00045).max()*1e6),normal_velocity_rms_mm_s=float(np.sqrt(np.mean(v[active,iz]**2))*1000),max_correction_residual_m_s=maxres,max_iterations=mi,unconverged_steps=fail,final_physics_time_s=d.time))
    np.savez_compressed(f/f'{label}.npz',t=ts,q=q,v=v,joint_names=names)
s.write(f/'desaxce_diagnostic.json',dict(rows=rows,source_sha=s.sha(__file__),adapter_sha=s.sha(s.ROOT/'desaxce_contact.py')));print(json.dumps(rows,indent=2))
