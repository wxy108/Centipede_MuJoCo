import json
import numpy as np,mujoco
import shared as s
f=s.ROOT/'runs/prescribed_slide_tgs_cpu_dt62_000';tr={k:v for k,v in np.load(f/'trajectory.npz').items()};names=tr['joint_names'].tolist();rows=[]
variants=[('baseline',1,0),('noslip20',1,20),('impratio100',100,0),('impratio10000',10000,0)]
for label,ratio,ns in variants:
    dt=.0000625;m=mujoco.MjModel.from_xml_path(str(f/'probe.xml'));m.opt.timestep=dt;m.geom_solref[:,0]=2*dt;m.opt.impratio=ratio;m.opt.noslip_iterations=ns;m.opt.noslip_tolerance=1e-12;d=mujoco.MjData(m);ids=[m.joint(n).qposadr[0] for n in names];d.qpos[ids]=tr['q'][0];d.qvel[ids]=tr['v'][0];ix=m.joint('x').qposadr[0];iy=m.joint('y').qposadr[0];iz=names.index('z');qs=[];vs=[]
    for k in range(round(.025/dt)):
        d.qpos[ix]=float(np.float32(k*dt*.03));d.qpos[iy]=0.;d.qvel[ix]=np.float32(.03);d.qvel[iy]=0.;mujoco.mj_step(m,d)
        if (k+1)*dt>=.005:qs.append(d.qpos[ids].copy());vs.append(d.qvel[ids].copy())
    q=np.array(qs);v=np.array(vs);rows.append(dict(label=label,impratio=ratio,noslip_iterations=ns,mean_height_um=float((q[:,iz]-.00045).mean()*1e6),max_height_um=float((q[:,iz]-.00045).max()*1e6),normal_velocity_rms_mm_s=float(np.sqrt(np.mean(v[:,iz]**2))*1000)))
s.write(f/'numerics_ablation.json',dict(rows=rows,source_sha=s.sha(__file__)));print(json.dumps(rows,indent=2))
