import json,numpy as np,mujoco
import shared as s
f=s.ROOT/'runs/oblique_slide_pgs_cpu_dt62_000';meta=json.loads((f/'results.json').read_text());tr=np.load(f/'trajectory.npz');names=tr['joint_names'].tolist();rows=[];arrays={}
for label,dt in [('dt62',.0000625),('dt15',.000015625)]:
    m=mujoco.MjModel.from_xml_path(str(f/'probe.xml'));m.opt.timestep=dt;m.geom_solref[:,0]=2*dt;d=mujoco.MjData(m)
    for k,n in enumerate(names):d.qpos[m.joint(n).qposadr[0]]=tr['q'][0,k];d.qvel[m.joint(n).dofadr[0]]=tr['v'][0,k]
    ids=[m.joint(n).qposadr[0] for n in names];ts=[0.];qs=[d.qpos[ids].copy()];vs=[d.qvel[ids].copy()]
    for k in range(round(.025/dt)):
        mujoco.mj_step(m,d);ts.append((k+1)*dt);qs.append(d.qpos[ids].copy());vs.append(d.qvel[ids].copy())
    arrays.update({f'{label}_t':ts,f'{label}_q':qs,f'{label}_v':vs})
    rows.append(dict(engine='MuJoCo',dt=dt,final_x_m=float(d.qpos[m.joint('x').qposadr[0]]),final_y_m=float(d.qpos[m.joint('y').qposadr[0]])))
rows.append(dict(engine='PhysX PGS',dt=meta['dt'],final_x_m=meta['final_x'],final_y_m=meta['final_y']))
np.savez_compressed(f/'mu_trajectory.npz',**arrays)
ideal=.05**2/(2*.5*9.81);s.write(f/'comparison.json',dict(rows=rows,approx_ideal_massless_carrier_xy_m=[ideal*np.cos(np.pi/6),ideal*np.sin(np.pi/6)],interpretation='Independent oblique sliding diagnostic; do not attribute whole-robot discrepancy to directional friction based on an untested hypothesis.'));print(json.dumps(rows,indent=2))
