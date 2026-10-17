import argparse,json
import numpy as np,mujoco
import shared as s
p=argparse.ArgumentParser();p.add_argument('name');a=p.parse_args();f=s.ROOT/'runs'/a.name;meta=json.loads((f/'results.json').read_text());tr={k:v for k,v in np.load(f/'trajectory.npz').items()};names=tr['joint_names'].tolist();rows=[]
def score(label,t,q,v):
    iz=names.index('z');active=np.asarray(t)>=.005;return dict(label=label,mean_height_um=float(np.mean(q[active,iz]-meta['radius'])*1e6),min_height_um=float((q[active,iz]-meta['radius']).min()*1e6),max_height_um=float((q[active,iz]-meta['radius']).max()*1e6),normal_velocity_rms_mm_s=float(np.sqrt(np.mean(v[active,iz]**2))*1000))
for dt in [meta['dt'],meta['dt']/4]:
    m=mujoco.MjModel.from_xml_path(str(f/'probe.xml'));m.opt.timestep=dt;m.geom_solref[:,0]=2*dt;d=mujoco.MjData(m);ids=[m.joint(n).qposadr[0] for n in names];d.qpos[ids]=tr['q'][0];d.qvel[ids]=tr['v'][0];ix=m.joint('x').qposadr[0];iy=m.joint('y').qposadr[0];t=[0.];qs=[d.qpos[ids].copy()];vs=[d.qvel[ids].copy()]
    for k in range(round(.025/dt)):
        d.qpos[ix]=float(np.float32(k*dt*.03));d.qpos[iy]=0.;d.qvel[ix]=np.float32(.03);d.qvel[iy]=0.;mujoco.mj_step(m,d);t.append((k+1)*dt);qs.append(d.qpos[ids].copy());vs.append(d.qvel[ids].copy())
    q=np.array(qs);v=np.array(vs);rows.append(score('MuJoCo '+str(dt),np.array(t),q,v));np.savez_compressed(f/f'mu_{dt}.npz',t=t,q=q,v=v,joint_names=names)
rows.append(score('PhysX TGS prescribed XY',tr['t'],tr['q'],tr['v']))
s.write(f/'comparison.json',dict(rows=rows,interpretation='Prescribed 30 mm/s X boundary, zero Y; free vertical dynamics. XY state is overwritten before each step in both. Not a freely sliding or energy-conserving system; intended solely to compare normal response during sustained sliding.',source_sha=s.sha(__file__)));print(json.dumps(rows,indent=2))
