import json
import numpy as np
import mujoco
import shared as s
f=s.ROOT/'runs/contact_drop_slide_cpu_dt62_000';meta=json.loads((f/'results.json').read_text());rows=[];traces={}
for label,dt in [('dt250',.00025),('dt62',.0000625),('dt15',.000015625),('dt7',.0000078125)]:
    m=mujoco.MjModel.from_xml_path(str(f/'probe.xml'));m.opt.timestep=dt;d=mujoco.MjData(m);iz=m.joint('z').qposadr[0];ix=m.joint('x').qposadr[0];d.qpos[iz]=np.float32(meta['radius']+meta['initial_clearance']);d.qvel[ix]=np.float32(meta['initial_vx']);ts=[0.];qs=[d.qpos.copy()];vs=[d.qvel.copy()]
    for k in range(round(.03/dt)):mujoco.mj_step(m,d);ts.append((k+1)*dt);qs.append(d.qpos.copy());vs.append(d.qvel.copy())
    q=np.array(qs);v=np.array(vs);traces[label]=dict(t=ts,q=q,v=v);rows.append(dict(engine='mujoco',dt=dt,final_x=float(d.qpos[ix]),min_surface_z=float(q[:,iz].min()-meta['radius'])))
for name in ['contact_drop_slide_cpu_dt62_000','contact_drop_slide_cpu_dt62_friction1um_000']:
    p=s.ROOT/'runs'/name/'results.json'
    if p.exists():r=json.loads(p.read_text());rows.append(dict(engine='physx',name=name,dt=r['dt'],friction_offset=r.get('friction_offset',.00004),final_x=r['final_x'],min_surface_z=r['min_surface_z']))
arrays={f'{label}_{key}':value for label,tr in traces.items() for key,value in tr.items()};np.savez_compressed(s.ROOT/'audit/contact_probe_mujoco.npz',**arrays);s.write(s.ROOT/'audit/contact_probe_comparison.json',dict(rows=rows,ideal_impact_x=meta['initial_vx']*np.sqrt(2*meta['initial_clearance']/9.81),note='Normal impact impulse is sufficient to remove horizontal velocity at mu=0.5; expected rigid limit stops at impact.'))
print(json.dumps(rows,indent=2))
