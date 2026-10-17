import json
import numpy as np
import mujoco
import shared as s
f=s.ROOT/'runs/contact_roll_pgs_cpu_dt62_000';native=json.loads((f/'results.json').read_text());rows=[];traces={}
for label,dt in [('dt62',.0000625),('dt31',.00003125),('dt15',.000015625)]:
    m=mujoco.MjModel.from_xml_path(str(f/'probe.xml'));m.opt.timestep=dt;m.geom_solref[:,0]=2*dt
    d=mujoco.MjData(m);iz=m.joint('z').qposadr[0];ix=m.joint('x').qposadr[0];ir=m.joint('roll').qposadr[0]
    d.qpos[iz]=np.float32(native['radius']+native['initial_clearance']);d.qvel[ix]=np.float32(native['initial_vx'])
    ts=[0.];qs=[d.qpos.copy()];vs=[d.qvel.copy()]
    for k in range(round(.03/dt)):
        mujoco.mj_step(m,d);ts.append((k+1)*dt);qs.append(d.qpos.copy());vs.append(d.qvel.copy())
    rows.append(dict(engine='mujoco',dt=dt,final_x=float(d.qpos[ix]),final_vx=float(d.qvel[ix]),final_roll_velocity=float(d.qvel[ir]),surface_slip_velocity=float(d.qvel[ix]-native['radius']*d.qvel[ir])))
    traces.update({label+'_t':ts,label+'_q':qs,label+'_v':vs})
native['engine']='physx_cpu';native['surface_slip_velocity']=native['final_vx']-native['radius']*native['final_roll_velocity'];rows.append(native)
np.savez_compressed(s.ROOT/'audit/roll_probe_mujoco.npz',**traces)
s.write(s.ROOT/'audit/roll_contact_probe_comparison.json',dict(rows=rows,ideal_rolling_vx=native['initial_vx']*(native['mass']+native['carrier_mass'])/(native['mass']+native['carrier_mass']+native['inertia']/native['radius']**2),source_sha=s.sha(__file__)))
print(json.dumps(rows,indent=2))
