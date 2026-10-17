import argparse,json
import numpy as np,mujoco
import shared as s
p=argparse.ArgumentParser();p.add_argument('name');a=p.parse_args();folder=s.ROOT/'runs'/a.name
meta=json.loads((folder/'results.json').read_text());tr={k:v for k,v in np.load(folder/'trajectory.npz').items()};native_dt=meta['arguments']['dt'];rows=[]
for dt in [.00003125,.000015625]:
    stride=round(native_dt/dt);assert abs(stride*dt-native_dt)<1e-12
    for i,c in enumerate(meta['rows']):
        m=mujoco.MjModel.from_xml_path(str(folder/(c['family']+'.xml')));m.opt.timestep=dt;m.opt.integrator=mujoco.mjtIntegrator.mjINT_EULER;m.dof_armature[:]=dt*c['kv']+dt**2*c['kp'];m.actuator_biasprm[:,2]=-(c['kv']+dt*c['kp']);mujoco.mj_setConst(m,mujoco.MjData(m));d=mujoco.MjData(m);q=[0.]
        for k in range((len(tr['t'])-1)*stride):
            command_time=(k//stride)*native_dt;d.ctrl[0]=float(np.float32(np.radians(1.) if command_time>=.01 else 0.));mujoco.mj_step(m,d)
            if (k+1)%stride==0:q.append(float(d.qpos[0]))
        q=np.array(q);denom=max(float(np.ptp(tr['ideal'][:,i])),1e-12);rms=float(np.sqrt(np.mean((q-tr['native'][:,i])**2))/denom)
        rows.append(dict(family=c['family'],mujoco_dt_s=dt,native_dt_s=native_dt,normalized_rms_error=rms,maximum_error_deg=float(np.degrees(np.abs(q-tr['native'][:,i])).max())))
s.write(folder/'coupled_timegrid_comparison.json',dict(rows=rows,passed=all(r['normalized_rms_error']<=.02 for r in rows),source_sha=s.sha(__file__)));print(json.dumps(dict(passed=all(r['normalized_rms_error']<=.02 for r in rows),maximum_normalized_rms=max(r['normalized_rms_error'] for r in rows)),indent=2))
