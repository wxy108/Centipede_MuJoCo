import argparse,json
import numpy as np
import mujoco
import shared as s
p=argparse.ArgumentParser();p.add_argument('name');a=p.parse_args();folder=s.ROOT/'runs'/a.name
meta=json.loads((folder/'results.json').read_text());tr=np.load(folder/'trajectory.npz');dt=meta['arguments']['dt'];mu=np.zeros((len(tr['t']),len(meta['rows']),2));rows=[]
for i,c in enumerate(meta['rows']):
    for j,adapter in enumerate([False,True]):
        m=mujoco.MjModel.from_xml_path(str(folder/(c['family']+'.xml')));d=mujoco.MjData(m)
        if adapter:m.actuator_biasprm[0,2]-=dt*c['kp']
        for k in range(len(tr['t'])-1):
            d.ctrl[0]=float(np.float32(np.radians(1.) if k*dt>=.01 else 0.));mujoco.mj_step(m,d);mu[k+1,i,j]=d.qpos[0]
    denom=max(float(np.ptp(tr['ideal'][:,i])),1e-12)
    rows.append(dict(**c,mu_analytic_nrmse=float(np.sqrt(np.mean((mu[:,i,0]-tr['ideal'][:,i])**2))/denom),mu_native_nrmse=float(np.sqrt(np.mean((mu[:,i,0]-tr['native'][:,i])**2))/denom),adapter_native_nrmse=float(np.sqrt(np.mean((mu[:,i,1]-tr['native'][:,i])**2))/denom)))
np.savez_compressed(folder/'paired_trajectory.npz',t=tr['t'],native=tr['native'],ideal=tr['ideal'],mu=mu)
s.write(folder/'comparison.json',dict(rows=rows,source_sha=s.sha(__file__)));print(json.dumps(rows,indent=2))
