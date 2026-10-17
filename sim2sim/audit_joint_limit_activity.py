import json
import numpy as np
import shared as s
lookup={j['name']:j for j in s.M['joints']};rows=[]
for name in ['mu_coupled_rigid_dt31_case03_000','mu_box_velocity_dt31_case03_000','native_pgs_validation_cpu_dt62_000','native_pgs_validation_gpu_dt62_000']:
    tr={k:v for k,v in np.load(s.ROOT/'runs'/name/'trajectory.npz').items()};names=tr['joint_names'].tolist();limits=np.array([lookup[n]['range_rad'] for n in names]);q=tr['q'][tr['t']>=2]
    violation=np.maximum(limits[:,0]-q,q-limits[:,1]);margin=np.minimum(q-limits[:,0],limits[:,1]-q)
    active=np.any(margin<np.radians(.1),axis=(0,1));rows.append(dict(run=name,maximum_violation_deg=float(np.degrees(np.maximum(violation,0)).max()),minimum_limit_margin_deg=float(np.degrees(margin.min())),joints_within_point1_deg=[names[i] for i in np.flatnonzero(active)]))
s.write(s.ROOT/'audit/joint_limit_activity.json',dict(rows=rows,threshold_deg=.1,window_s=[2,6]));print(json.dumps(rows,indent=2))
