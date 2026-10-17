import json
import numpy as np
import shared as s
mf=s.ROOT/'runs/mu_no_contact_case03_dt62_000';nf=s.ROOT/'runs/native_no_contact_cpu_case03_dt62_000'
m={k:v for k,v in np.load(mf/'trajectory.npz').items()};n={k:v for k,v in np.load(nf/'trajectory.npz').items()}
assert np.array_equal(m['t'],n['t'])
names=m['joint_names'].tolist();idx=[n['joint_names'].tolist().index(k) for k in names]
err=n['q'][:,0,idx].astype(float)-m['q'][:,0];active=m['t']>=1.25;rows=[]
for family in ['body_yaw','body_pitch','body_roll','legbase_yaw','legbase_pitch','tarsus_yaw','tarsus_pitch','tarsus_roll']:
    js=[j for j,k in enumerate(names) if k.startswith(family+'_')]
    if js:rows.append(dict(family=family,rms_error_deg=float(np.degrees(np.sqrt(np.mean(err[active][:,js]**2)))),max_error_deg=float(np.degrees(np.max(np.abs(err[active][:,js]))))))
bn=m['body_names'].tolist();bi=[n['body_names'].tolist().index(k) for k in bn];native_body=n['body'][:,0,bi].astype(float)
initial=np.array(json.loads((nf/'readback.json').read_text())['initial_root_pose'][0])[:3]
native_body-=initial-np.array(s.M['initial_root_pose_wxyz'][:3])
body_error=native_body-m['body'][:,0];worst=np.unravel_index(np.abs(err).argmax(),err.shape)
out=dict(gait=s.cases('holdout')[3],same_dt_s=.0000625,gravity=0,ground_collision=False,
    rows=rows,max_all_joint_error_deg=float(np.degrees(np.abs(err).max())),worst_joint=names[worst[1]],worst_time_s=float(m['t'][worst[0]]),
    maximum_body_origin_error_m=float(np.linalg.norm(body_error,axis=2).max()),
    rms_body_origin_error_m=float(np.sqrt(np.mean(body_error**2))),
    interpretation='Two-second mechanical diagnostic, not a six-second locomotion score; only the known initial environment offset was removed.',source_sha=s.sha(__file__))
s.write(s.ROOT/'audit/no_contact_chain_comparison.json',out);print(json.dumps(out,indent=2))
