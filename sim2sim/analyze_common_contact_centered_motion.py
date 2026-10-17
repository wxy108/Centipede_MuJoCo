"""Compare recorded whole-body motion as well as final displacement."""
import csv,json,numpy as np
import shared as s
R=s.ROOT
def read(name):return {k:v for k,v in np.load(R/'runs'/name/'trajectory.npz').items()}
def features(tr,e):
    bn=tr['body_names'].tolist();jn=tr['joint_names'].tolist();b=tr['body'][:,e].astype(float);q=tr['q'][:,e].astype(float);ids=[bn.index(f'link_body_{i}') for i in range(21)];xyz=b[:,ids].mean(1);xy=(xyz[200:,:2]-xyz[200,:2])*1000
    chord=b[:,bn.index('link_body_20')]-b[:,bn.index('link_body_0')];heading=np.unwrap(np.arctan2(chord[:,1],chord[:,0]));heading=np.degrees(heading[200:]-heading[0])
    amp={}
    for label,prefix in [('body_yaw','body_yaw_'),('leg_yaw','legbase_yaw_'),('leg_pitch','legbase_pitch_')]:
        ids=[i for i,n in enumerate(jn) if n.startswith(prefix) and n!='body_yaw_1'];amp[label]=float(np.degrees(np.ptp(q[200:,ids],axis=0)/2).max())
    return dict(xy=xy,heading=heading,q=q[200:],amplitudes=amp,jn=jn)
rows=[]
for grid in [31,15]:
    native={d:read(f'native_common_contact_centered_validation_{d}_dt{grid}_000') for d in ['cpu','gpu'] if (R/f'runs/native_common_contact_centered_validation_{d}_dt{grid}_000/complete.json').exists()}
    for i,c in enumerate(s.cases('holdout')):
        mu=features(read(f'mu_common_contact_dt{grid}_case{i:02d}_000'),0)
        for device,tr in native.items():
            n=features(tr,i+1);order=[mu['jn'].index(k) for k in n['jn']];delta=n['xy']-mu['xy'];da={k:n['amplitudes'][k]-mu['amplitudes'][k] for k in mu['amplitudes']}
            rows.append(dict(device=device,grid_us=31.25 if grid==31 else 15.625,case=i,wave=c['wave'],family='AB'[i%2],max_forward_trace_difference_mm=float(abs(delta[:,0]).max()),max_lateral_trace_difference_mm=float(abs(delta[:,1]).max()),max_centroid_xy_path_difference_mm=float(np.linalg.norm(delta,axis=1).max()),max_heading_difference_deg=float(abs(n['heading']-mu['heading']).max()),max_joint_position_difference_deg=float(np.degrees(abs(n['q']-mu['q'][:,order])).max()),body_yaw_amplitude_difference_deg=da['body_yaw'],leg_yaw_amplitude_difference_deg=da['leg_yaw'],leg_pitch_amplitude_difference_deg=da['leg_pitch'],native_lateral_displacement_mm=float(n['xy'][-1,1]),mu_lateral_displacement_mm=float(mu['xy'][-1,1])))
s.write(R/'audit/common_contact_centered_motion_comparison.json',dict(rows=rows,window_s=[2,6],definitions=dict(path='Unweighted 21-body centroid, relative to its position at 2 s',heading='Planar link_body_0 to link_body_20 chord angle, unwrapped and relative to its initial angle; no claim that it is a rigid root orientation',amplitudes='Maximum per-joint half peak-to-peak over 2–6 s in each named active family; excludes fixed-command body_yaw_1'),source_sha=s.sha(__file__)))
with (R/'common_contact_centered_motion_comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
keys=['max_centroid_xy_path_difference_mm','max_heading_difference_deg','max_joint_position_difference_deg']
print(json.dumps({d:{k:max(r[k] for r in rows if r['device']==d) for k in keys} for d in ['cpu','gpu'] if any(r['device']==d for r in rows)}))
