"""Measure origin dependence without replacing the frozen holdout bank."""
import json,numpy as np
import shared as s
R=s.ROOT
def read(name):return {k:v for k,v in np.load(R/'runs'/name/'trajectory.npz').items()}
probe=read('native_common_contact_origin_probe_cpu_dt31_000')
control=read('native_common_contact_validation_cpu_dt31_000')
mu=read('mu_common_contact_dt31_case06_000')
errors={k:float(abs(probe[k][:,1].astype(float)-control[k][:,7].astype(float)).max()) for k in ['q','v','body']}
def motion(t,e):
    names=t['body_names'].tolist();b=t['body'][:,e].astype(float);ids=[names.index(f'link_body_{i}') for i in range(21)];c=b[:,ids].mean(1)
    xy=(c[200:,:2]-c[200,:2])*1000
    d=b[:,names.index('link_body_20')]-b[:,names.index('link_body_0')];h=np.degrees(np.unwrap(np.arctan2(d[:,1],d[:,0])));h=h[200:]-h[0]
    return xy,h
mxy,mh=motion(mu,0);rows=[]
for e,y in enumerate([0.,1.75]):
    xy,h=motion(probe,e);dif=xy-mxy
    rows.append(dict(global_y_m=y,forward_displacement_mm=float(-xy[-1,0]),lateral_displacement_mm=float(xy[-1,1]),mu_forward_displacement_mm=float(-mxy[-1,0]),mu_lateral_displacement_mm=float(mxy[-1,1]),max_xy_difference_mm=float(np.linalg.norm(dif,axis=1).max()),max_lateral_difference_mm=float(abs(dif[:,1]).max()),max_heading_difference_deg=float(abs(h-mh).max())))
out=dict(full_duration_s=float(probe['t'][-1]),control_errors=errors,control_bitwise_identical=all(np.array_equal(probe[k][:,1],control[k][:,7]) for k in errors),case='4.5/A',rows=rows,adopted=False,qualification='Only world Y changes in this complete gait diagnostic. This does not replace the fixed bank or prove the sole cause of old native-contact differences.',source_sha=s.sha(__file__))
s.write(R/'audit/common_contact_origin_comparison.json',out);print(json.dumps(out))
