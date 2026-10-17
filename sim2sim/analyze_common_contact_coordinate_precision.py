"""Independent origin and constant-velocity checks; never alter main gates."""
import json,numpy as np
import shared as s
R=s.ROOT
def read(name,file='trajectory.npz'):return {k:v for k,v in np.load(R/'runs'/name/file).items()}
def motion(t,e):
    names=t['body_names'].tolist();b=t['body'][:,e].astype(float);ids=[names.index(f'link_body_{i}') for i in range(21)];c=b[:,ids].mean(1)
    xy=(c[200:,:2]-c[200,:2])*1000
    d=b[:,names.index('link_body_20')]-b[:,names.index('link_body_0')];h=np.degrees(np.unwrap(np.arctan2(d[:,1],d[:,0])));h=h[200:]-h[0]
    return xy,h
gaits=[];translations=[];controls=[]
for grid,dt in [(31,.00003125),(15,.000015625)]:
    name=f'native_common_contact_origin_probe_cpu_dt{grid}_000'
    if (R/'runs'/name/'complete.json').exists():
        p=read(name);c=read(f'native_common_contact_validation_cpu_dt{grid}_000');m=read(f'mu_common_contact_dt{grid}_case06_000');mxy,mh=motion(m,0)
        controls.append(dict(grid_us=dt*1e6,bitwise_identical=all(np.array_equal(p[k][:,1],c[k][:,7]) for k in ['q','v','body'])))
        for e,y in enumerate([0.,1.75]):
            xy,h=motion(p,e);d=xy-mxy
            gaits.append(dict(grid_us=dt*1e6,global_y_m=y,forward_displacement_mm=float(-xy[-1,0]),lateral_displacement_mm=float(xy[-1,1]),mu_forward_displacement_mm=float(-mxy[-1,0]),mu_lateral_displacement_mm=float(mxy[-1,1]),max_xy_difference_mm=float(np.linalg.norm(d,axis=1).max()),max_lateral_difference_mm=float(abs(d[:,1]).max()),max_heading_difference_deg=float(abs(h-mh).max())))
    name=f'native_common_contact_translation_cpu_dt{grid}_001'
    if (R/'runs'/name/'complete.json').exists():
        p=read(name,'translation_probe.npz');q=read(name);res=json.loads((R/'runs'/name/'results.json').read_text());setup=json.loads((R/'runs'/name/'translation_probe_setup.json').read_text())
        assert setup['gravity']==[0,0,0] and res['max_contact_point_force_N']==0
        for e,y in enumerate([0.,1.75]):
            v=p['velocity'][0,e,1];position=np.float32(y);history=[float(position)];stride=round(.01/dt)
            for k in range(round(.5/dt)):
                position=np.float32(position+np.float32(v*np.float32(dt)))
                if (k+1)%stride==0:history.append(float(position))
            measured=(p['position'][:,e,1].astype(float)-p['position'][0,e,1])*1000;prediction=(np.array(history)-y)*1000
            translations.append(dict(grid_us=dt*1e6,global_y_m=y,expected_displacement_mm=1.5,measured_displacement_mm=float(measured[-1]),float32_repeated_addition_displacement_mm=float(prediction[-1]),maximum_prediction_trace_error_mm=float(abs(measured-prediction).max()),maximum_velocity_change_m_s=float(abs(p['velocity'][:,e]-p['velocity'][0,e]).max()),maximum_contact_point_force_N=res['max_contact_point_force_N'],max_abs_joint_position_rad=float(abs(q['q'][:,e]).max()),ulp_at_initial_y_nm=float(np.spacing(np.float32(y))*1e9)))
out=dict(case='4.5/A',gait_window_s=[2,6],translation_window_s=[0,.5],control_checks=controls,gait_rows=gaits,translation_rows=translations,adopted=False,qualification='Selected causal diagnostics, not replacements for the original bank. Constant-velocity prediction is independently calculated from measured initial velocity using repeated float32 addition. Origin dependence does not establish the unique cause of historical native-contact gaps.',source_sha=s.sha(__file__))
s.write(R/'audit/common_contact_coordinate_precision.json',out);print(json.dumps(out))
