import argparse,json
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
import shared as s
p=argparse.ArgumentParser();p.add_argument('name');a=p.parse_args();f=s.ROOT/'runs'/a.name;meta=json.loads((f/'readback.json').read_text());b={k:v for k,v in np.load(f/'backend.npz').items()};m=mujoco.MjModel.from_xml_path(str(s.ROOT/'assets/common.xml'));lookup={x['name']:x for x in s.M['joints']};rows=[]
for e in range(len(b['masses'])):
    mass_err=0.;inertia_err=0.;com_err=0.;kp_err=0.;kd_err=0.
    for k,n in enumerate(meta['body_names']):
        body=m.body(n);r=Rotation.from_quat(np.roll(body.iquat,-1)).as_matrix();I=r@np.diag(body.inertia)@r.T;mass_err=max(mass_err,float(abs(b['masses'][e,k]-body.mass[0])/body.mass[0]));inertia_err=max(inertia_err,float(np.linalg.norm(b['inertias'][e,k].reshape(3,3)-I)/np.linalg.norm(I)));com_err=max(com_err,float(np.linalg.norm(b['coms'][e,k,:3]-body.ipos)))
    for k,n in enumerate(meta['joint_names']):
        j=lookup[n];kp_err=max(kp_err,float(abs(b['dof_stiffnesses'][e,k]-j['kp'])/j['kp']));kd_err=max(kd_err,float(abs(b['dof_dampings'][e,k]-j['kv'])/j['kv']))
    rows.append(dict(env=e,mass_rel=mass_err,inertia_rel=inertia_err,com_error_m=com_err,kp_rel=kp_err,kd_rel=kd_err))
checks=dict(materials=np.unique(b['material_properties'].reshape(-1,3),axis=0).tolist(),contact_offsets=np.unique(b['contact_offsets']).tolist(),rest_offsets=np.unique(b['rest_offsets']).tolist(),armatures=np.unique(b['dof_armatures']).tolist(),joint_friction=np.unique(b['dof_friction_coefficients']).tolist(),force_caps=np.unique(b['dof_max_forces']).tolist(),velocity_caps=np.unique(b['dof_max_velocities']).tolist(),solver_type=meta.get('solver_type'),dt=meta['physics_dt_s'])
passed=all(r['mass_rel']<=1e-5 and r['inertia_rel']<=.001 and r['com_error_m']<=1e-6 and r['kp_rel']<=1e-6 and r['kd_rel']<=1e-6 for r in rows) and checks['materials']==[[.5,.5,0.]] and checks['armatures']==[0.] and checks['joint_friction']==[0.] and checks['force_caps']==[1.] and checks['velocity_caps']==[1e6] and checks['rest_offsets']==[0.] and np.allclose(checks['contact_offsets'],[2e-5],rtol=1e-6,atol=0) and checks['solver_type']==1 and json.loads((f/'tgs_external_iteration.json').read_text())['enabled'] is True
s.write(f/'backend_validation.json',dict(passed=passed,rows=rows,checks=checks));print(json.dumps(dict(passed=passed,envs=len(rows),checks=checks),indent=2))
