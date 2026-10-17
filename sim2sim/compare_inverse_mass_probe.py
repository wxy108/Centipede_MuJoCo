import json,numpy as np,mujoco
import shared as s
m=mujoco.MjModel.from_xml_path(str(s.ROOT/'assets/common.xml'));d=mujoco.MjData(m);d.qpos[:7]=s.M['initial_root_pose_wxyz'];mujoco.mj_forward(m,d);mu=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,d,mu);rows=[]
for name in ['inverse_mass_cpu_001','inverse_mass_gpu_001']:
    f=s.ROOT/'runs'/name;meta=json.loads((f/'inverse_probe.json').read_text());tr={k:v for k,v in np.load(f/'inverse_probe.npz').items()};names=tr['joint_names'].tolist();order=list(range(6))+[m.joint(n).dofadr[0] for n in names];ref=mu[np.ix_(order,order)]
    x,y,z=m.body_ipos[1];transform=np.eye(m.nv);transform[:3,3:6]=[[0,-z,y],[z,0,-x],[-y,x,0]];ref=transform.T@ref@transform
    assert np.max(abs(tr['stiffness']))==0 and np.max(abs(tr['damping']))==0
    native=np.c_[tr['root_v_after']-tr['root_v_before'],tr['v_after']-tr['v_before']];forces=np.c_[np.zeros((7,6)),tr['force']];expected=np.linalg.solve(ref,forces.T).T*meta['dt'];case_rows=[]
    for e,j in enumerate(meta['selected_joints']):
        error=native[e]-expected[e];idx=6+names.index(j);case_rows.append(dict(joint=j,expected_own_velocity_rad_s=float(expected[e,idx]),native_own_velocity_rad_s=float(native[e,idx]),own_relative_error=float(abs(error[idx]/expected[e,idx])),kinetic_norm_relative_error=float(np.sqrt((error@ref@error)/(expected[e]@ref@expected[e])))))
    rows.append(dict(run=name,cases=case_rows));np.savez_compressed(f/'inverse_comparison.npz',expected=expected,native=native)
s.write(s.ROOT/'audit/inverse_mass_response_comparison.json',dict(rows=rows,method='One 1 µs step, zero initial velocity, zero gravity and zero drives. Compare measured impulse response with physical MuJoCo M inverse, root COM frame transformed.',source_sha=s.sha(__file__)));print(json.dumps(rows,indent=2))
