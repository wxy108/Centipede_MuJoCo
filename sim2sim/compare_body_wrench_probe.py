import argparse,json
import numpy as np,mujoco
import shared as s
p=argparse.ArgumentParser();p.add_argument('name');a=p.parse_args();f=s.ROOT/'runs'/a.name;meta=json.loads((f/'body_wrench_probe.json').read_text());tr={k:v for k,v in np.load(f/'body_wrench_probe.npz').items()};names=tr['joint_names'].tolist();bn=tr['body_names'].tolist()
m=mujoco.MjModel.from_xml_path(str(s.ROOT/'assets/common.xml'));d=mujoco.MjData(m);d.qpos[:7]=s.M['initial_root_pose_wxyz'];mujoco.mj_forward(m,d);M=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,d,M);order=list(range(6))+[m.joint(n).dofadr[0] for n in names];x,y,z=m.body_ipos[1];T=np.eye(m.nv);T[:3,3:6]=[[0,-z,y],[z,0,-x],[-y,x,0]];ref=T.T@M[np.ix_(order,order)]@T;rows=[]
assert np.max(abs(tr['stiffness']))==0 and np.max(abs(tr['damping']))==0
native=np.c_[tr['root_v_after']-tr['root_v_before'],tr['v_after']-tr['v_before']]
for e,bname in enumerate(meta['selected_bodies']):
    i=bn.index(bname);bid=m.body(bname).id;qf=np.zeros(m.nv);mujoco.mj_applyFT(m,d,tr['force'][e,i].astype(float),tr['torque'][e,i].astype(float),d.xipos[bid],bid,qf);expected=np.linalg.solve(ref,T.T@qf[order])*meta['dt'];err=native[e]-expected;rel=float(np.sqrt((err@ref@err)/(expected@ref@expected)));rows.append(dict(body=bname,kinetic_norm_relative_error=rel,passed=rel<=.02))
s.write(f/'comparison.json',dict(rows=rows,passed=all(r['passed'] for r in rows),method='Compare production external-wrench API impulse response with independently assembled MuJoCo J-transpose body-COM wrench and full inverse physical inertia. Root velocity references converted.',source_sha=s.sha(__file__)));print(json.dumps(dict(passed=all(r['passed'] for r in rows),maximum_kinetic_relative_error=max(r['kinetic_norm_relative_error'] for r in rows)),indent=2))
