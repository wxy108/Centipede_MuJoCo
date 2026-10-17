import argparse,json
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
import shared as s
p=argparse.ArgumentParser();p.add_argument('name');a=p.parse_args();folder=s.ROOT/'runs'/a.name
tr=np.load(folder/'fk.npz');m=mujoco.MjModel.from_xml_path(str(s.ROOT/'assets/common.xml'));d=mujoco.MjData(m)
rows=[]
for v in range(len(tr['q'])):
    d.qpos[:7]=s.M['initial_root_pose_wxyz']
    for k,n in enumerate(tr['joint_names']):d.qpos[m.joint(str(n)).qposadr[0]]=tr['q'][v,0,k]
    mujoco.mj_kinematics(m,d)
    errors=[];angles=[]
    for k,n in enumerate(tr['body_names']):
        b=m.body(str(n)).id;errors.append(float(np.linalg.norm(d.xpos[b]-tr['transforms'][v,0,k,:3])))
        mu=Rotation.from_quat(np.roll(d.xquat[b],-1));nv=Rotation.from_quat(tr['transforms'][v,0,k,3:]);angles.append(float((mu.inv()*nv).magnitude()))
    rows.append(dict(configuration=v,max_position_error_m=max(errors),max_orientation_error_rad=max(angles),worst_body=str(tr['body_names'][np.argmax(errors)])))
s.write(folder/'fk_comparison.json',dict(rows=rows));print(json.dumps(rows))
