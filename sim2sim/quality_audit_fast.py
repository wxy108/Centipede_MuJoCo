"""Reconstruct recorded poses by rigid alignment, never advance physics."""
import argparse,json
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
import shared as s
p=argparse.ArgumentParser();p.add_argument('name');a=p.parse_args();folder=s.ROOT/'runs'/a.name;tr={k:v.copy() for k,v in np.load(folder/'trajectory.npz').items()}
m=mujoco.MjModel.from_xml_path(str(s.ROOT/'assets/common.xml'));d=mujoco.MjData(m)
names=tr['joint_names'].tolist();bn=tr['body_names'].tolist();bids=[m.body(n).id for n in bn];qids=[m.joint(n).qposadr[0] for n in names]
spine=[m.body(f'link_body_{i}').id for i in range(1,20)];footgeoms=[j for j in range(1,m.ngeom) if m.geom_type[j]==mujoco.mjtGeom.mjGEOM_SPHERE]
meshes=[]
for g in range(1,m.ngeom):
    if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH:
        mid=m.geom_dataid[g];start=m.mesh_vertadr[mid];meshes.append((g,m.mesh_vert[start:start+m.mesh_vertnum[mid]].copy()))
rows=[];rootposes=np.zeros((len(tr['t']),tr['q'].shape[1],7))
for e in range(tr['q'].shape[1]):
    stats=[]
    for k,t in enumerate(tr['t']):
        d.qpos[:7]=[0,0,0,1,0,0,0];d.qpos[qids]=tr['q'][k,e];mujoco.mj_kinematics(m,d)
        x=d.xpos[bids].copy();y=tr['body'][k,e].astype(float);xc=x.mean(0);yc=y.mean(0);u,sv,vt=np.linalg.svd((x-xc).T@(y-yc));r=vt.T@np.diag([1,1,np.linalg.det(vt.T@u.T)])@u.T
        pos=yc-r@xc;quat=np.roll(Rotation.from_matrix(r).as_quat(),1);d.qpos[:7]=np.r_[pos,quat];rootposes[k,e]=d.qpos[:7];mujoco.mj_kinematics(m,d)
        if t<2:continue
        residual=float(np.linalg.norm(d.xpos[bids]-y,axis=1).max());up=float(d.xmat[spine].reshape(-1,3,3)[:,2,2].min())
        foot_min=float(np.min(d.geom_xpos[footgeoms,2]-m.geom_size[footgeoms,0]));mesh_min=min(float((verts@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])[:,2].min()) for g,verts in meshes)
        stats.append([residual,up,foot_min,mesh_min])
    st=np.array(stats);rows.append(dict(case=e,finite=bool(np.isfinite(tr['q'][:,e]).all() and np.isfinite(tr['body'][:,e]).all()),max_pose_fit_residual_m=float(st[:,0].max()),min_spine_up_dot=float(st[:,1].min()),min_foot_sphere_surface_z_m=float(st[:,2].min()),min_spine_collider_z_m=float(st[:,3].min()),inverted_spine_samples=int(np.sum(st[:,1]<0)),spine_below_plane_samples=int(np.sum(st[:,3]<0))))
np.savez_compressed(folder/'reconstructed_root_poses.npz',t=tr['t'],root=rootposes)
s.write(folder/'quality_audit.json',dict(rows=rows,method='Kabsch fit of joint-conditioned common model to all recorded body origins; no physics steps',source_sha=s.sha(__file__)));print(json.dumps(rows,indent=2))
