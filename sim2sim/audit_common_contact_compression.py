"""Geometry diagnostics of the new compliant model; never infer force contact from a sampled pose."""
import argparse,json
import numpy as np,mujoco
import shared as s
p=argparse.ArgumentParser();p.add_argument('name');a=p.parse_args();folder=s.ROOT/'runs'/a.name
tr=np.load(folder/'trajectory.npz');roots=np.load(folder/'reconstructed_root_poses.npz')['root'];m=mujoco.MjModel.from_xml_path(str(s.ROOT/'assets/common.xml'));d=mujoco.MjData(m);qids=[m.joint(n).qposadr[0] for n in tr['joint_names']]
meshes=[]
for g in range(1,m.ngeom):
    if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH:
        mid=m.geom_dataid[g];start=m.mesh_vertadr[mid];meshes.append((g,m.mesh_vert[start:start+m.mesh_vertnum[mid]].astype(float)))
ids=np.flatnonzero(tr['t']>=2);rows=[]
for e in range(tr['q'].shape[1]):
    heights=[]
    for k in ids:
        d.qpos[:7]=roots[k,e];d.qpos[qids]=tr['q'][k,e];mujoco.mj_kinematics(m,d)
        heights.append([float((v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])[:,2].min()) for g,v in meshes])
    z=np.array(heights);per=[]
    for j,(g,_) in enumerate(meshes):
        if z[:,j].min()<0:per.append(dict(body=m.body(int(m.geom_bodyid[g])).name,collider=m.geom(g).name,maximum_compression_um=float(max(0,-z[:,j].min())*1e6),fraction_sampled_poses_below_plane=float(np.mean(z[:,j]<0))))
    rows.append(dict(environment=e,compressed_meshes=per,maximum_mesh_compression_um=float(max(0,-z.min())*1e6)))
result=dict(physical_model='COMMON_CONTACT_V2.json',window_s=[2,6],sample_dt_s=.01,rows=rows,qualification='Pose-based compression, including reconstruction uncertainty; this is not a measured force-contact probability. Positive compression is part of this explicit compliant law, not native rigid penetration tolerance.',source_sha=s.sha(__file__))
s.write(folder/'compliance_geometry_audit.json',result);print(json.dumps(dict(name=a.name,rows=rows)))
