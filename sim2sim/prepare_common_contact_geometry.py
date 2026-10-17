import json
import numpy as np,mujoco
import shared as s
m=mujoco.MjModel.from_xml_path(str(s.ROOT/'assets/common.xml'));points=[];meshes=[]
for g in range(1,m.ngeom):
    body=m.body(int(m.geom_bodyid[g])).name;typ=m.geom_type[g];rot=np.zeros(9);mujoco.mju_quat2Mat(rot,m.geom_quat[g]);rot=rot.reshape(3,3);pos=m.geom_pos[g]
    if typ==mujoco.mjtGeom.mjGEOM_SPHERE:
        points.append(dict(body=body,local_center=pos.tolist(),radius=float(m.geom_size[g,0]),geom=m.geom(g).name,kind='sphere'))
    elif typ==mujoco.mjtGeom.mjGEOM_CAPSULE:
        for sign in [-1,1]:points.append(dict(body=body,local_center=(pos+sign*m.geom_size[g,1]*rot[:,2]).tolist(),radius=float(m.geom_size[g,0]),geom=m.geom(g).name,kind='capsule_endpoint'))
    elif typ==mujoco.mjtGeom.mjGEOM_MESH:
        mid=m.geom_dataid[g];start=m.mesh_vertadr[mid];verts=m.mesh_vert[start:start+m.mesh_vertnum[mid]].astype(float)@rot.T+pos
        meshes.append(dict(body=body,vertices=verts.tolist(),geom=m.geom(g).name))
    else:raise ValueError(str(typ))
s.write(s.ROOT/'assets/common_contact_geometry.json',dict(points=points,meshes=meshes,source_model_sha=s.sha(s.ROOT/'assets/common.xml'),source_sha=s.sha(__file__)))
# Validate fast COM velocity conversion against the engine's public object accessor.
d=mujoco.MjData(m);d.qpos[:7]=s.M['initial_root_pose_wxyz'];d.qvel[:]=np.random.default_rng(947).normal(0,.01,m.nv);mujoco.mj_forward(m,d)
fast=d.cvel[:,3:]+np.cross(d.cvel[:,:3],d.xipos-d.subtree_com[m.body_rootid]);err=0.;v=np.zeros(6)
for b in range(1,m.nbody):mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,b,v,0);err=max(err,float(np.max(np.abs(fast[b]-v[3:]))))
s.write(s.ROOT/'audit/common_contact_geometry_preparation.json',dict(points=len(points),mesh_supports=len(meshes),fast_com_velocity_max_error=err,passed=err<1e-12));assert err<1e-12
print(json.dumps(dict(points=len(points),mesh_supports=len(meshes),fast_com_velocity_max_error=err)))
