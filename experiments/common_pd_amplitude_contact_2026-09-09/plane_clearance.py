"""Exact lowest support height of collision geometry above the model plane.

Read-only geometry calculation; the caller must update kinematics for current qpos.
Mesh extrema equal their convex-hull extrema along a plane normal.
"""
import numpy as np
import mujoco

class PlaneClearance:
    def __init__(self,model,ground,geom_ids):
        self.m=model;self.ground=int(ground);self.ids=np.asarray(geom_ids,int)
        assert model.geom_type[self.ground]==mujoco.mjtGeom.mjGEOM_PLANE
        self.mesh={}
        for g in self.ids:
            if model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH:
                mid=int(model.geom_dataid[g]);start=int(model.mesh_vertadr[mid]);n=int(model.mesh_vertnum[mid])
                self.mesh[int(g)]=np.asarray(model.mesh_vert[start:start+n],float)

    def values(self,data,ids=None):
        ids=self.ids if ids is None else np.asarray(ids,int)
        normal=data.geom_xmat[self.ground].reshape(3,3)[:,2]
        origin=data.geom_xpos[self.ground];out=[]
        for g in ids:
            kind=self.m.geom_type[g];size=self.m.geom_size[g]
            local=data.geom_xmat[g].reshape(3,3).T@normal
            height=float((data.geom_xpos[g]-origin)@normal)
            if kind==mujoco.mjtGeom.mjGEOM_SPHERE:offset=-size[0]
            elif kind==mujoco.mjtGeom.mjGEOM_CAPSULE:offset=-size[0]-size[1]*abs(local[2])
            elif kind==mujoco.mjtGeom.mjGEOM_BOX:offset=-float(np.abs(local)@size)
            elif kind==mujoco.mjtGeom.mjGEOM_ELLIPSOID:offset=-float(np.linalg.norm(size*local))
            elif kind==mujoco.mjtGeom.mjGEOM_CYLINDER:offset=-size[0]*np.linalg.norm(local[:2])-size[1]*abs(local[2])
            elif kind==mujoco.mjtGeom.mjGEOM_MESH:offset=float(np.min(self.mesh[int(g)]@local))
            else:raise ValueError(('Unsupported collision geometry',int(g),int(kind)))
            out.append(height+offset)
        return np.asarray(out)

def separate_leg_geoms(ev):
    m=ev.model;groups=[]
    for segment in range(19):
        for side in ('L','R'):
            root=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,f'link_leg_{segment}_{side}_0')
            assert root>=0
            descendants=set()
            for b in range(m.nbody):
                p=b
                while p>0:
                    if p==root:descendants.add(b);break
                    p=int(m.body_parentid[p])
            ids=np.array([g for g in ev.segment_collision_geoms[segment] if int(m.geom_bodyid[g]) in descendants],int)
            assert len(ids)>0;groups.append(ids)
    for segment in range(19):
        assert set(np.concatenate(groups[2*segment:2*segment+2]))==set(ev.segment_collision_geoms[segment])
    return groups
