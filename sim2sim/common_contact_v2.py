"""One shared SI contact response for both native dynamics backends.

This is a new explicit compliant-contact benchmark, not an assertion that native
MuJoCo/PhysX contact formulations are identical. No simulator-specific parameters.
"""
import json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
C=json.loads((R/'COMMON_CONTACT_V2.json').read_text())

def quat_matrix(q):
    q=q/np.linalg.norm(q,axis=-1,keepdims=True);w,x,y,z=np.moveaxis(q,-1,0)
    return np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],axis=-1).reshape(q.shape[:-1]+(3,3))

class CommonContact:
    def __init__(self,body_names,num_envs,dt,geometry=None):
        G=json.loads((R/'assets/common_contact_geometry.json').read_text()) if geometry is None else geometry
        self.dt=dt;self.nenv=num_envs;self.nb=len(body_names);self.points=G['points'];self.meshes=G['meshes'];self.nr=len(self.points);self.np=self.nr+len(self.meshes)
        self.ids=np.array([body_names.index(p['body']) for p in self.points]+[body_names.index(p['body']) for p in self.meshes],int)
        self.local=np.array([p['local_center'] for p in self.points],float).reshape(-1,3);self.radii=np.array([p['radius'] for p in self.points]);self.mesh_verts=[np.array(p['vertices'],float) for p in self.meshes]
        self.xi=np.zeros((num_envs,self.np,2));self.mesh_index=np.full((num_envs,len(self.meshes)),-1,int)
    def reset(self):self.xi.fill(0);self.mesh_index.fill(-1)
    def forces(self,position,rotation,com,velocity):
        # Positions are link origins, rotations map link axes to world. COM linear
        # and angular velocities are both world-frame, linear first.
        point=np.zeros((self.nenv,self.np,3));bi=self.ids[:self.nr]
        point[:,:self.nr]=np.einsum('npij,pj->npi',rotation[:,bi],self.local)+position[:,bi];point[:,:self.nr,2]-=self.radii
        for j,verts in enumerate(self.mesh_verts):
            b=self.ids[self.nr+j];world=np.einsum('nij,vj->nvi',rotation[:,b],verts)+position[:,b,None,:];index=np.argmin(world[:,:,2],axis=1);point[:,self.nr+j]=world[np.arange(self.nenv),index];changed=index!=self.mesh_index[:,j];self.xi[changed,self.nr+j]=0.;self.mesh_index[:,j]=index
        arm=point-com[:,self.ids];surface_v=velocity[:,self.ids,:3]+np.cross(velocity[:,self.ids,3:],arm)
        depth=np.maximum(-point[:,:,2],0.);fn=np.where(depth>0,np.maximum(C['normal_stiffness_N_per_m']*depth-C['normal_damping_N_s_per_m']*surface_v[:,:,2],0.),0.)
        loaded=fn>0;self.xi+=self.dt*surface_v[:,:,:2];self.xi[~loaded]=0.
        trial=-C['tangential_stiffness_N_per_m']*self.xi-C['tangential_damping_N_s_per_m']*surface_v[:,:,:2];limit=C['friction_coefficient']*fn;length=np.linalg.norm(trial,axis=2);ratio=np.minimum(1.,limit/np.maximum(length,1e-30));ft=trial*ratio[:,:,None];sliding=(length>limit)&loaded
        self.xi[sliding]=-(ft+C['tangential_damping_N_s_per_m']*surface_v[:,:,:2])[sliding]/C['tangential_stiffness_N_per_m']
        force=np.concatenate([ft,fn[:,:,None]],axis=2);torque=np.cross(arm,force);F=np.zeros((self.nenv,self.nb,3));T=F.copy()
        for e in range(self.nenv):np.add.at(F[e],self.ids,force[e]);np.add.at(T[e],self.ids,torque[e])
        diagnostics=dict(max_penetration_m=float(depth.max()),max_point_force_N=float(np.linalg.norm(force,axis=2).max()),loaded_points=loaded.sum(1).tolist())
        return F,T,diagnostics
