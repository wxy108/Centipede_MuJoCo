"""Analytic square-limit representation in MuJoCo's four-edge pyramid.

Physical tangent-axis bounds are mu*N. Numeric pyramid vertices are sqrt(2)*mu
along a 45-degree-rotated basis. No locomotion-dependent coefficient is fitted.
"""
import numpy as np
import mujoco

class BoxFrictionAdapter:
    def __init__(self,model,basis):
        self.basis=basis
        self.velocity=np.zeros((model.nbody,6))
        self.physical_mu=model.geom_friction[:,0].copy()
        model.opt.cone=mujoco.mjtCone.mjCONE_PYRAMIDAL
        model.geom_friction[:,0]=self.physical_mu*np.sqrt(2.)

    def step(self,m,d):
        mujoco.mj_step1(m,d)
        if d.ncon:
            frame=d.contact.frame.reshape(-1,3,3)
            normal=frame[:,0].copy()
            fallback=np.column_stack([np.zeros(d.ncon),-normal[:,2],normal[:,1]])
            alt=np.column_stack([-normal[:,1],normal[:,0],np.zeros(d.ncon)])
            use_alt=np.abs(normal[:,0])>=.70710678
            fallback[use_alt]=alt[use_alt]
            tangent=fallback
            if self.basis=='velocity':
                bodies=m.geom_bodyid[d.contact.geom]
                for b in np.unique(bodies):
                    if b:mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,int(b),self.velocity[b],0)
                rel=self.velocity[bodies[:,0],3:]-self.velocity[bodies[:,1],3:]
                slip=rel-np.sum(rel*normal,axis=1)[:,None]*normal
                use_velocity=np.sum(slip*slip,axis=1)>1.e-4
                tangent[use_velocity]=slip[use_velocity]
            tangent/=np.linalg.norm(tangent,axis=1)[:,None]
            second=np.cross(normal,tangent)
            frame[:,1]=(tangent+second)/np.sqrt(2.)
            frame[:,2]=np.cross(normal,frame[:,1])
            mujoco.mj_makeConstraint(m,d)
            mujoco.mj_projectConstraint(m,d)
            mujoco.mj_referenceConstraint(m,d)
        mujoco.mj_step2(m,d)
