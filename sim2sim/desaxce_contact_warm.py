"""Experimental non-associated Coulomb correction around MuJoCo's cone solve.

Physical mu, mass and PD are unchanged. Normal reference is corrected so sliding
friction does not impose associated-flow normal separation. Not yet adopted.
"""
import numpy as np,mujoco

def step(m,d,max_iterations=20,velocity_tolerance=1e-8):
    assert m.opt.integrator==mujoco.mjtIntegrator.mjINT_EULER
    assert m.opt.cone==mujoco.mjtCone.mjCONE_ELLIPTIC
    mujoco.mj_step1(m,d)
    mujoco.mj_fwdActuation(m,d);mujoco.mj_fwdAcceleration(m,d)
    base=d.efc_aref.copy()
    contacts=[c for c in d.contact if c.efc_address>=0 and c.dim>1]
    assert all(c.dim==3 and abs(c.friction[0]-c.friction[1])<1e-12 for c in contacts)
    normal=np.array([c.efc_address for c in contacts],dtype=int)
    tangent=normal[:,None]+np.array([1,2])[None,:]
    mu=np.array([c.friction[0] for c in contacts])
    corr=np.zeros(len(contacts));jacacc=np.zeros(d.nefc)
    if len(contacts) and max_iterations:
        mujoco.mj_mulJacVec(m,d,jacacc,d.qacc_smooth)
        corr=mu*np.linalg.norm((jacacc-base)[tangent],axis=1)
    residual=0.;iterations=0
    for it in range(max_iterations+1):
        d.efc_aref[:]=base
        d.efc_aref[normal]-=corr
        mujoco.mj_fwdConstraint(m,d)
        if not len(contacts) or max_iterations==0:break
        mujoco.mj_mulJacVec(m,d,jacacc,d.qacc)
        alpha=jacacc-base+d.efc_R*d.efc_force
        update=mu*np.linalg.norm(alpha[tangent],axis=1)
        residual=float(np.max(np.abs(update-corr))*m.opt.timestep)
        iterations=it+1
        if residual<=velocity_tolerance:break
        corr=update
        d.qacc_warmstart[:]=d.qacc
    mujoco.mj_Euler(m,d)
    return dict(iterations=iterations,residual_m_s=residual,converged=residual<=velocity_tolerance)
