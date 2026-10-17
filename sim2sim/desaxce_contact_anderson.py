"""Experimental De Saxce fixed point with Anderson acceleration.

Acceleration changes numerical root finding only. The same correction equation,
physical contact cone, PD, mass, normal regularizer and tolerance are retained.
"""
import numpy as np,mujoco

def step(m,d,max_iterations=60,velocity_tolerance=1e-8):
    assert m.opt.integrator==mujoco.mjtIntegrator.mjINT_EULER
    assert m.opt.cone==mujoco.mjtCone.mjCONE_ELLIPTIC
    mujoco.mj_step1(m,d);mujoco.mj_fwdActuation(m,d);mujoco.mj_fwdAcceleration(m,d)
    contacts=[c for c in d.contact if c.efc_address>=0 and c.dim>1]
    assert all(c.dim==3 and abs(c.friction[0]-c.friction[1])<1e-12 for c in contacts)
    if not contacts or max_iterations==0:
        mujoco.mj_fwdConstraint(m,d);mujoco.mj_Euler(m,d);return dict(iterations=0,residual_m_s=0.,converged=True)
    h=m.opt.timestep;base=d.efc_aref.copy();normal=np.array([c.efc_address for c in contacts],int);tangent=normal[:,None]+np.array([1,2])[None,:];mu=np.array([c.friction[0] for c in contacts]);ja=np.zeros(d.nefc)
    mujoco.mj_mulJacVec(m,d,ja,d.qacc_smooth);x=h*mu*np.linalg.norm((ja-base)[tangent],axis=1)
    xs=[];fs=[];best=None;best_res=np.inf;residual=np.inf
    for it in range(max_iterations+1):
        d.efc_aref[:]=base;d.efc_aref[normal]-=x/h;mujoco.mj_fwdConstraint(m,d)
        mujoco.mj_mulJacVec(m,d,ja,d.qacc);alpha=ja-base+d.efc_R*d.efc_force;fx=h*mu*np.linalg.norm(alpha[tangent],axis=1)-x;residual=float(np.max(np.abs(fx)))
        if residual<=velocity_tolerance:break
        d.qacc_warmstart[:]=d.qacc
        if residual<best_res:best_res=residual;best=(x.copy(),fx.copy())
        if len(xs) and residual>4*max(float(np.max(np.abs(fs[-1]))),velocity_tolerance):
            x=best[0]+.5*best[1];xs=[];fs=[];continue
        xs.append(x.copy());fs.append(fx.copy());xs=xs[-7:];fs=fs[-7:]
        candidate=x+fx
        if len(xs)>1:
            dx=np.diff(np.array(xs),axis=0).T;df=np.diff(np.array(fs),axis=0).T
            gamma=np.linalg.lstsq(df,fx,rcond=1e-10)[0]
            accelerated=candidate-(dx+df)@gamma
            if np.isfinite(accelerated).all() and np.linalg.norm(accelerated-x)<=10*max(np.linalg.norm(candidate),1e-6):candidate=accelerated
        x=np.maximum(candidate,0.)
    mujoco.mj_Euler(m,d)
    return dict(iterations=it+1,residual_m_s=residual,converged=residual<=velocity_tolerance)
