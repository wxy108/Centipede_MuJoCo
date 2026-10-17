"""Geometric diagnostic: tortuosity is not equivalent to body self-intersection."""
import core as c
import itertools,json
np=c.np


def main():
    m=c.e.runtime_model(c.PLANTS['current']);ix=c.e.fx.opt.tip.LeanIndex(m);d=c.mujoco.MjData(m)
    qa=m.jnt_qposadr[ix.active_joint_ids[ix.body_positions]]
    bodyids=ix.chassis_body_ids
    def tort():
        x=d.xpos[bodyids]
        return float(np.linalg.norm(np.diff(x,axis=0),axis=1).sum()/np.linalg.norm(x[-1]-x[0]))
    def prescribed(w,a):
        values=[]
        for phase in np.linspace(0,2*np.pi,181)[:-1]:
            c.mujoco.mj_resetData(m,d)
            d.qpos[qa]=np.radians(a)*np.sin(phase-2*np.pi*w*np.arange(19)/18)
            d.qpos[qa[0]]=0
            c.mujoco.mj_kinematics(m,d);values.append(tort())
        return max(values)
    caps=[]
    for w in c.WAVES:
        lo,hi=0.,36.
        atmax=prescribed(w,36)
        if atmax<=1.2:lo=36.
        else:
            for _ in range(16):
                mid=(lo+hi)/2
                if prescribed(w,mid)<=1.2:lo=mid
                else:hi=mid
        caps.append(dict(wave=w,ideal_body_amplitude_cap_deg=lo,tortuosity_at_36_deg=atmax,
            capped_by_search_domain=atmax<=1.2))
    bridge=c.read(c.DATA/'bridge_analysis.json')['rows']
    selected=[r for r in bridge if r['plant']=='current' and r['family']=='rl_v4' and r['phase']=='midpoint' and r['wave'] in (1.5,2,3)]
    geoms=[next(g for g in range(m.ngeom) if m.geom_bodyid[g]==b and m.geom_contype[g]) for b in bodyids]
    pairs=[(i,j) for i,j in itertools.combinations(range(len(geoms)),2) if j-i>1]
    pair_i,pair_j=np.array(pairs).T
    boxcenter=[];boxhalf=[]
    for g in geoms:
        mesh=int(m.geom_dataid[g]);start=int(m.mesh_vertadr[mesh]);count=int(m.mesh_vertnum[mesh])
        v=m.mesh_vert[start:start+count];low=v.min(0);high=v.max(0)
        boxcenter.append((low+high)/2);boxhalf.append((high-low)/2)
    boxcenter=np.array(boxcenter);boxhalf=np.array(boxhalf)
    def box_separation():
        rot=d.geom_xmat[geoms].reshape(-1,3,3)
        centers=d.geom_xpos[geoms]+np.einsum('nij,nj->ni',rot,boxcenter)
        ra,rb=rot[pair_i],rot[pair_j]
        a,b=ra.transpose(0,2,1),rb.transpose(0,2,1)
        cross=np.cross(a[:,:,None,:],b[:,None,:,:]).reshape(-1,9,3)
        axis=np.concatenate([a,b,cross],axis=1)
        norm=np.linalg.norm(axis,axis=2);axis=axis/np.maximum(norm[:,:,None],1e-15)
        radius_a=(np.abs(np.einsum('nki,nij->nkj',axis,ra))*boxhalf[pair_i,None]).sum(2)
        radius_b=(np.abs(np.einsum('nki,nij->nkj',axis,rb))*boxhalf[pair_j,None]).sum(2)
        gaps=np.abs(np.einsum('nki,ni->nk',axis,centers[pair_i]-centers[pair_j]))-radius_a-radius_b
        gaps[norm<1e-10]=-np.inf
        return gaps.max(1)
    results=[];shapes={}
    for r in selected:
        z=np.load(r['motion']);q=z['pose_t']>=1.5
        xposes=[];ts=[];ratios=[];minsep=.1;argmin=None;boxgap=.1;body_contact_count=0
        for t,pose in zip(z['pose_t'][q],z['qpos'][q]):
            d.qpos[:]=pose;c.mujoco.mj_forward(m,d)
            ratios.append(tort());xposes.append(d.xpos[bodyids].copy());ts.append(float(t))
            boxgap=min(boxgap,float(box_separation().min()))
            lookup={g:i for i,g in enumerate(geoms)}
            body_contact_count+=sum(x.geom1 in lookup and x.geom2 in lookup and abs(lookup[x.geom1]-lookup[x.geom2])>1 for x in d.contact)
            for i,j in pairs:
                sep=float(c.mujoco.mj_geomDistance(m,d,geoms[i],geoms[j],.1,None))
                if sep<minsep:minsep=sep;argmin=dict(t=float(t),body_indices=[i,j])
        j=int(np.argmax(ratios));shapes[r['wave']]=np.asarray(xposes[j])*1000
        results.append(dict(id=r['id'],wave=r['wave'],D=r['D'],failed=r['failed'],
            body_command_deg=r['amplitude']['body'],body_measured_deg=r['realized_body_harmonic_median_deg'],
            tortuosity_max_full_observer=r['metrics']['spine_tortuosity_max'],
            tortuosity_max_pose_sample=ratios[j],max_pose_time=ts[j],
            nonadjacent_chassis_min_clearance_mm=minsep*1000,min_clearance_where=argmin,
            mesh_distance_unreliable=True,
            mesh_distance_note='Some mesh pairs return zero with noncoincident closest points and no collision; do not interpret as penetration or clearance proof.',
            conservative_box_separation_lower_bound_mm=boxgap*1000,
            nonadjacent_chassis_contact_samples=body_contact_count,
            pose_samples=len(ts),sample_rate_hz=60,
            limitation='Saved-pose geometric clearance, not a continuous-time guarantee or a leg-leg collision audit.'))
    c.write(c.DATA/'shape_geometry_audit.json',dict(ideal_caps=caps,actual_poses=results,
        diagnostic_only=True,acceptance_changed=False,code_sha256=c.sha(__file__)))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
    axes[0].plot([r['wave'] for r in caps],[r['ideal_body_amplitude_cap_deg'] for r in caps],'o-',label='Largest ideal joint amplitude passing R <= 1.2')
    axes[0].set(xlabel='Coupled wave number',ylabel='Body joint amplitude (deg)',ylim=(0,38),title='The shape gate restricts low waves more')
    axes[0].grid(alpha=.2);axes[0].legend(fontsize=8,loc='lower right')
    for offset,r in enumerate(results):
        x=shapes[r['wave']];u=x[-1,:2]-x[0,:2];u=u/np.linalg.norm(u)
        xy=(x[:,:2]-x[:1,:2])@np.array([[u[0],-u[1]],[u[1],u[0]]])
        xy[:,1]+=offset*35
        axes[1].plot(xy[:,0],xy[:,1],'.-',label=f"w={r['wave']:g}, R={r['tortuosity_max_pose_sample']:.3f}")
    axes[1].set(xlabel='End-to-end axis (mm)',ylabel='Lateral position, offset for clarity (mm)',title='Actual RL-amplitude poses at maximum R')
    axes[1].axis('equal');axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
    fig.savefig(c.ROOT/'shape_gate_diagnostic.png',dpi=180);fig.savefig(c.ROOT/'shape_gate_diagnostic.pdf')
    print(json.dumps(dict(caps=caps,actual=results),indent=2))


if __name__=='__main__':main()
