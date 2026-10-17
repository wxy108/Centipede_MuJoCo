"""Static geometry/filter/recovery proofs on saved poses; zero physics steps."""
from pathlib import Path
import contact_erase_transient as tr
import screening
c=tr.c;np=c.np
from plane_clearance import PlaneClearance,separate_leg_geoms

def evaluator():
    m=c.e.runtime_model(c.PLANTS['current']);ix=c.e.fx.opt.tip.LeanIndex(m)
    return c.e.BASE_EVALUATOR(m,ix)

def main():
    ev=evaluator();m,d=ev.model,ev.data
    legs=separate_leg_geoms(ev);alllegs=np.concatenate(legs)
    support=PlaneClearance(m,ev.ground,alllegs)
    checks=[];contact_checks=[]
    sources=sorted((tr.ce.ps.SOURCE_OUT/'physical_geometry_v2/trials/physical_parity').glob('*/motion.npz'))
    assert len(sources)>=3
    for path in sources:
        with np.load(path) as z:
            for i in np.linspace(len(z['qpos'])//2,len(z['qpos'])-1,4,dtype=int):
                d.qpos[:]=z['qpos'][i];c.mujoco.mj_forward(m,d)
                gaps=support.values(d);lookup=dict(zip(map(int,alllegs),gaps))
                observed={}
                for con in d.contact:
                    a,b=int(con.geom1),int(con.geom2)
                    g=b if a==ev.ground else a if b==ev.ground else -1
                    if g in lookup:observed[g]=min(observed.get(g,float('inf')),float(con.dist))
                for g,gap in observed.items():
                    err=abs(lookup[g]-gap)
                    contact_checks.append(dict(geom=g,error_m=err))
                    assert err<1e-7,(g,lookup[g],gap,str(path),int(i))
                normal=d.geom_xmat[ev.ground].reshape(3,3)[:,2].copy()
                assert m.jnt_type[0]==c.mujoco.mjtJoint.mjJNT_FREE and m.jnt_qposadr[0]==0
                d.qpos[:3]+=.001*normal;c.mujoco.mj_kinematics(m,d)
                translated=support.values(d)
                np.testing.assert_allclose(translated-gaps,.001,rtol=0,atol=1e-12)
                checks.append(dict(source=str(path),pose=int(i),translation_exact=True,ground_contacts_compared=len(observed)))
    assert len(contact_checks)>50
    # Isolated static state transitions exercise penetrating and clear recovery.
    transition=[]
    for high in (False,True):
        ev=evaluator();m,d=ev.model,ev.data;c.mujoco.mj_resetData(m,d);c.mujoco.mj_forward(m,d)
        c._active_task=dict(erasure=dict(probability=1.,seed=0,kind=tr.KIND),duration=6.)
        ch=tr.Channel(ev,.000125,c.ROOT/'outputs/transient_static_only');ch.before(ch.start)
        assert ch.selected.all();ch.matrix_check()
        ch.desired[1]=False # Synthetic recovery request for this no-dynamics check only.
        d.qpos[2]+=1. if high else -1.
        ch.before(ch.start+ch.period)
        assert bool(ch.selected.any())==(not high)
        assert len(ch.restore_events)==(38 if high else 0)
        if high:assert ch.min_restore_clearance_m>=50e-6
        ch.matrix_check();transition.append(dict(raised_clear=high,restored=len(ch.restore_events),no_penetrating_restore=True))
    # The guard builds a new function namespace at first run: verify it retains
    # the transient channel rather than reverting to the persistent class.
    screening.install(False)
    assert c.e.fx.pilot.run_task.__globals__['CONTACT_CHANNEL'] is tr.Channel
    for seed in (0,1,100):
        u=np.random.default_rng(909202600+seed).random((21,19))
        np.testing.assert_array_equal(u[:9],np.random.default_rng(909202600+seed).random((9,19)))
        previous=np.zeros_like(u,dtype=bool)
        for p in (0,.2,.4,.6,.8,1):
            desired=u<p;assert np.all(~previous|desired);previous=desired
    receipt=dict(passed=True,physics_steps_executed=0,pose_checks=checks,
        contact_distance_comparisons=len(contact_checks),max_contact_distance_error_m=max(x['error_m'] for x in contact_checks),
        transitions=transition,guard_namespace_correct=True,desired_masks_nested=True,long_duration_prefix_exact=True,
        source_hashes={str(p):c.sha(p) for p in [Path(__file__),Path(tr.__file__),c.ROOT/'plane_clearance.py',c.ROOT/'contact_erase.py',c.ROOT/'TRANSIENT_ERASURE_PROTOCOL.md']})
    c.write(c.ROOT/'outputs/data/transient_static_check.json',receipt)
    print(dict(passed=True,physics_steps_executed=0,contact_comparisons=len(contact_checks),max_error_m=receipt['max_contact_distance_error_m']))

if __name__=='__main__':main()
