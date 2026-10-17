"""Equivalent ground-force observation; physics and control remain unchanged.

The normal component of a pyramidal contact is the sum of its edge forces;
for elliptic/frictionless contacts it is the first scalar constraint force.
This observation is checked against mj_contactForce before use.
"""
import inspect, json, time
from pathlib import Path
import core as c
from numba import njit

@njit(cache=False)
def accumulate(g1,g2,addresses,dims,dist,efc,ground,categories,pyramidal,out,feet):
    penetration=0.
    for j in range(len(g1)):
        other=g2[j] if g1[j]==ground else (g1[j] if g2[j]==ground else -1)
        if other<0:continue
        adr=addresses[j];fn=0.
        if adr>=0:
            if not pyramidal or dims[j]==1:fn=efc[adr]
            else:
                for k in range(2*(dims[j]-1)):fn+=efc[adr+k]
        if fn<0:fn=0.
        cat=categories[other];out[cat]+=fn
        if cat<38:feet[cat]=True
        if -dist[j]>penetration:penetration=-dist[j]
    return penetration

def generate():
    source=inspect.getsource(c.e.fx.pilot.run_task)
    start=source.index('        for j in range(d.ncon):\n')
    end=source.index('        if k%stride==0:\n',start)
    source=source[:start]+'''        pen[k]=ACCUMULATE_GROUND(d.contact.geom1,d.contact.geom2,d.contact.efc_address,
            d.contact.dim,d.contact.dist,d.efc_force,ev.ground,ground_categories,
            int(m.opt.cone)==0,force[k],foot_mask[k])
'''+source[end:]
    marker="    n=int(round(duration/dt)); weight=float(m.body_mass.sum()*9.81)\n"
    assert source.count(marker)==1
    source=source.replace(marker,'''    ground_categories=np.full(m.ngeom,59,dtype=np.int64)
    for g in range(m.ngeom):
        if g in foot_map:ground_categories[g]=foot_map[g]
        elif int(m.geom_bodyid[g]) in body_map:ground_categories[g]=38+body_map[int(m.geom_bodyid[g])]
'''+marker)
    path=c.ROOT/'fast_run_task.py'
    if path.exists():assert path.read_text()==source
    else:path.write_text(source,encoding='utf-8')
    return path

def install():
    receipt=c.read(c.DATA/'fast_observation_check.json')
    assert receipt['passed'] and receipt['helper_sha256']==c.sha(__file__)
    path=c.ROOT/'fast_run_task.py';assert receipt['evaluator_sha256']==c.sha(path)
    pilot=c.e.fx.pilot
    ns=dict(vars(pilot),ACCUMULATE_GROUND=accumulate)
    exec(compile(path.read_text(),str(path),'exec'),ns)
    pilot.run_task=ns['run_task'];pilot.ACCUMULATE_GROUND=accumulate

def check():
    path=generate();checks=[]
    rows=[r for r in c.records() if r.get('motion_path')]
    assert rows
    for row in rows[:3]:
        m=c.e.runtime_model(c.PLANTS['current']);ix=c.e.fx.opt.tip.LeanIndex(m)
        c.ORIGINAL_APPLY(m,ix,c.CFG);ev=c.e.BASE_EVALUATOR(m,ix);d=ev.data
        z=c.np.load(row['motion_path'])
        fmap={int(g):i for i,g in enumerate(ev.foot_geoms)}
        bmap={int(b):i for i,b in enumerate(ix.chassis_body_ids)}
        cat=c.np.full(m.ngeom,59,dtype=c.np.int64)
        for g in range(m.ngeom):
            if g in fmap:cat[g]=fmap[g]
            elif int(m.geom_bodyid[g]) in bmap:cat[g]=38+bmap[int(m.geom_bodyid[g])]
        max_diff=0.;contacts=0;elapsed_old=elapsed_new=0.
        for index in range(90,len(z['pose_t']),3):
            d.qpos[:]=z['qpos'][index];c.mujoco.mj_forward(m,d)
            old=c.np.zeros(60);new=c.np.zeros(60);of=c.np.zeros(38,dtype=bool);nf=of.copy();cf=c.np.zeros(6);pen=0.
            t=time.perf_counter()
            for j in range(d.ncon):
                con=d.contact[j];g1,g2=int(con.geom1),int(con.geom2)
                other=g2 if g1==ev.ground else (g1 if g2==ev.ground else -1)
                if other<0:continue
                c.mujoco.mj_contactForce(m,d,j,cf);old[cat[other]]+=max(0.,float(cf[0]))
                if cat[other]<38:of[cat[other]]=True
                pen=max(pen,-float(con.dist))
            elapsed_old+=time.perf_counter()-t
            t=time.perf_counter()
            newpen=accumulate(d.contact.geom1,d.contact.geom2,d.contact.efc_address,d.contact.dim,d.contact.dist,
                d.efc_force,ev.ground,cat,int(m.opt.cone)==0,new,nf)
            elapsed_new+=time.perf_counter()-t
            diff=float(c.np.max(c.np.abs(old-new)));max_diff=max(max_diff,diff);contacts+=d.ncon
            assert c.np.allclose(old,new,rtol=1e-13,atol=1e-16)
            assert c.np.array_equal(of,nf) and newpen==pen
        checks.append(dict(wave=row['wave'],contacts_checked=contacts,max_force_difference=max_diff,oracle_s=elapsed_old,fast_including_jit_s=elapsed_new))
    result=dict(passed=True,helper_sha256=c.sha(__file__),evaluator_sha256=c.sha(path),checks=checks,
        scope='Force observation equivalence; full-trajectory equivalence still required before search uses this observer.')
    c.write(c.DATA/'fast_observation_check.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':check()
