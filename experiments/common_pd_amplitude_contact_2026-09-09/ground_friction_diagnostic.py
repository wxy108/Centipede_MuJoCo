"""Separate matched 2x2 friction/self-collision diagnosis; never an optimizer arm."""
from pathlib import Path
import argparse,json
import physical_search as ps
import core as c
np=c.np
BASE_MODEL=c.e.runtime_model;SOURCE_DATA=c.DATA

def matrix(m):
    a=m.geom_contype;b=m.geom_conaffinity
    out=((a[:,None]&b[None,:])!=0)|((a[None,:]&b[:,None])!=0);np.fill_diagonal(out,False);return out

def treatment(m,mu,self_off,priority=True):
    ground_ids=np.flatnonzero(m.geom_type==int(c.mujoco.mjtGeom.mjGEOM_PLANE));assert len(ground_ids)==1
    ground=int(ground_ids[0]);active=(m.geom_contype!=0)|(m.geom_conaffinity!=0)
    robot=active.copy();robot[ground]=False
    assert np.all(m.geom_friction[active,0]==1.6)
    assert np.all(m.geom_friction[robot]==np.array([1.6,.02,.001]))
    combined=np.maximum(m.geom_friction[ground],m.geom_friction[robot][0])
    assert np.all(m.geom_priority[active]==0) and np.all(m.geom_condim[active]==3)
    for key in ('geom_solref','geom_solimp','geom_solmix','geom_margin','geom_gap'):
        x=getattr(m,key);assert np.all(x[active]==x[ground]),key
    original=matrix(m)
    if self_off:
        m.geom_contype[active]=2;m.geom_conaffinity[active]=1
        m.geom_contype[ground]=1;m.geom_conaffinity[ground]=2
        expected=np.zeros_like(original);expected[ground]=original[ground];expected[:,ground]=original[:,ground]
        assert np.array_equal(matrix(m),expected)
    if priority:
        m.geom_priority[ground]=1;m.geom_friction[ground]=combined;m.geom_friction[ground,0]=float(mu)
    else:assert mu==1.6
    return ground

def runtime_model(path):
    m=BASE_MODEL(path);tag=c._active_task['ground_friction_diagnostic']
    treatment(m,tag['sliding_mu'],tag['self_off']);return m

c.e.runtime_model=runtime_model
c.OUT=ps.SOURCE_OUT/'ground_friction_diagnostic';c.DATA=c.OUT/'data';c.e.TRIALS=c.OUT/'trials'

def static_check(references):
    checks=[]
    fields=['geom','dist','pos','frame','dim','solref','solreffriction','solimp']
    for self_off in (False,True):
        for mu in (1.6,.5):
            ma=BASE_MODEL(c.PLANTS['current']);mb=BASE_MODEL(c.PLANTS['current'])
            ga=treatment(ma,1.6,self_off,False);gb=treatment(mb,mu,self_off)
            for ref in references[:2]:
                with np.load(ref['motion']) as z:qpos=z['qpos']
                for i in np.linspace(len(qpos)//2,len(qpos)-1,3,dtype=int):
                    da=c.mujoco.MjData(ma);db=c.mujoco.MjData(mb)
                    da.qpos[:]=qpos[i];db.qpos[:]=qpos[i]
                    c.mujoco.mj_forward(ma,da);c.mujoco.mj_forward(mb,db)
                    assert da.ncon==db.ncon
                    for key in fields:np.testing.assert_array_equal(getattr(da.contact,key),getattr(db.contact,key))
                    expected=da.contact.friction.copy();ground=(da.contact.geom1==ga)|(da.contact.geom2==ga)
                    expected[ground,:2]=mu
                    np.testing.assert_array_equal(db.contact.friction,expected)
                    checks.append(dict(mu=mu,self_off=self_off,n_contacts=da.ncon,n_ground_contacts=int(ground.sum()),only_ground_sliding_coefficients_changed=True))
    c.write(c.DATA/'static_check.json',dict(passed=True,physics_steps_executed=0,checks=checks))

def main(workers):
    references=[r for r in c.read(ps.SOURCE_DATA/'bridge_analysis.json')['rows'] if r['plant']=='current' and r['phase']=='midpoint' and r['family'] in ('recovery','rl_v4') and r['wave'] in (1.5,3.,4.5)]
    assert len(references)==6
    manifest=c.read(SOURCE_DATA/'manifest.json')
    manifest.update(diagnostic_only=True,no_pd_change=True,main_optimization_unchanged=True,
        treatment='Ground sliding friction1.6versus0.5, factorial with internal collisions on/off. Ground priority selects only friction; contact geometry, dimension, solref/solimp/margin/gap/solmix must match.',
        native_comparison_scope='0.5 comes from the original RL configuration/default-material source chain; not yet an actual native material readback.')
    for p in [Path(__file__),c.ROOT/'FRICTION_DIAGNOSTIC_PROTOCOL.md',ps.SOURCE_DATA/'bridge_analysis.json',ps.SOURCE_OUT/'self_collision_diagnostic/data/comparison.json']:
        manifest['frozen_inputs'][str(p)]=c.sha(p)
    path=c.DATA/'manifest.json'
    if path.exists():assert c.read(path)==manifest
    else:c.write(path,manifest)
    static_check(references)
    def task(ref,mu,self_off,stage):
        t=c.task(ref['wave'],[ref['amplitude'][k] for k in c.KEYS],stage,phase='midpoint',trace=True,prefix=False)
        t['ground_friction_diagnostic']=dict(sliding_mu=mu,self_off=self_off);return t
    baseline=[(r,off,task(r,1.6,off,'priority_parity')) for r in references if r['wave']==1.5 for off in (False,True)]
    records={r['id']:r for r in c.bank([t for _,_,t in baseline],'priority_parity',workers)}
    oldoff=c.read(ps.SOURCE_OUT/'self_collision_diagnostic/data/comparison.json')['comparisons'];parity=[]
    for ref,off,t in baseline:
        expected=next(r['self_off_D'] for r in oldoff if r['wave']==ref['wave'] and r['family']==ref['family']) if off else ref['D']
        row=records[c.identity(t)];assert abs(row['D']-expected)<1e-8,(row['D'],expected)
        parity.append(dict(wave=ref['wave'],family=ref['family'],self_off=off,D_difference_mm=row['D']-expected,passed=True))
    c.write(c.DATA/'priority_parity.json',dict(passed=True,checks=parity))
    planned=[(r,off,task(r,.5,off,'friction_matched')) for r in references for off in (False,True)]
    measured={r['id']:r for r in c.bank([t for _,_,t in planned],'friction_matched',workers)}
    result=[]
    for ref,off,t in planned:
        r=measured[c.identity(t)]
        old=next(x for x in oldoff if x['wave']==ref['wave'] and x['family']==ref['family'])
        source_D=old['self_off_D'] if off else ref['D']
        result.append(dict(wave=ref['wave'],family=ref['family'],self_off=off,amplitude=r['amplitudes'],
            high_mu_D_mm=source_D,low_mu_D_mm=r['D'],difference_mm=r['D']-source_D,
            low_mu_failed=r['failed'],record=r['record'],pd_readback=r['pd_readback']))
    c.write(c.DATA/'comparison.json',dict(diagnostic_only=True,comparisons=result,used_for_optimization=False,
        interpretation='Matched finite-amplitude factorial diagnosis. No new amplitudes optimized, no PD changed, no source-matched native material readback claimed.'))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=3);main(p.parse_args().workers)
