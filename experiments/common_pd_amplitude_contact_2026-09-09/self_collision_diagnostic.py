"""Matched model diagnosis only: internal collisions on versus off, same PD."""
import argparse,json
from pathlib import Path
import physical_search as ps
import core as c
from diagnose import harmonic
np=c.np
BASE_MODEL=c.e.runtime_model
SOURCE_DATA=c.DATA
RL_ASSET=Path(r'C:\Users\wxy22\Documents\Centipede_IsaacLab_RL\outputs\analysis\mujoco_transfer_refinement_20260902\assets\centipede_rebuild_renamed_tip_experiment.usda')


def matrix(a,b):return ((a[:,None]&b[None,:])!=0)|((a[None,:]&b[:,None])!=0)
def runtime_model(path):
    m=BASE_MODEL(path)
    planes=np.flatnonzero(m.geom_type==int(c.mujoco.mjtGeom.mjGEOM_PLANE))
    assert len(planes)==1
    ground=int(planes[0])
    a=m.geom_contype.copy();b=m.geom_conaffinity.copy();old=matrix(a,b)
    active=(a!=0)|(b!=0);assert np.all(a[active]==1) and np.all(b[active]==1)
    m.geom_contype[active]=2;m.geom_conaffinity[active]=1
    m.geom_contype[ground]=1;m.geom_conaffinity[ground]=2
    expected=np.zeros_like(old);expected[ground,:]=old[ground,:];expected[:,ground]=old[:,ground]
    # Ground's compatibility with itself is not a possible collision pair.
    np.fill_diagonal(expected,False);actual=matrix(m.geom_contype,m.geom_conaffinity);np.fill_diagonal(actual,False)
    assert np.array_equal(actual,expected)
    return m


c.e.runtime_model=runtime_model
c.OUT=ps.SOURCE_OUT/'self_collision_diagnostic';c.DATA=c.OUT/'data';c.e.TRIALS=c.OUT/'trials'


def main(workers):
    assert 'bool physxArticulation:enabledSelfCollisions = 0' in RL_ASSET.read_text(encoding='utf-8')
    manifest=c.read(SOURCE_DATA/'manifest.json')
    manifest.update(diagnostic_only=True,treatment='All robot internal collisions off; every original ground pair preserved',
        no_pd_change=True,rl_asset_explicit_self_collisions=False)
    for p in (Path(__file__),RL_ASSET):manifest['frozen_inputs'][str(p)]=c.sha(p)
    path=c.DATA/'manifest.json'
    if path.exists():assert c.read(path)==manifest
    else:c.write(path,manifest)
    before=[r for r in c.read(ps.SOURCE_DATA/'bridge_analysis.json')['rows'] if r['plant']=='current' and r['phase']=='midpoint' and r['family'] in ('recovery','rl_v4') and r['wave'] in (1.5,3.,4.5)]
    tasks=[]
    for r in before:
        t=c.task(r['wave'],[r['amplitude'][k] for k in c.KEYS],'self_collision_matched',phase='midpoint',trace=True,prefix=False)
        t['internal_collisions']='off_match_original_rl_asset';tasks.append(t)
    plan=c.DATA/'matched_plan.json'
    if not plan.exists():c.write(plan,dict(references=before,tasks=tasks))
    rows={r['id']:r for r in c.bank(tasks,'self_collision_matched',workers)};comparisons=[]
    for original,t in zip(before,tasks):
        r=rows[c.identity(t)];z=np.load(r['motion_path']);good=(z['t']>=2)&(z['t']<6)
        body=float(np.median(harmonic(z['t'][good],z['bq'][good,1:,0])))
        comparisons.append(dict(wave=r['wave'],family=original['family'],amplitude=r['amplitudes'],
            self_on_D=original['D'],self_off_D=r['D'],difference_mm=r['D']-original['D'],
            self_on_body_measured_deg=original['realized_body_harmonic_median_deg'],self_off_body_measured_deg=body,
            shared_engineering_failures_self_on=[f for f in original['failed'] if f!='spine'],
            shared_engineering_failures_self_off=[f for f in r['failed'] if f!='nonadjacent_chassis_contact'],
            reference_spine_gate_failed='spine' in original['failed'],record=r['record'],
            actual_pd_sha256=r['pd_readback']['active_pd_sha256']))
    c.write(c.DATA/'comparison.json',dict(diagnostic_only=True,comparisons=comparisons,
        interpretation='The changed variable is collision semantics. This is not a new amplitude optimum and is not pooled with the main plant.'))
    print(json.dumps(comparisons,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=6);main(p.parse_args().workers)
