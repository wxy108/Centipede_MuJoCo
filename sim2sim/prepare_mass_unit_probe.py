"""Prepare a unit-invariance diagnostic; this does not change physical SI parameters."""
from pathlib import Path
import os,sys
import shared as s
lib=Path(sys.executable).parent/'Lib/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+8131b85d.wx64.r.cp311'
sys.path.insert(0,str(lib));dll_handles=[os.add_dll_directory(str(lib/'bin'))]
from pxr import Usd,UsdPhysics,Gf
src=s.ROOT/'assets/common.usda';target=s.ROOT/'assets/common_massunit_mg.usda'
assert not target.exists();stage=Usd.Stage.Open(str(src));stage.GetRootLayer().Export(str(target));stage=Usd.Stage.Open(str(target))
count=0
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.MassAPI):
        api=UsdPhysics.MassAPI(prim);ma=api.GetMassAttr();ia=api.GetDiagonalInertiaAttr()
        if ma.HasAuthoredValueOpinion():ma.Set(ma.Get()*1e6);count+=1
        if ia.HasAuthoredValueOpinion():ia.Set(Gf.Vec3f(*(v*1e6 for v in ia.Get())))
stage.GetRootLayer().Save();assert count==251
source=(s.ROOT/'run_native_v6.py').read_text();changes=[
    ("AppLauncher.add_app_launcher_args(p);a=p.parse_args()","p.add_argument('--case-indices',type=int,nargs='+');p.add_argument('--mass-scale',type=float,choices=[1.,1e6],default=1.)\nAppLauncher.add_app_launcher_args(p);a=p.parse_args()"),
    ("rows=s.cases('calibration')+s.cases('holdout') if a.bank=='validation' else s.cases(a.bank);N=len(rows);started=time.monotonic()","full_rows=s.cases('calibration')+s.cases('holdout') if a.bank=='validation' else s.cases(a.bank);case_indices=list(range(len(full_rows))) if a.case_indices is None else a.case_indices;rows=[full_rows[i] for i in case_indices];N=len(rows);started=time.monotonic();asset=ROOT/'assets'/('common.usda' if a.mass_scale==1 else 'common_massunit_mg.usda')"),
    ("e*.25,0.","case_indices[e]*.25,0."),
    ("kp={x['name']:x['kp'] for x in s.M['joints']};kv={x['name']:x['kv'] for x in s.M['joints']}","kp={x['name']:x['kp']*a.mass_scale for x in s.M['joints']};kv={x['name']:x['kv']*a.mass_scale for x in s.M['joints']}"),
    ("usd_path=str(ROOT/'assets/common.usda')","usd_path=str(asset)"),
    ("effort_limit_sim=1.","effort_limit_sim=a.mass_scale"),
    ("torch.arange(N,device=sim.device)*.25","torch.tensor(case_indices,device=sim.device)*.25"),
    ("physics_dt_s=sim.get_physics_dt(),solver_type=","physics_dt_s=sim.get_physics_dt(),numerical_mass_scale=a.mass_scale,physical_mass_unit_kg=1/a.mass_scale,case_indices=case_indices,solver_type="),
    ("ROOT/'assets/common.usda',ROOT/'COMMON_CONTRACT.json'","asset,ROOT/'COMMON_CONTRACT.json'"),
    ("peak_pre_step_estimated_pd_effort_Nm=arr(max_effort).tolist()","peak_pre_step_estimated_pd_effort_Nm=(arr(max_effort)/a.mass_scale).tolist()")]
for old,new in changes:
    assert old in source,old;source=source.replace(old,new)
(s.ROOT/'run_native_v7.py').write_text(source,encoding='utf-8')
s.write(s.ROOT/'audit/mass_unit_probe_preparation.json',dict(body_count=count,numerical_scale=1e6,physical_mass_unit_kg=1e-6,mass_and_inertia_and_pd_and_force_cap_scaled=True,length_and_time_unchanged=True,physical_pd_fitting=False,asset_sha=s.sha(target),runner_sha=s.sha(s.ROOT/'run_native_v7.py')))
print('Prepared mg numerical mass-unit asset and subset runner; no physics advanced.')
