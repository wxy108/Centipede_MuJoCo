from pathlib import Path
import os,sys,json,hashlib,re,datetime,xml.etree.ElementTree as ET
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parent
MU=ROOT.parent/'Centipede_MUJOCO-main'
RL=ROOT.parent/'Centipede_IsaacLab_RL'
OLD=MU/'experiments/common_pd_amplitude_contact_2026-09-09'
NATIVE=RL/'outputs/analysis/common_pd_amplitude_contact_20260909'
XML=MU/'experiments/mujoco_tip_passive_gliding_2026-09-01/models/centipede_flat_localz_k2em05_d3p2_r38.xml'
USD=RL/'outputs/analysis/mujoco_transfer_refinement_20260902/assets/centipede_rebuild_renamed_tip_experiment.usda'
LIB=Path(sys.executable).parent/'Lib/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+8131b85d.wx64.r.cp311'
sys.path.insert(0,str(LIB));DLL_HANDLES=[os.add_dll_directory(str(LIB/'bin'))]
from pxr import Usd,UsdGeom,UsdPhysics,Gf
(ROOT/'audit').mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def safe(v):
    if isinstance(v,float) and not np.isfinite(v):return str(v)
    if v is None or isinstance(v,(str,bool,int,float)):return v
    if hasattr(v,'GetReal'):return [float(v.GetReal()),*map(float,v.GetImaginary())]
    try:return [safe(x) for x in v]
    except TypeError:return str(v)
def attrs(prim,prefix):return {a.GetName():safe(a.Get()) for a in prim.GetAttributes() if a.GetName().startswith(prefix)}
tree=ET.parse(XML);sz=tree.getroot().find('size');sz.attrib.clear();sz.set('memory','64M')
m=mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(),encoding='unicode'))
d=mujoco.MjData(m);mujoco.mj_kinematics(m,d)
mb=[]
for i in range(1,m.nbody):
    mb.append(dict(name=m.body(i).name,id=i,parent=m.body(int(m.body_parentid[i])).name,
                   weld_root=m.body(int(m.body_weldid[i])).name,mass=float(m.body_mass[i]),
                   center_of_mass=m.body_ipos[i].tolist(),principal_inertia=m.body_inertia[i].tolist(),
                   principal_quat=m.body_iquat[i].tolist(),local_pos=m.body_pos[i].tolist(),local_quat=m.body_quat[i].tolist(),
                   world_pos=d.xpos[i].tolist(),world_quat=d.xquat[i].tolist()))
mj=[]
for i in range(1,m.njnt):
    mj.append(dict(name=m.joint(i).name,body=m.body(int(m.jnt_bodyid[i])).name,axis=m.jnt_axis[i].tolist(),
                   anchor=m.jnt_pos[i].tolist(),range=m.jnt_range[i].tolist(),stiffness=float(m.jnt_stiffness[i]),
                   damping=float(m.dof_damping[m.jnt_dofadr[i]]),armature=float(m.dof_armature[m.jnt_dofadr[i]])))
mg=[]
for i in range(m.ngeom):
    if m.geom_group[i]!=0:continue
    mg.append(dict(name=m.geom(i).name,body=m.body(int(m.geom_bodyid[i])).name,type=int(m.geom_type[i]),
                   size=m.geom_size[i].tolist(),pos=m.geom_pos[i].tolist(),quat=m.geom_quat[i].tolist(),
                   condim=int(m.geom_condim[i]),friction=m.geom_friction[i].tolist(),
                   solref=m.geom_solref[i].tolist(),solimp=m.geom_solimp[i].tolist(),
                   margin=float(m.geom_margin[i]),gap=float(m.geom_gap[i]),
                   mesh_id=int(m.geom_dataid[i]) if m.geom_type[i]==mujoco.mjtGeom.mjGEOM_MESH else None))
stage=Usd.Stage.Open(str(USD));cache=UsdGeom.XformCache();ub=[];uj=[];ug=[]
for prim in Usd.PrimRange.Stage(stage,Usd.TraverseInstanceProxies()):
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        mat=cache.GetLocalToWorldTransform(prim);q=mat.ExtractRotationQuat()
        ub.append(dict(name=prim.GetName(),path=str(prim.GetPath()),**attrs(prim,('physics:','physxRigidBody:')),
                       world_pos=safe(mat.ExtractTranslation()),world_quat=safe(q)))
    if prim.IsA(UsdPhysics.Joint):
        uj.append(dict(name=prim.GetName(),path=str(prim.GetPath()),type=prim.GetTypeName(),
                       body0=[str(x) for x in UsdPhysics.Joint(prim).GetBody0Rel().GetTargets()],
                       body1=[str(x) for x in UsdPhysics.Joint(prim).GetBody1Rel().GetTargets()],
                       **attrs(prim,('physics:','physxJoint:','drive:'))))
    if prim.HasAPI(UsdPhysics.CollisionAPI):
        rb=prim
        while rb and not rb.HasAPI(UsdPhysics.RigidBodyAPI):rb=rb.GetParent()
        mat=cache.GetLocalToWorldTransform(prim)
        local=mat*cache.GetLocalToWorldTransform(rb).GetInverse()
        item=dict(path=str(prim.GetPath()),name=prim.GetName(),body=rb.GetName(),type=prim.GetTypeName(),
                  pos=safe(local.ExtractTranslation()),quat=safe(local.ExtractRotationQuat()),
                  **attrs(prim,('physics:','physxCollision:','radius','height','axis','points')))
        if 'points' in item:item['points_count']=len(item.pop('points'))
        ug.append(item)
by_mu={x['name']:x for x in mb};by_usd={x['name']:x for x in ub}
matched=[]
for name in sorted(by_mu.keys()&by_usd.keys()):
    a=by_mu[name];b=by_usd[name];mass=b.get('physics:mass')
    if mass is None:continue
    row=dict(name=name,mujoco_mass=a['mass'],usd_mass=mass,mass_rel_difference=(mass-a['mass'])/a['mass'] if a['mass'] else None,
             mujoco_inertia=a['principal_inertia'],usd_inertia=b.get('physics:diagonalInertia'),
             mujoco_com=a['center_of_mass'],usd_com=b.get('physics:centerOfMass'))
    if row['usd_inertia'] is not None:
        row['sorted_inertia_rel_max']=float(np.max(np.abs(np.sort(row['usd_inertia'])/np.sort(row['mujoco_inertia'])-1))) if min(row['mujoco_inertia'])>0 else None
    matched.append(row)
files=[XML,USD,RL/'envs/centipede_env.py',RL/'envs/centipede_env_cfg.py',RL/'controllers/impedance_controller_torch.py',RL/'controllers/kinematics.py',NATIVE/'code/runtime.py',OLD/'core.py',OLD/'outputs/data/manifest.json',NATIVE/'physical_contract_readback/backend_arrays.npz']
for folder in [MU/'controllers',MU/'configs',RL/'controllers',RL/'configs',RL/'envs']:
    files.extend(p for p in folder.rglob('*') if p.is_file() and p.suffix in ('.py','.yaml','.json') and '__pycache__' not in str(p))
payload=dict(time=datetime.datetime.now().astimezone().isoformat(),physics_steps=0,mujoco_version=mujoco.__version__,
             source_hashes={str(p):sha(p) for p in sorted(set(files))},
             mujoco=dict(nbody=m.nbody-1,njoints=m.njnt-1,nv=m.nv,mass=float(m.body_mass.sum()),
                dt=float(m.opt.timestep),integrator=int(m.opt.integrator),solver=int(m.opt.solver),iterations=int(m.opt.iterations),
                tolerance=float(m.opt.tolerance),bodies=mb,joints=mj,geoms=mg),
             usd=dict(stage_units=UsdGeom.GetStageMetersPerUnit(stage),bodies=ub,joints=uj,geoms=ug,
                nbody=len(ub),n_joints=len(uj),moving_joints=sum(x['type']!='PhysicsFixedJoint' for x in uj),
                authored_mass=sum(x.get('physics:mass',0) for x in ub)),
             named_body_comparison=matched,
             caution='Named-body inertia differences are not yet welded-group differences. USD values are composed authored values; fresh backend inertia readback remains required.')
(ROOT/'audit/static_inventory.json').write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps(dict(mujoco={k:payload['mujoco'][k] for k in ['nbody','njoints','nv','mass','dt']},
    usd={k:payload['usd'][k] for k in ['nbody','n_joints','moving_joints','authored_mass']},
    matched=len(matched),mu_only=sorted(by_mu.keys()-by_usd.keys()),usd_only=sorted(by_usd.keys()-by_mu.keys()),
    mass_differences=sorted([x for x in matched if x['mass_rel_difference'] is not None],key=lambda x:abs(x['mass_rel_difference']),reverse=True)[:8],
    joint_examples=uj[:3]),indent=2))
