"""Build paired assets from the audited composed native plant, without physics steps."""
from pathlib import Path
import os,sys,json,hashlib,re,collections,xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation as R
import mujoco
ROOT=Path(__file__).resolve().parent
assert (ROOT/'AUDIT.md').exists()
C=json.loads((ROOT/'COMMON_CONTRACT.json').read_text())
A=json.loads((ROOT/'audit/static_inventory.json').read_text())
SOURCE=ROOT.parent/'Centipede_IsaacLab_RL/outputs/analysis/mujoco_transfer_refinement_20260902/assets/centipede_rebuild_renamed_tip_experiment.usda'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==A['source_hashes'][str(SOURCE)]
LIB=Path(sys.executable).parent/'Lib/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+8131b85d.wx64.r.cp311'
sys.path.insert(0,str(LIB))
DLL_HANDLES=[os.add_dll_directory(str(LIB/'bin'))]
from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,Sdf,Gf
DEST=ROOT/'assets';DEST.mkdir(exist_ok=True);(DEST/'meshes').mkdir(exist_ok=True)
def vals(x):return ' '.join(format(float(v),'.17g') for v in np.asarray(x).ravel())
def quat(q):
    q=np.array(q,float)
    if np.linalg.norm(q)<1e-12:return np.array([1.,0,0,0])
    return q/np.linalg.norm(q)
def rot(q):q=quat(q);return R.from_quat(q[[1,2,3,0]]).as_matrix()
def wxyz(mat):q=R.from_matrix(mat).as_quat();return q[[3,0,1,2]]
def family(name):
    if name=='body_pitch_0':return 'head_pitch'
    if name.startswith('tarsus_'):return 'tip'
    return re.sub(r'_\d+(?:_[LR])?$','',name)
def physx(prim,schema,prefix,fields):
    prim.AddAppliedSchema(schema)
    for key,value in fields.items():
        typ=Sdf.ValueTypeNames.Bool if isinstance(value,bool) else Sdf.ValueTypeNames.Int if isinstance(value,int) else Sdf.ValueTypeNames.Token if isinstance(value,str) else Sdf.ValueTypeNames.Float
        prim.CreateAttribute(prefix+':'+key,typ,custom=False).Set(value)
stage=Usd.Stage.Open(Usd.Stage.Open(str(SOURCE)).Flatten())
while True:
    instances=[p for p in stage.Traverse() if p.IsInstance()]
    if not instances:break
    for p in instances:p.SetInstanceable(False)
rootprim=stage.GetDefaultPrim()
mat=UsdShade.Material.Define(stage,str(rootprim.GetPath())+'/commonMaterial')
mp=UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
mp.CreateStaticFrictionAttr(C['material']['static_friction']);mp.CreateDynamicFrictionAttr(C['material']['dynamic_friction']);mp.CreateRestitutionAttr(0.)
physx(mat.GetPrim(),'PhysxMaterialAPI','physxMaterial',dict(frictionCombineMode='average'))
for p in stage.Traverse():
    if p.HasAPI(UsdPhysics.RigidBodyAPI):
        physx(p,'PhysxRigidBodyAPI','physxRigidBody',dict(linearDamping=0.,angularDamping=0.,maxLinearVelocity=1000.,maxAngularVelocity=1e8,sleepThreshold=0.,stabilizationThreshold=0.))
    if p.HasAPI(UsdPhysics.ArticulationRootAPI):
        physx(p,'PhysxArticulationAPI','physxArticulation',dict(enabledSelfCollisions=False,solverPositionIterationCount=16,solverVelocityIterationCount=4,sleepThreshold=0.,stabilizationThreshold=0.))
    if p.HasAPI(UsdPhysics.CollisionAPI):
        physx(p,'PhysxCollisionAPI','physxCollision',dict(contactOffset=C['candidate_numerics']['native_contact_offset_per_shape_m'],restOffset=0.))
        UsdShade.MaterialBindingAPI.Apply(p).Bind(mat,UsdShade.Tokens.weakerThanDescendants,'physics')
    if p.IsA(UsdPhysics.RevoluteJoint):
        physx(p,'PhysxJointAPI','physxJoint',dict(armature=0.,jointFriction=0.,maxJointVelocity=C['joint_velocity_limit_rad_s']))
        # Runtime tensor writes use SI-radian gains and verify actual PhysX readback.
        drive=UsdPhysics.DriveAPI.Apply(p,'angular');drive.CreateTypeAttr('force')
        drive.CreateStiffnessAttr(0);drive.CreateDampingAttr(0);drive.CreateMaxForceAttr(1)
        drive.CreateTargetPositionAttr(0);drive.CreateTargetVelocityAttr(0)
stage.GetRootLayer().Export(str(DEST/'common.usda'))

bodies={b['name']:b for b in A['usd']['bodies']}
incoming={};children=collections.defaultdict(list)
for joint in A['usd']['joints']:
    assert len(joint['body0'])==len(joint['body1'])==1
    parent=joint['body0'][0].split('/')[-1];child=joint['body1'][0].split('/')[-1]
    assert child not in incoming
    incoming[child]=joint;children[parent].append(child)
roots=set(bodies)-set(incoming);assert roots=={'link_body_0'},roots
world={n:(np.array(b['world_pos']),rot(b['world_quat'])) for n,b in bodies.items()}
base=ET.Element('mujoco',model='centipede_sim2sim_common_v1')
ET.SubElement(base,'compiler',angle='radian',inertiafromgeom='false',balanceinertia='false',fusestatic='false',meshdir=str(DEST/'meshes'))
ET.SubElement(base,'size',memory='64M')
ET.SubElement(base,'option',timestep='.00025',gravity='0 0 -9.81',integrator='implicitfast',cone='elliptic',iterations='200',tolerance='1e-12')
default=ET.SubElement(base,'default')
ET.SubElement(default,'joint',damping='0',armature='0',frictionloss='0')
ET.SubElement(default,'geom',contype='1',conaffinity='2',condim='3',friction='.5 0 0',margin='0',gap='0',solref=vals(C['candidate_numerics']['mujoco_solref']),solimp=vals(C['candidate_numerics']['mujoco_solimp']))
assets=ET.SubElement(base,'asset');wb=ET.SubElement(base,'worldbody');act=ET.SubElement(base,'actuator')
ET.SubElement(wb,'geom',name='ground',type='plane',size='1 1 .01',contype='2',conaffinity='1',rgba='.7 .7 .7 1')
elements={};joint_defs=[]
def add_body(name,parent_element,parent_name=None):
    b=bodies[name];pos,rm=world[name]
    if parent_name:
        ppos,pr=world[parent_name];localpos=pr.T@(pos-ppos);localrot=pr.T@rm
    else:localpos=pos;localrot=rm
    elem=ET.SubElement(parent_element,'body',name=name,pos=vals(localpos),quat=vals(wxyz(localrot)))
    elements[name]=elem
    q=quat(b['physics:principalAxes'])
    if np.linalg.norm(b['physics:principalAxes'])<1e-12:
        assert np.ptp(b['physics:diagonalInertia'])<1e-20
    ET.SubElement(elem,'inertial',mass=str(b['physics:mass']),pos=vals(b['physics:centerOfMass']),diaginertia=vals(b['physics:diagonalInertia']),quat=vals(q))
    if not parent_name:ET.SubElement(elem,'freejoint',name='root')
    else:
        j=incoming[name]
        if j['type']=='PhysicsRevoluteJoint':
            axis=rot(j['physics:localRot1'])@np.eye(3)['XYZ'.index(j['physics:axis'])]
            lower,upper=np.radians([j['physics:lowerLimit'],j['physics:upperLimit']])
            ET.SubElement(elem,'joint',name=j['name'],type='hinge',axis=vals(axis),pos=vals(j['physics:localPos1']),limited='true',range=vals([lower,upper]),solreflimit='.0005 1',solimplimit='.9999 .9999 .00001 .5 2')
            kp,kv=C['impedance_Nm_per_rad_and_Nms_per_rad'][family(j['name'])]
            ET.SubElement(act,'general',name=j['name'],joint=j['name'],gainprm=vals([kp,0,0]),biasprm=vals([0,-kp,-kv]),biastype='affine',gaintype='fixed',forcelimited='true',forcerange='-1 1')
            joint_defs.append(dict(name=j['name'],family=family(j['name']),kp=kp,kv=kv,axis=axis.tolist(),range_rad=[lower,upper],body=name))
        else:assert j['type']=='PhysicsFixedJoint'
    for child in children[name]:add_body(child,elem,name)
add_body('link_body_0',wb)
cache=UsdGeom.XformCache();geometry=[]
for i,g in enumerate(A['usd']['geoms']):
    p=stage.GetPrimAtPath(g['path']);assert p.IsValid()
    rb=bodies[g['body']];rprim=stage.GetPrimAtPath(rb['path'])
    local=cache.GetLocalToWorldTransform(p)*cache.GetLocalToWorldTransform(rprim).GetInverse()
    matrix=np.array(local)
    if g['type']!='Mesh':assert np.allclose(np.linalg.norm(matrix[:3,:3],axis=1),1,atol=1e-6),(g['path'],matrix)
    center=np.array(local.ExtractTranslation());q=local.ExtractRotationQuat();qr=quat([q.GetReal(),*q.GetImaginary()])
    name=f'collision_{i:03d}'
    kw=dict(name=name,pos=vals(center),quat=vals(qr),rgba='.65 .26 .08 1')
    if g['type']=='Sphere':kw.update(type='sphere',size=str(float(UsdGeom.Sphere(p).GetRadiusAttr().Get())))
    elif g['type']=='Capsule':
        cap=UsdGeom.Capsule(p);axis=np.eye(3)['XYZ'.index(cap.GetAxisAttr().Get())]
        radius=float(cap.GetRadiusAttr().Get());half=.5*float(cap.GetHeightAttr().Get())
        ends=[center-half*rot(qr)@axis,center+half*rot(qr)@axis]
        kw.pop('pos');kw.pop('quat');kw.update(type='capsule',fromto=vals(ends),size=str(radius))
    else:
        assert g['type']=='Mesh'
        mesh=UsdGeom.Mesh(p);verts=np.array(mesh.GetPointsAttr().Get(),float)
        verts=(np.c_[verts,np.ones(len(verts))]@matrix)[:,:3]
        kw['pos']='0 0 0';kw['quat']='1 0 0 0'
        counts=list(mesh.GetFaceVertexCountsAttr().Get());indices=list(mesh.GetFaceVertexIndicesAttr().Get())
        path=DEST/'meshes'/f'{name}.obj'
        lines=['v '+vals(v) for v in verts];cursor=0
        for n in counts:
            face=indices[cursor:cursor+n];cursor+=n
            for k in range(1,n-1):lines.append('f '+' '.join(str(int(v)+1) for v in [face[0],face[k],face[k+1]]))
        path.write_text('\n'.join(lines)+'\n',encoding='ascii')
        ET.SubElement(assets,'mesh',name=name,file=path.name)
        kw.update(type='mesh',mesh=name)
    ET.SubElement(elements[g['body']],'geom',**kw)
    geometry.append(dict(id=i,mujoco_name=name,usd_path=g['path'],body=g['body'],type=g['type']))
ET.indent(base)
xml=DEST/'common.xml';ET.ElementTree(base).write(xml,encoding='utf-8',xml_declaration=True)
m=mujoco.MjModel.from_xml_path(str(xml));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
assert m.njnt-1==173 and m.nbody-1==251
assert abs(m.body_mass.sum()-sum(b['physics:mass'] for b in bodies.values()))<1e-14
maxpos=0;maxmass=0;maxinertia=0
for name,b in bodies.items():
    idx=m.body(name).id
    maxpos=max(maxpos,float(np.max(np.abs(d.xpos[idx]-b['world_pos']))))
    maxmass=max(maxmass,abs(m.body_mass[idx]/b['physics:mass']-1))
    # Compare full tensors, not an ordering-sensitive list of principal moments.
    u=rot(b['physics:principalAxes']);mu=rot(m.body_iquat[idx])
    ui=u@np.diag(b['physics:diagonalInertia'])@u.T;mi=mu@np.diag(m.body_inertia[idx])@mu.T
    maxinertia=max(maxinertia,float(np.linalg.norm(mi-ui)/np.linalg.norm(ui)))
assert maxpos<1e-7 and maxmass<1e-12 and maxinertia<1e-9,(maxpos,maxmass,maxinertia)
gap=[]
for i in range(1,m.ngeom):
    typ=m.geom_type[i];pos=d.geom_xpos[i];rm=d.geom_xmat[i].reshape(3,3)
    if typ==mujoco.mjtGeom.mjGEOM_SPHERE:v=pos[2]-m.geom_size[i,0]
    elif typ==mujoco.mjtGeom.mjGEOM_CAPSULE:v=pos[2]-abs(rm[2,2])*m.geom_size[i,1]-m.geom_size[i,0]
    else:
        mid=m.geom_dataid[i];start=m.mesh_vertadr[mid];n=m.mesh_vertnum[mid]
        v=float(np.min((m.mesh_vert[start:start+n]@rm.T+pos)[:,2]))
    gap.append(v)
init=d.qpos[:7].copy();init[2]+=C['initialization']['minimum_collider_plane_clearance_m']-min(gap)
mapping=dict(joints=joint_defs,geoms=geometry,body_names=list(bodies),initial_root_pose_wxyz=init.tolist(),
             source_usd_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
             common_usd_sha256=hashlib.sha256((DEST/'common.usda').read_bytes()).hexdigest(),
             common_xml_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),
             contract_sha256=hashlib.sha256((ROOT/'COMMON_CONTRACT.json').read_bytes()).hexdigest(),
             static_checks=dict(bodies=251,hinge_joints=173,total_mass_kg=float(m.body_mass.sum()),max_world_position_error_m=maxpos,max_relative_mass_error=maxmass,max_relative_inertia_tensor_error=maxinertia))
(DEST/'mapping.json').write_text(json.dumps(mapping,indent=2),encoding='utf-8')
print(json.dumps(mapping['static_checks'],indent=2));print('initial root',init.tolist())
