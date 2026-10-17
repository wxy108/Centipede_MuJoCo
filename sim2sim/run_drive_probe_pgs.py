"""Isolated fixed-base one-hinge probes: same physical PD and inertia in both engines."""
from pathlib import Path
import argparse,os,sys,json,time
ROOT=Path(__file__).resolve().parent
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--dt',type=float,default=.00025);p.add_argument('--inertia-mode',choices=['fixed','actual'],default='fixed')
AppLauncher.add_app_launcher_args(p);a=p.parse_args();folder=ROOT/'runs'/a.name;assert not folder.exists();folder.mkdir(parents=True)
(folder/'pid.json').write_text(json.dumps(dict(pid=os.getpid(),argv=sys.argv)))
app=AppLauncher(a).app
import numpy as np
import torch
import isaaclab.sim as su
from isaaclab.assets import Articulation,ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from pxr import Usd,UsdGeom,UsdPhysics,PhysxSchema,Gf
from scipy.linalg import expm
import shared as s
def main():
    cfg=su.SimulationCfg(dt=a.dt,device=a.device,gravity=(0.,0.,0.),render_interval=100000);cfg.physx.enable_stabilization=False;cfg.physx.solver_type=0
    sim=su.SimulationContext(cfg)
    rigs=[];mus=[];constants=[]
    for i,(family,gains) in enumerate(s.C['impedance_Nm_per_rad_and_Nms_per_rad'].items()):
        kp,kv=np.array(gains,np.float32).astype(float);I=1e-10;mass=1e-5
        if a.inertia_mode=='actual':I=next(x['free_median'] for x in json.loads((ROOT/'audit/joint_effective_inertia.json').read_text())['rows'] if x['family']==family)
        st=Usd.Stage.CreateNew(str(folder/f'{family}.usda'));root=UsdGeom.Xform.Define(st,'/Probe');st.SetDefaultPrim(root.GetPrim());UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
        for name in ['base','link']:
            b=UsdGeom.Xform.Define(st,f'/Probe/{name}');UsdPhysics.RigidBodyAPI.Apply(b.GetPrim());mp=UsdPhysics.MassAPI.Apply(b.GetPrim());mp.CreateMassAttr(mass);mp.CreateCenterOfMassAttr(Gf.Vec3f(0));mp.CreateDiagonalInertiaAttr(Gf.Vec3f(I));mp.CreatePrincipalAxesAttr(Gf.Quatf(1))
            rb=PhysxSchema.PhysxRigidBodyAPI.Apply(b.GetPrim());rb.CreateLinearDampingAttr(0.);rb.CreateAngularDampingAttr(0.)
        fixed=UsdPhysics.FixedJoint.Define(st,'/Probe/root_fixed');fixed.CreateBody1Rel().SetTargets(['/Probe/base'])
        j=UsdPhysics.RevoluteJoint.Define(st,'/Probe/hinge');j.CreateBody0Rel().SetTargets(['/Probe/base']);j.CreateBody1Rel().SetTargets(['/Probe/link']);j.CreateAxisAttr('Z');j.CreateLowerLimitAttr(-180.);j.CreateUpperLimitAttr(180.)
        dr=UsdPhysics.DriveAPI.Apply(j.GetPrim(),'angular');dr.CreateTypeAttr('force');dr.CreateStiffnessAttr(0.);dr.CreateDampingAttr(0.);dr.CreateMaxForceAttr(1.)
        st.GetRootLayer().Save()
        rc=ArticulationCfg(prim_path=f'/World/probe_{i}',spawn=su.UsdFileCfg(usd_path=str(folder/f'{family}.usda'),articulation_props=su.ArticulationRootPropertiesCfg(solver_position_iteration_count=16,solver_velocity_iteration_count=4,sleep_threshold=0.,stabilization_threshold=0.)),actuators={'pd':ImplicitActuatorCfg(joint_names_expr=['hinge'],stiffness=kp,damping=kv,armature=0.,friction=0.,effort_limit_sim=1.,velocity_limit_sim=1e6)})
        rigs.append(Articulation(rc));constants.append(dict(family=family,kp=kp,kv=kv,inertia=I,mass=mass))
        xml=f'<mujoco><option timestep="{a.dt}" gravity="0 0 0" integrator="implicitfast"/><worldbody><body name="link"><inertial pos="0 0 0" mass="{mass}" diaginertia="{I} {I} {I}"/><joint name="hinge" axis="0 0 1"/></body></worldbody><actuator><general joint="hinge" gainprm="{kp}" biastype="affine" biasprm="0 {-kp} {-kv}"/></actuator></mujoco>'
        (folder/f'{family}.xml').write_text(xml)
    sim.reset()
    for r in rigs:r.write_joint_state_to_sim(torch.zeros((1,1),device=sim.device),torch.zeros((1,1),device=sim.device));r.reset()
    sim.forward()
    ts=[0.];native=[np.zeros(len(rigs))];mu=[np.zeros((len(rigs),2))];ideal=[np.zeros(len(rigs))];states=[np.zeros(2) for r in rigs]
    transitions=[]
    for c in constants:
        mat=np.array([[0,1,0],[-c['kp']/c['inertia'],-c['kv']/c['inertia'],c['kp']/c['inertia']],[0,0,0.]])
        transitions.append(expm(mat*a.dt))
    for k in range(round(.3/a.dt)):
        target=float(np.float32(np.radians(1.) if k*a.dt>=.01 else 0.))
        for r in rigs:r.set_joint_position_target(torch.tensor([[target]],device=sim.device));r.set_joint_velocity_target(torch.zeros((1,1),device=sim.device));r.write_data_to_sim()
        sim.step(render=False)
        for r in rigs:r.update(a.dt)
        for i in range(len(rigs)):
            states[i]=(transitions[i]@np.r_[states[i],target])[:2]
        ts.append((k+1)*a.dt);native.append([float(r.data.joint_pos[0,0]) for r in rigs]);ideal.append([v[0] for v in states])
    native=np.array(native);ideal=np.array(ideal);rows=[]
    for i,c in enumerate(constants):
        denom=max(float(np.ptp(ideal[:,i])),1e-12)
        rows.append(dict(**c,native_analytic_nrmse=float(np.sqrt(np.mean((native[:,i]-ideal[:,i])**2))/denom)))
    np.savez_compressed(folder/'trajectory.npz',t=ts,native=native,ideal=ideal)
    s.write(folder/'results.json',dict(arguments=vars(a),rows=rows,source_sha256=s.sha(__file__),note='adapter changes discrete velocity coefficient by dt*K, not physical damping; diagnostic only'))
    print(json.dumps(rows),flush=True);s.write(folder/'complete.json',dict(complete=True));sys.stdout.flush();sys.stderr.flush();os._exit(0)
try:main()
except Exception:
    import traceback
    (folder/'error.txt').write_text(traceback.format_exc());traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
