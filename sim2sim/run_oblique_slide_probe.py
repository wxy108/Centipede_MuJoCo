"""Independent oblique Coulomb slide, with rotation locked and three free translations."""
from pathlib import Path
import argparse,sys,os,json
ROOT=Path(__file__).resolve().parent
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--dt',type=float,default=.0000625);p.add_argument('--solver',type=int,default=0)
AppLauncher.add_app_launcher_args(p);a=p.parse_args();folder=ROOT/'runs'/a.name;assert not folder.exists();folder.mkdir(parents=True);(folder/'pid.json').write_text(json.dumps(dict(pid=os.getpid(),argv=sys.argv)))
app=AppLauncher(a).app
import numpy as np,torch
import isaaclab.sim as su
from isaaclab.assets import Articulation,ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,PhysxSchema,Gf
import shared as s

def main():
    radius=.00045;mass=1e-6;carrier=1e-10;inertia=.4*mass*radius**2
    cfg=su.SimulationCfg(dt=a.dt,device=a.device,gravity=(0.,0.,-9.81),render_interval=100000);cfg.physx.enable_stabilization=False;cfg.physx.friction_offset_threshold=.00004;cfg.physx.friction_correlation_distance=.00005;cfg.physx.solver_type=a.solver
    cfg.worker_thread_count=4
    sim=su.SimulationContext(cfg)
    mat=UsdShade.Material.Define(sim.stage,'/World/Material');ma=UsdPhysics.MaterialAPI.Apply(mat.GetPrim());ma.CreateStaticFrictionAttr(.5);ma.CreateDynamicFrictionAttr(.5);ma.CreateRestitutionAttr(0.)
    plane=UsdGeom.Plane.Define(sim.stage,'/World/Ground');plane.CreateAxisAttr('Z');UsdPhysics.CollisionAPI.Apply(plane.GetPrim());pc=PhysxSchema.PhysxCollisionAPI.Apply(plane.GetPrim());pc.CreateContactOffsetAttr(.00002);pc.CreateRestOffsetAttr(0.);UsdShade.MaterialBindingAPI.Apply(plane.GetPrim()).Bind(mat,UsdShade.Tokens.weakerThanDescendants,'physics')
    st=Usd.Stage.CreateNew(str(folder/'probe.usda'));root=UsdGeom.Xform.Define(st,'/Probe');st.SetDefaultPrim(root.GetPrim());UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    for n in ['base','carrier_z','carrier_y','ball']:
        b=UsdGeom.Xform.Define(st,f'/Probe/{n}');UsdPhysics.RigidBodyAPI.Apply(b.GetPrim());mp=UsdPhysics.MassAPI.Apply(b.GetPrim());mp.CreateMassAttr(mass if n=='ball' else carrier);mp.CreateCenterOfMassAttr(Gf.Vec3f(0.));mp.CreateDiagonalInertiaAttr(Gf.Vec3f(inertia if n=='ball' else 1e-15));mp.CreatePrincipalAxesAttr(Gf.Quatf(1.));rb=PhysxSchema.PhysxRigidBodyAPI.Apply(b.GetPrim());rb.CreateLinearDampingAttr(0.);rb.CreateAngularDampingAttr(0.)
    fixed=UsdPhysics.FixedJoint.Define(st,'/Probe/fixed');fixed.CreateBody1Rel().SetTargets(['/Probe/base'])
    for n,par,ch,axis in [('z','base','carrier_z','Z'),('y','carrier_z','carrier_y','Y'),('x','carrier_y','ball','X')]:
        j=UsdPhysics.PrismaticJoint.Define(st,f'/Probe/{n}');j.CreateBody0Rel().SetTargets([f'/Probe/{par}']);j.CreateBody1Rel().SetTargets([f'/Probe/{ch}']);j.CreateAxisAttr(axis);dr=UsdPhysics.DriveAPI.Apply(j.GetPrim(),'linear');dr.CreateTypeAttr('force');dr.CreateStiffnessAttr(0.);dr.CreateDampingAttr(0.);dr.CreateMaxForceAttr(1.)
    shape=UsdGeom.Sphere.Define(st,'/Probe/ball/shape');shape.CreateRadiusAttr(radius);UsdPhysics.CollisionAPI.Apply(shape.GetPrim());pc=PhysxSchema.PhysxCollisionAPI.Apply(shape.GetPrim());pc.CreateContactOffsetAttr(.00002);pc.CreateRestOffsetAttr(0.);st.GetRootLayer().Save()
    rc=ArticulationCfg(prim_path='/World/Probe',spawn=su.UsdFileCfg(usd_path=str(folder/'probe.usda'),articulation_props=su.ArticulationRootPropertiesCfg(enabled_self_collisions=False,solver_position_iteration_count=16,solver_velocity_iteration_count=4,sleep_threshold=0.,stabilization_threshold=0.)),actuators={'free':ImplicitActuatorCfg(joint_names_expr=['.*'],stiffness=0.,damping=0.,armature=0.,friction=0.,effort_limit_sim=1.,velocity_limit_sim=1e6)})
    robot=Articulation(rc);UsdShade.MaterialBindingAPI.Apply(sim.stage.GetPrimAtPath('/World/Probe/ball/shape')).Bind(mat,UsdShade.Tokens.weakerThanDescendants,'physics');sim.reset()
    names=robot.joint_names;iz=names.index('z');ix=names.index('x');iy=names.index('y');q=torch.zeros((1,3),device=sim.device);v=q.clone();q[0,iz]=radius;v[0,ix]=.05*np.cos(np.pi/6);v[0,iy]=.05*np.sin(np.pi/6);robot.write_joint_state_to_sim(q,v);robot.reset();sim.forward();robot.update(0.)
    def arr(t):return t.detach().cpu().numpy().copy()
    ts=[0.];qs=[arr(robot.data.joint_pos)[0]];vs=[arr(robot.data.joint_vel)[0]]
    for k in range(round(.025/a.dt)):
        robot.write_data_to_sim();sim.step(render=False);robot.update(a.dt);ts.append((k+1)*a.dt);qs.append(arr(robot.data.joint_pos)[0]);vs.append(arr(robot.data.joint_vel)[0])
    np.savez_compressed(folder/'trajectory.npz',t=ts,q=qs,v=vs,joint_names=names)
    np.savez_compressed(folder/'backend.npz',masses=arr(robot.root_physx_view.get_masses()),material_properties=arr(robot.root_physx_view.get_material_properties()),contact_offsets=arr(robot.root_physx_view.get_contact_offsets()),rest_offsets=arr(robot.root_physx_view.get_rest_offsets()))
    xml=f'<mujoco><option timestep="{a.dt}" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic" iterations="200" tolerance="1e-12"/><default><geom friction=".5 0 0" condim="3" solref="{2*a.dt} 1" solimp=".9999 .9999 .00001 .5 2"/></default><worldbody><geom type="plane" size="1 1 .01"/><body><inertial mass="{carrier}" diaginertia="1e-15 1e-15 1e-15" pos="0 0 0"/><joint type="slide" name="z" axis="0 0 1"/><body><inertial mass="{carrier}" diaginertia="1e-15 1e-15 1e-15" pos="0 0 0"/><joint type="slide" name="y" axis="0 1 0"/><body><inertial mass="{mass}" diaginertia="{inertia} {inertia} {inertia}" pos="0 0 0"/><joint type="slide" name="x" axis="1 0 0"/><geom type="sphere" size="{radius}"/></body></body></body></worldbody></mujoco>'
    (folder/'probe.xml').write_text(xml);s.write(folder/'results.json',dict(radius=radius,mass=mass,carrier_mass=carrier,initial_speed=.05,angle_deg=30,dt=a.dt,solver=a.solver,final_x=float(qs[-1][ix]),final_y=float(qs[-1][iy]),source_sha=s.sha(__file__)));s.write(folder/'complete.json',dict(complete=True));print((folder/'results.json').read_text(),flush=True);sys.stdout.flush();sys.stderr.flush();os._exit(0)
try:main()
except Exception:
    import traceback
    (folder/'error.txt').write_text(traceback.format_exc());traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
