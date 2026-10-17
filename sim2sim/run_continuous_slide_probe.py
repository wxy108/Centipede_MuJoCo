"""Matched small sphere, fixed rotation, free vertical/horizontal translation: drop and slide."""
from pathlib import Path
import argparse,sys,os,json
ROOT=Path(__file__).resolve().parent
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--dt',type=float,default=.0000625);p.add_argument('--friction-offset',type=float,default=.00004);p.add_argument('--contact-offset',type=float,default=.00002);p.add_argument('--pos-iters',type=int,default=16);p.add_argument('--vel-iters',type=int,default=4)
AppLauncher.add_app_launcher_args(p);a=p.parse_args();folder=ROOT/'runs'/a.name;assert not folder.exists();folder.mkdir(parents=True);(folder/'pid.json').write_text(json.dumps(dict(pid=os.getpid(),argv=sys.argv)))
app=AppLauncher(a).app
import numpy as np,torch
import isaaclab.sim as su
from isaaclab.assets import Articulation,ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,PhysxSchema,Gf
import shared as s
def main():
    radius=.00045;mass=1e-6;carrier=1e-9;inertia=.4*mass*radius**2;clearance=.0002
    cfg=su.SimulationCfg(dt=a.dt,device=a.device,gravity=(0.,0.,-9.81),render_interval=100000);cfg.physx.enable_stabilization=False;cfg.physx.friction_offset_threshold=a.friction_offset;cfg.physx.friction_correlation_distance=.00005
    cfg.physx.solver_type=1
    sim=su.SimulationContext(cfg);scene=next(p for p in sim.stage.Traverse() if p.IsA(UsdPhysics.Scene));PhysxSchema.PhysxSceneAPI.Apply(scene).CreateEnableExternalForcesEveryIterationAttr(True)
    mat=UsdShade.Material.Define(sim.stage,'/World/Material');ma=UsdPhysics.MaterialAPI.Apply(mat.GetPrim());ma.CreateStaticFrictionAttr(.5);ma.CreateDynamicFrictionAttr(.5);ma.CreateRestitutionAttr(0.)
    plane=UsdGeom.Plane.Define(sim.stage,'/World/Ground');plane.CreateAxisAttr('Z');UsdPhysics.CollisionAPI.Apply(plane.GetPrim());pc=PhysxSchema.PhysxCollisionAPI.Apply(plane.GetPrim());pc.CreateContactOffsetAttr(a.contact_offset);pc.CreateRestOffsetAttr(0.);UsdShade.MaterialBindingAPI.Apply(plane.GetPrim()).Bind(mat,UsdShade.Tokens.weakerThanDescendants,'physics')
    st=Usd.Stage.CreateNew(str(folder/'probe.usda'));root=UsdGeom.Xform.Define(st,'/Probe');st.SetDefaultPrim(root.GetPrim());UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    for n in ['base','carrier','ball']:
        b=UsdGeom.Xform.Define(st,f'/Probe/{n}');UsdPhysics.RigidBodyAPI.Apply(b.GetPrim());mp=UsdPhysics.MassAPI.Apply(b.GetPrim());mp.CreateMassAttr(mass if n=='ball' else carrier);mp.CreateCenterOfMassAttr(Gf.Vec3f(0.));mp.CreateDiagonalInertiaAttr(Gf.Vec3f(inertia if n=='ball' else 1e-15));mp.CreatePrincipalAxesAttr(Gf.Quatf(1.));rb=PhysxSchema.PhysxRigidBodyAPI.Apply(b.GetPrim());rb.CreateLinearDampingAttr(0.);rb.CreateAngularDampingAttr(0.)
    fixed=UsdPhysics.FixedJoint.Define(st,'/Probe/fixed');fixed.CreateBody1Rel().SetTargets(['/Probe/base'])
    for n,par,ch,axis in [('z','base','carrier','Z'),('x','carrier','ball','X')]:
        j=UsdPhysics.PrismaticJoint.Define(st,f'/Probe/{n}');j.CreateBody0Rel().SetTargets([f'/Probe/{par}']);j.CreateBody1Rel().SetTargets([f'/Probe/{ch}']);j.CreateAxisAttr(axis);dr=UsdPhysics.DriveAPI.Apply(j.GetPrim(),'linear');dr.CreateTypeAttr('force');dr.CreateStiffnessAttr(0.);dr.CreateDampingAttr(0.);dr.CreateMaxForceAttr(1.)
    shape=UsdGeom.Sphere.Define(st,'/Probe/ball/shape');shape.CreateRadiusAttr(radius);UsdPhysics.CollisionAPI.Apply(shape.GetPrim());pc=PhysxSchema.PhysxCollisionAPI.Apply(shape.GetPrim());pc.CreateContactOffsetAttr(a.contact_offset);pc.CreateRestOffsetAttr(0.)
    st.GetRootLayer().Save()
    rc=ArticulationCfg(prim_path='/World/Probe',spawn=su.UsdFileCfg(usd_path=str(folder/'probe.usda'),articulation_props=su.ArticulationRootPropertiesCfg(enabled_self_collisions=False,solver_position_iteration_count=a.pos_iters,solver_velocity_iteration_count=a.vel_iters,sleep_threshold=0.,stabilization_threshold=0.)),actuators={'free':ImplicitActuatorCfg(joint_names_expr=['.*'],stiffness={'x':100.,'z':0.},damping={'x':.01,'z':0.},armature=0.,friction=0.,effort_limit_sim=1.,velocity_limit_sim=1e6)})
    robot=Articulation(rc);UsdShade.MaterialBindingAPI.Apply(sim.stage.GetPrimAtPath('/World/Probe/ball/shape')).Bind(mat,UsdShade.Tokens.weakerThanDescendants,'physics');sim.reset()
    names=robot.joint_names;iz=names.index('z');ix=names.index('x');q=torch.zeros((1,2),device=sim.device);v=q.clone();q[0,iz]=radius;v[0,ix]=0.;robot.write_joint_state_to_sim(q,v);robot.reset();sim.forward();robot.update(0.)
    def arr(t):return t.detach().cpu().numpy().copy()
    ts=[0.];qs=[arr(robot.data.joint_pos)[0]];vs=[arr(robot.data.joint_vel)[0]]
    for k in range(round(.03/a.dt)):
        at=k*a.dt;target=q.clone();target.zero_();target[0,ix]=.03*at;target[0,iz]=0.;velocity=target.clone();velocity.zero_();velocity[0,ix]=.03;velocity[0,iz]=0.;robot.set_joint_position_target(target);robot.set_joint_velocity_target(velocity);robot.write_data_to_sim();sim.step(render=False);robot.update(a.dt);ts.append((k+1)*a.dt);qs.append(arr(robot.data.joint_pos)[0]);vs.append(arr(robot.data.joint_vel)[0])
    np.savez_compressed(folder/'trajectory.npz',t=ts,q=qs,v=vs,joint_names=names)
    np.savez_compressed(folder/'backend.npz',masses=arr(robot.root_physx_view.get_masses()),inertias=arr(robot.root_physx_view.get_inertias()),material_properties=arr(robot.root_physx_view.get_material_properties()),contact_offsets=arr(robot.root_physx_view.get_contact_offsets()),rest_offsets=arr(robot.root_physx_view.get_rest_offsets()),dof_stiffnesses=arr(robot.root_physx_view.get_dof_stiffnesses()),dof_dampings=arr(robot.root_physx_view.get_dof_dampings()))
    s.write(folder/'readback.json',dict(contact_offsets=arr(robot.root_physx_view.get_contact_offsets()).tolist(),rest_offsets=arr(robot.root_physx_view.get_rest_offsets()).tolist(),sphere_contact_attr=sim.stage.GetPrimAtPath('/World/Probe/ball/shape').GetAttribute('physxCollision:contactOffset').Get(),plane_contact_attr=plane.GetPrim().GetAttribute('physxCollision:contactOffset').Get(),scene_friction_attr=sim.stage.GetPrimAtPath('/physicsScene').GetAttribute('physxScene:frictionOffsetThreshold').Get() if sim.stage.GetPrimAtPath('/physicsScene').IsValid() else None))
    xml=f'<mujoco><option timestep="{a.dt}" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic" iterations="200" tolerance="1e-12"/><default><geom friction=".5 0 0" condim="3" solref=".0005 1" solimp=".9999 .9999 .00001 .5 2"/></default><worldbody><geom type="plane" size="1 1 .01"/><body><inertial mass="{carrier}" diaginertia="1e-15 1e-15 1e-15" pos="0 0 0"/><joint type="slide" name="z" axis="0 0 1"/><body><inertial mass="{mass}" diaginertia="{inertia} {inertia} {inertia}" pos="0 0 0"/><joint type="slide" name="x" axis="1 0 0"/><geom type="sphere" size="{radius}"/></body></body></worldbody></mujoco>'
    (folder/'probe.xml').write_text(xml);s.write(folder/'results.json',dict(radius=radius,mass=mass,carrier_mass=carrier,inertia=inertia,initial_clearance=0.,initial_vx=0.,horizontal_kp=100.,horizontal_kd=.01,vertical_kp=0.,vertical_kd=0.,x_target_speed=.03,reference_amplitude_m=.00002,reference_frequency_hz=50.,dt=a.dt,friction_offset=a.friction_offset,contact_offset=a.contact_offset,final_x=float(qs[-1][ix]),min_surface_z=float(np.array(qs)[:,iz].min()-radius),source_sha=s.sha(__file__)));s.write(folder/'complete.json',dict(complete=True));print((folder/'results.json').read_text(),flush=True);sys.stdout.flush();sys.stderr.flush();os._exit(0)
try:main()
except Exception:
    import traceback
    (folder/'error.txt').write_text(traceback.format_exc());traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
