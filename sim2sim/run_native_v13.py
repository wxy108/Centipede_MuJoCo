from pathlib import Path
import argparse,os,sys,time,json
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('WARP_CACHE_PATH',str(ROOT/'cache/warp'))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--bank',default='calibration');p.add_argument('--dt',type=float,default=.00025);p.add_argument('--duration',type=float,default=0);p.add_argument('--pos-iters',type=int,default=16);p.add_argument('--vel-iters',type=int,default=4)
p.add_argument('--fk-check',action='store_true')
p.add_argument('--cpu-threads',type=int,default=8)
AppLauncher.add_app_launcher_args(p);a=p.parse_args()
folder=ROOT/'runs'/a.name;assert not folder.exists();folder.mkdir(parents=True)
(folder/'pid.json').write_text(json.dumps(dict(pid=os.getpid(),argv=sys.argv)),encoding='utf-8')
launcher=AppLauncher(a);app=launcher.app
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation,ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from pxr import UsdGeom,UsdPhysics,UsdShade,PhysxSchema,Gf
import shared as s
from command_cache import CommandCache

def main():
    rows=s.cases('calibration')+s.cases('holdout') if a.bank=='validation' else s.cases(a.bank);N=len(rows);started=time.monotonic()
    cfg=sim_utils.SimulationCfg(dt=a.dt,device=a.device,render_interval=100000,gravity=(0,0,-9.81))
    cfg.worker_thread_count=a.cpu_threads
    cfg.physx.solver_type=1;cfg.physx.enable_stabilization=False
    cfg.physx.friction_offset_threshold=s.C['candidate_numerics']['native_friction_offset_threshold_m']
    cfg.physx.friction_correlation_distance=s.C['candidate_numerics']['native_friction_correlation_distance_m']
    sim=sim_utils.SimulationContext(cfg)
    scene=next(p for p in sim.stage.Traverse() if p.IsA(UsdPhysics.Scene));scene_api=PhysxSchema.PhysxSceneAPI.Apply(scene);scene_api.CreateEnableExternalForcesEveryIterationAttr(True);s.write(folder/'tgs_external_iteration.json',dict(enabled=scene_api.GetEnableExternalForcesEveryIterationAttr().Get(),solver='TGS',scene=str(scene.GetPath())))
    import carb
    worker_threads=carb.settings.get_settings().get('/persistent/physics/numThreads')
    print(json.dumps(dict(worker_threads=worker_threads)),flush=True)
    # Analytic plane, with no external ground asset or second terrain surface.
    plane=UsdGeom.Plane.Define(sim.stage,'/World/Ground');plane.CreateAxisAttr('Z');plane.CreateWidthAttr(20.);plane.CreateLengthAttr(20.)
    UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
    ca=PhysxSchema.PhysxCollisionAPI.Apply(plane.GetPrim());ca.CreateContactOffsetAttr(.00002);ca.CreateRestOffsetAttr(0.)
    material=UsdShade.Material.Define(sim.stage,'/World/GroundMaterial');ma=UsdPhysics.MaterialAPI.Apply(material.GetPrim());ma.CreateStaticFrictionAttr(.5);ma.CreateDynamicFrictionAttr(.5);ma.CreateRestitutionAttr(0.)
    UsdShade.MaterialBindingAPI.Apply(plane.GetPrim()).Bind(material,UsdShade.Tokens.weakerThanDescendants,'physics')
    for e in range(N):
        env=UsdGeom.Xform.Define(sim.stage,f'/World/env_{e}');env.AddTranslateOp().Set(Gf.Vec3d(0.,e*.25,0.))
        group=UsdPhysics.CollisionGroup.Define(sim.stage,f'/World/group_{e}');group.GetCollidersCollectionAPI().CreateIncludesRel().SetTargets([f'/World/env_{e}'])
        group.CreateFilteredGroupsRel().SetTargets([f'/World/group_{j}' for j in range(N) if j!=e])
    kp={x['name']:x['kp'] for x in s.M['joints']};kv={x['name']:x['kv'] for x in s.M['joints']}
    robot_cfg=ArticulationCfg(prim_path='/World/env_.*/Robot',
        spawn=sim_utils.UsdFileCfg(usd_path=str(ROOT/'assets/common.usda'),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False,solver_position_iteration_count=a.pos_iters,solver_velocity_iteration_count=a.vel_iters,sleep_threshold=0.,stabilization_threshold=0.)),
        init_state=ArticulationCfg.InitialStateCfg(pos=tuple(s.M['initial_root_pose_wxyz'][:3])),
        actuators={'all':ImplicitActuatorCfg(joint_names_expr=['.*'],stiffness=kp,damping=kv,effort_limit_sim=1.,velocity_limit_sim=1e6,armature=0.,friction=0.)})
    robot=Articulation(robot_cfg);sim.reset()
    names=list(robot.joint_names);bn=list(robot.body_names);assert set(names)==set(kp) and len(bn)==251
    initial=torch.tensor(s.M['initial_root_pose_wxyz'],device=sim.device,dtype=torch.float32).repeat(N,1);initial[:,1]+=torch.arange(N,device=sim.device)*.25
    robot.write_root_pose_to_sim(initial);robot.write_root_velocity_to_sim(torch.zeros((N,6),device=sim.device));robot.write_joint_state_to_sim(torch.zeros((N,len(names)),device=sim.device),torch.zeros((N,len(names)),device=sim.device));robot.reset();sim.forward();robot.update(0.)
    view=robot.root_physx_view
    def arr(x):return x.detach().cpu().numpy().copy() if hasattr(x,'detach') else np.array(x)
    arrays={key:arr(getattr(view,'get_'+key)()) for key in ['masses','coms','inertias','dof_stiffnesses','dof_dampings','dof_armatures','dof_limits','dof_max_forces','dof_max_velocities','dof_friction_coefficients','material_properties','contact_offsets','rest_offsets']}
    np.savez_compressed(folder/'backend.npz',**arrays)
    s.write(folder/'readback.json',dict(engine='isaaclab_physx',device=sim.device,joint_names=names,body_names=bn,shapes=int(view.max_shapes),physics_dt_s=sim.get_physics_dt(),solver_type=cfg.physx.solver_type,worker_threads=worker_threads,pos_iters=a.pos_iters,vel_iters=a.vel_iters,friction_offset_threshold=cfg.physx.friction_offset_threshold,friction_correlation_distance=cfg.physx.friction_correlation_distance,initial_root_pose=arr(initial).tolist(),pid=os.getpid(),bootstrap_may_advance_physics=True,
        source_hashes={str(x):s.sha(x) for x in [Path(__file__),ROOT/'shared.py',ROOT/'command_cache.py',ROOT/'assets/common.usda',ROOT/'COMMON_CONTRACT.json']}))
    if a.fk_check:
        fk=[];fq=[]
        for scale in [.05,.2]:
            testq=np.array([[scale*np.sin(sorted(names).index(n)+.4) for n in names]]*N,np.float32)
            robot.write_joint_state_to_sim(torch.as_tensor(testq,device=sim.device),torch.zeros((N,len(names)),device=sim.device));sim.forward()
            fq.append(testq);fk.append(arr(view.get_link_transforms()))
        np.savez_compressed(folder/'fk.npz',q=fq,transforms=fk,joint_names=names,body_names=bn)
        robot.write_root_pose_to_sim(initial);robot.write_root_velocity_to_sim(torch.zeros((N,6),device=sim.device));robot.write_joint_state_to_sim(torch.zeros((N,len(names)),device=sim.device),torch.zeros((N,len(names)),device=sim.device));robot.reset();sim.forward();robot.update(0.)
    ts=[0.];qs=[arr(robot.data.joint_pos)];vs=[arr(robot.data.joint_vel)];bs=[arr(robot.data.body_pos_w)]
    stride=round(.01/a.dt);steps=round(a.duration/a.dt)
    cache=CommandCache(names,a.dt,rows);clock_start=sim.current_time
    max_effort=torch.zeros(N,device=sim.device);stopped=False
    for k in range(steps):
        q,v=cache.get(k)
        robot.set_joint_position_target(torch.as_tensor(q,device=sim.device));robot.set_joint_velocity_target(torch.as_tensor(v,device=sim.device));robot.write_data_to_sim()
        max_effort=torch.maximum(max_effort,robot.actuators['all'].computed_effort.abs().amax(dim=1))
        sim.step(render=False);robot.update(a.dt)
        if (k+1)%stride==0:
            ts.append((k+1)*a.dt);qs.append(arr(robot.data.joint_pos));vs.append(arr(robot.data.joint_vel));bs.append(arr(robot.data.body_pos_w))
        if (k+1)%round(.5/a.dt)==0:
            np.savez_compressed(folder/'partial_trajectory.npz',t=ts,q=qs,v=vs,body=bs,joint_names=names,body_names=bn)
            print(json.dumps(dict(sim_time=(k+1)*a.dt,wall_s=time.monotonic()-started)),flush=True)
            if (ROOT/'STOP_SIM2SIM').exists():stopped=True;break
    t=np.array(ts);q=np.array(qs);v=np.array(vs);body=np.array(bs)
    np.savez_compressed(folder/'trajectory.npz',t=t,q=q,v=v,body=body,joint_names=names,body_names=bn)
    result=dict(engine='isaaclab_physx',device=sim.device,arguments=vars(a),wall_s=time.monotonic()-started,solver='TGS',peak_pre_step_estimated_pd_effort_Nm=arr(max_effort).tolist(),stopped=stopped,actual_physics_elapsed_s=sim.current_time-clock_start,rows=s.score(t,body,q,rows,names,bn))
    s.write(folder/'results.json',result);print(json.dumps(result['rows']),flush=True)
    (folder/'complete.json').write_text(json.dumps(dict(complete=not stopped,physics_steps=k+1 if steps else 0)),encoding='utf-8')
    # Headless app.close hung after complete.json in the first readback run.
    # This owned one-shot worker has fully written evidence before process exit.
    sys.stdout.flush();sys.stderr.flush();os._exit(0)
try:main()
except Exception:
    import traceback
    (folder/'error.txt').write_text(traceback.format_exc(),encoding='utf-8');traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
