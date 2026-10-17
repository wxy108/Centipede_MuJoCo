from pathlib import Path
import ast
R=Path(__file__).resolve().parent
mu=(R/'run_mujoco_v8.py').read_text(encoding='utf-8').replace('from command_cache import CommandCache','from command_cache import CommandCache\nfrom common_contact_v2 import CommonContact')
mu=mu.replace('m.geom_solref[:,0]=2*a.dt','m.geom_solref[:,0]=2*a.dt;m.opt.disableflags|=mujoco.mjtDisableBit.mjDSBL_CONTACT')
mu=mu.replace('peak_force=0.;stopped=False','peak_force=0.;stopped=False;max_depth=0.;max_contact_force=0.;contact=CommonContact(body_names,1,a.dt)')
mu=mu.replace('        mujoco.mj_step(m,d)', '''        mujoco.mj_step1(m,d)
        com_v=d.cvel[:,3:]+np.cross(d.cvel[:,:3],d.xipos-d.subtree_com[m.body_rootid])
        velocity=np.concatenate([com_v[1:],d.cvel[1:,:3]],axis=1)[None]
        F,T,info=contact.forces(d.xpos[1:][None],d.xmat[1:].reshape(1,-1,3,3),d.xipos[1:][None],velocity)
        d.xfrc_applied[1:,:3]=F[0];d.xfrc_applied[1:,3:]=T[0];max_depth=max(max_depth,info['max_penetration_m']);max_contact_force=max(max_contact_force,info['max_point_force_N'])
        mujoco.mj_step2(m,d)''')
mu=mu.replace('stopped=stopped))','stopped=stopped,max_contact_depth_m=max_depth,max_contact_force_N=max_contact_force))')
mu=mu.replace("s.ROOT/'COMMON_CONTRACT.json']","s.ROOT/'COMMON_CONTRACT.json',s.ROOT/'COMMON_CONTACT_V2.json',s.ROOT/'common_contact_v2.py',s.ROOT/'assets/common_contact_geometry.json']")
mu=mu.replace("stopped=[x['stopped'] for x in traces]))", "stopped=[x['stopped'] for x in traces],contact_model='COMMON_CONTACT_V2.json; shared compliant point contact',native_contact_disabled=True,max_contact_depth_m=[x['max_contact_depth_m'] for x in traces],max_contact_point_force_N=[x['max_contact_force_N'] for x in traces]))")
native=(R/'run_native_v14.py').read_text(encoding='utf-8').replace('from command_cache import CommandCache','from command_cache import CommandCache\nfrom common_contact_v2 import CommonContact,quat_matrix')
native=native.replace('cfg.physx.solver_type=1','cfg.physx.solver_type=0').replace("solver='TGS'","solver='PGS'")
native=native.replace('UsdPhysics.CollisionAPI.Apply(plane.GetPrim())','UsdPhysics.CollisionAPI.Apply(plane.GetPrim()).CreateCollisionEnabledAttr(False)')
native=native.replace('max_effort=torch.zeros(N,device=sim.device);stopped=False', "max_effort=torch.zeros(N,device=sim.device);stopped=False;contact=CommonContact(bn,N,a.dt);max_depth=0.;max_contact_force=0.;s.write(folder/'common_contact_setup.json',dict(contract='COMMON_CONTACT_V2.json',native_ground_collision=UsdPhysics.CollisionAPI(plane.GetPrim()).GetCollisionEnabledAttr().Get(),body_force_frame='world, torque about COM'))")
native=native.replace('        q,v=cache.get(k//command_stride)', '''        pose=arr(robot.data.body_link_pose_w).astype(float);com=arr(robot.data.body_com_pose_w)[...,:3].astype(float);velocity=arr(robot.data.body_com_vel_w).astype(float)
        F,T,info=contact.forces(pose[...,:3],quat_matrix(pose[...,3:]),com,velocity);max_depth=max(max_depth,info['max_penetration_m']);max_contact_force=max(max_contact_force,info['max_point_force_N'])
        robot.set_external_force_and_torque(torch.as_tensor(F,dtype=torch.float32,device=sim.device),torch.as_tensor(T,dtype=torch.float32,device=sim.device),is_global=True)
        q,v=cache.get(k//command_stride)''')
native=native.replace("ROOT/'COMMON_CONTRACT.json']", "ROOT/'COMMON_CONTRACT.json',ROOT/'COMMON_CONTACT_V2.json',ROOT/'common_contact_v2.py',ROOT/'assets/common_contact_geometry.json']")
native=native.replace("actual_physics_elapsed_s=sim.current_time-clock_start,rows=", "actual_physics_elapsed_s=sim.current_time-clock_start,contact_model='COMMON_CONTACT_V2.json; shared compliant point contact',max_contact_depth_m=max_depth,max_contact_point_force_N=max_contact_force,rows=")
for name,src in [('run_mujoco_common_contact.py',mu),('run_native_common_contact.py',native)]:
    ast.parse(src);p=R/name;assert not p.exists();p.write_text(src,encoding='utf-8')
print('Prepared independent shared-contact runners; old scripts/assets unchanged.')
