import shared as s
src=(s.ROOT/'run_native_v7.py').read_text()
src=src.replace("gravity=(0,0,-9.81)","gravity=(0,0,0)")
old="ts=[0.];qs=[arr(robot.data.joint_pos)];vs=[arr(robot.data.joint_vel)];bs=[arr(robot.data.body_pos_w)]"
new="""assert N==7 and a.mass_scale==1
    selected=['body_yaw_9','body_pitch_9','body_roll_9','body_pitch_0','legbase_yaw_9_L','legbase_pitch_9_L','tarsus_yaw_9_L']
    robot.write_joint_stiffness_to_sim(torch.zeros((N,len(names)),device=sim.device));robot.write_joint_damping_to_sim(torch.zeros((N,len(names)),device=sim.device))
    force=torch.zeros((N,len(names)),device=sim.device)
    for e,n in enumerate(selected):force[e,names.index(n)]=1e-7
    before_q=arr(robot.data.joint_pos);before_v=arr(robot.data.joint_vel);before_root_v=arr(robot.data.root_vel_w)
    view.set_dof_actuation_forces(force,robot._ALL_INDICES);sim.step(render=False);robot.update(a.dt)
    np.savez_compressed(folder/'inverse_probe.npz',force=arr(force),q_before=before_q,v_before=before_v,root_v_before=before_root_v,q_after=arr(robot.data.joint_pos),v_after=arr(robot.data.joint_vel),root_v_after=arr(robot.data.root_vel_w),stiffness=arr(view.get_dof_stiffnesses()),damping=arr(view.get_dof_dampings()),joint_names=names,body_names=bn)
    s.write(folder/'inverse_probe.json',dict(dt=a.dt,torque_Nm=1e-7,selected_joints=selected,gravity=0,drive_stiffness=0,drive_damping=0,contacts='none: initially 100 µm clearance, zero gravity, one microsecond step',generalized_mass_getter_before_step=False,source_sha=s.sha(__file__)))
    s.write(folder/'complete.json',dict(complete=True,diagnostic_only=True,physics_steps=1));print('Inverse inertia response recorded',flush=True);sys.stdout.flush();sys.stderr.flush();os._exit(0)
    ts=[0.];qs=[arr(robot.data.joint_pos)];vs=[arr(robot.data.joint_vel)];bs=[arr(robot.data.body_pos_w)]"""
assert old in src;src=src.replace(old,new);(s.ROOT/'run_inverse_mass_probe.py').write_text(src,encoding='utf-8')
print('Prepared force-to-velocity inverse inertia diagnostic, no mass-matrix getter before step.')
