from pathlib import Path
import ast
R=Path(__file__).resolve().parent
src=(R/'run_inverse_mass_probe.py').read_text(encoding='utf-8');start=src.index('    assert N==7 and a.mass_scale==1');end=src.index('    ts=[0.];qs=',start)
block='''    assert N==7 and a.mass_scale==1
    selected=['link_body_0','link_body_10','link_body_20','link_leg_9_L_2','link_leg_9_L_3','link_body_10','link_leg_9_L_3']
    robot.write_joint_stiffness_to_sim(torch.zeros((N,len(names)),device=sim.device));robot.write_joint_damping_to_sim(torch.zeros((N,len(names)),device=sim.device))
    forces=torch.zeros((N,len(bn),3),device=sim.device);torques=forces.clone()
    samples=[[1e-4,0,0],[0,1e-4,0],[0,0,1e-4],[0,0,1e-5],[1e-5,-2e-5,3e-5],[0,0,0],[0,0,0]]
    for e,b in enumerate(selected):forces[e,bn.index(b)]=torch.tensor(samples[e],device=sim.device)
    torques[5,bn.index(selected[5]),0]=1e-8;torques[6,bn.index(selected[6]),1]=1e-9
    before_v=arr(robot.data.joint_vel);before_root_v=arr(robot.data.root_vel_w)
    robot.set_external_force_and_torque(forces,torques,is_global=True);robot.write_data_to_sim();sim.step(render=False);robot.update(a.dt)
    np.savez_compressed(folder/'body_wrench_probe.npz',force=arr(forces),torque=arr(torques),v_before=before_v,root_v_before=before_root_v,v_after=arr(robot.data.joint_vel),root_v_after=arr(robot.data.root_vel_w),stiffness=arr(view.get_dof_stiffnesses()),damping=arr(view.get_dof_dampings()),joint_names=names,body_names=bn)
    s.write(folder/'body_wrench_probe.json',dict(dt=a.dt,selected_bodies=selected,gravity=0,drive_stiffness=0,drive_damping=0,application='production robot.set_external_force_and_torque, world frame, no explicit position',source_sha=s.sha(__file__)))
    s.write(folder/'complete.json',dict(complete=True,diagnostic_only=True,physics_steps=1));print('Body wrench response recorded',flush=True);sys.stdout.flush();sys.stderr.flush();os._exit(0)
'''
src=src[:start]+block+src[end:];ast.parse(src);p=R/'run_body_wrench_probe.py';assert not p.exists();p.write_text(src,encoding='utf-8')
