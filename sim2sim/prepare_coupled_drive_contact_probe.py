import shared as s
source=(s.ROOT/'run_contact_probe_pgs.py').read_text()
changes=[
    ("stiffness=0.,damping=0.,armature=0.","stiffness={'x':2.,'z':0.},damping={'x':.01,'z':0.},armature=0."),
    ("q[0,iz]=radius+clearance;v[0,ix]=.02","q[0,iz]=radius;v[0,ix]=0."),
    ("robot.write_data_to_sim();sim.step(render=False)","at=k*a.dt;target=q.clone();target.zero_();target[0,ix]=.00002*np.sin(2*np.pi*50*at);velocity=target.clone();velocity.zero_();velocity[0,ix]=.00002*2*np.pi*50*np.cos(2*np.pi*50*at);robot.set_joint_position_target(target);robot.set_joint_velocity_target(velocity);robot.write_data_to_sim();sim.step(render=False)"),
    ("initial_clearance=clearance,initial_vx=.02","initial_clearance=0.,initial_vx=0.,horizontal_kp=2.,horizontal_kd=.01,reference_amplitude_m=.00002,reference_frequency_hz=50."),
    ("rest_offsets=arr(robot.root_physx_view.get_rest_offsets()))","rest_offsets=arr(robot.root_physx_view.get_rest_offsets()),dof_stiffnesses=arr(robot.root_physx_view.get_dof_stiffnesses()),dof_dampings=arr(robot.root_physx_view.get_dof_dampings()))")]
for old,new in changes:
    assert old in source,old;source=source.replace(old,new)
(s.ROOT/'run_coupled_drive_contact_probe.py').write_text(source,encoding='utf-8')
print('Prepared coupled drive/contact probe; no physics advanced.')
