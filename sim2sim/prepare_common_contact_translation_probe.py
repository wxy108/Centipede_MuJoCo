from pathlib import Path
R=Path(__file__).resolve().parent
src=(R/'run_native_common_contact_origin_probe.py').read_text()
src=src.replace('gravity=(0,0,-9.81)','gravity=(0.,0.,0.)')
src=src.replace('robot.write_root_velocity_to_sim(torch.zeros((N,6),device=sim.device))',"root_velocity=torch.zeros((N,6),device=sim.device);root_velocity[:,1]=.003;robot.write_root_velocity_to_sim(root_velocity)")
src=src.replace("ts=[0.];qs=", "root_positions=[arr(view.get_root_transforms())[:,:3]];root_velocities=[arr(view.get_root_velocities())]\n    ts=[0.];qs=")
src=src.replace("ts.append((k+1)*a.dt);qs.append", "root_positions.append(arr(view.get_root_transforms())[:,:3]);root_velocities.append(arr(view.get_root_velocities()))\n            ts.append((k+1)*a.dt);qs.append")
src=src.replace("s.write(folder/'timing.json'", "np.savez_compressed(folder/'translation_probe.npz',t=ts,position=root_positions,velocity=root_velocities)\n    s.write(folder/'translation_probe_setup.json',dict(gravity=[0,0,0],root_velocity_y_m_s=.003,global_y_m=[0,1.75],analytical_displacement_mm=.003*a.duration*1000,qualification='No gait before 1 s; native joint targets remain zero; common contact forces must be zero.'))\n    s.write(folder/'timing.json'")
(R/'run_native_common_contact_translation_probe.py').write_text(src)
