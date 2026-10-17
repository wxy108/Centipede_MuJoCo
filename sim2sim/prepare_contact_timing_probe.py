from pathlib import Path
R=Path(__file__).resolve().parent
src=(R/'run_native_common_contact_v2.py').read_text()
src=src.replace('for k in range(steps):','timings=np.zeros(5)\n    for k in range(steps):\n        tick=time.perf_counter()')
src=src.replace('F,T,info=contact.forces',"tpose=time.perf_counter();timings[0]+=tpose-tick\n        F,T,info=contact.forces")
src=src.replace('robot.set_external_force_and_torque(torch.as_tensor', 'tcontact=time.perf_counter();timings[1]+=tcontact-tpose\n        robot.set_external_force_and_torque(torch.as_tensor')
src=src.replace('sim.step(render=False);robot.update(a.dt)', 'twrite=time.perf_counter();timings[2]+=twrite-tcontact\n        sim.step(render=False);robot.update(a.dt)\n        tstep=time.perf_counter();timings[3]+=tstep-twrite')
src=src.replace("t=np.array(ts);q=np.array(qs)","s.write(folder/'timing.json',dict(steps=steps,pose_s=timings[0],contact_s=timings[1],write_s=timings[2],physics_s=timings[3],torch_threads=torch.get_num_threads()))\n    t=np.array(ts);q=np.array(qs)")
(R/'run_native_contact_timing_probe.py').write_text(src)
