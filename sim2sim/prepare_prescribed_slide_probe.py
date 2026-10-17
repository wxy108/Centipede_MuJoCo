"""Prescribed lateral velocity boundary, free normal motion: check contact dilatancy."""
from pathlib import Path
R=Path(__file__).resolve().parent
src=(R/'run_oblique_slide_probe.py').read_text(encoding='utf-8')
src=src.replace("p.add_argument('--solver',type=int,default=0)","p.add_argument('--solver',type=int,default=1)")
src=src.replace('sim=su.SimulationContext(cfg)',"sim=su.SimulationContext(cfg);scene=next(p for p in sim.stage.Traverse() if p.IsA(UsdPhysics.Scene));PhysxSchema.PhysxSceneAPI.Apply(scene).CreateEnableExternalForcesEveryIterationAttr(True)")
src=src.replace("v[0,ix]=.05*np.cos(np.pi/6);v[0,iy]=.05*np.sin(np.pi/6)","v[0,ix]=.03;v[0,iy]=0.")
src=src.replace("for k in range(round(.025/a.dt)):\n        robot.write_data_to_sim();", "for k in range(round(.025/a.dt)):\n        q=robot.data.joint_pos.clone();v=robot.data.joint_vel.clone();q[0,ix]=k*a.dt*.03;q[0,iy]=0.;v[0,ix]=.03;v[0,iy]=0.;robot.write_joint_state_to_sim(q,v);robot.update(0.);robot.write_data_to_sim();")
src=src.replace('initial_speed=.05,angle_deg=30','initial_speed=.03,angle_deg=0,prescribed_xy=True,external_forces_each_iteration=True')
p=R/'run_prescribed_slide_probe.py';assert not p.exists();p.write_text(src,encoding='utf-8')
print('Prepared fixed-speed contact boundary probe; not locomotion.')
