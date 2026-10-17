from pathlib import Path
import ast
R=Path(__file__).resolve().parent
src=(R/'run_normal_drive_contact_probe.py').read_text(encoding='utf-8')
src=src.replace('cfg.physx.solver_type=0','cfg.physx.solver_type=1')
src=src.replace('sim=su.SimulationContext(cfg)',"sim=su.SimulationContext(cfg);scene=next(p for p in sim.stage.Traverse() if p.IsA(UsdPhysics.Scene));PhysxSchema.PhysxSceneAPI.Apply(scene).CreateEnableExternalForcesEveryIterationAttr(True)")
src=src.replace("stiffness={'x':2.,'z':2.},damping={'x':.01,'z':.01}","stiffness={'x':100.,'z':0.},damping={'x':.01,'z':0.}")
src=src.replace('target[0,ix]=.00002*np.sin(2*np.pi*50*at);target[0,iz]=radius-.00002+.00004*np.sin(2*np.pi*50*at)', 'target[0,ix]=.03*at;target[0,iz]=0.')
src=src.replace('velocity[0,ix]=.00002*2*np.pi*50*np.cos(2*np.pi*50*at);velocity[0,iz]=.00004*2*np.pi*50*np.cos(2*np.pi*50*at)','velocity[0,ix]=.03;velocity[0,iz]=0.')
src=src.replace('horizontal_kp=2.,horizontal_kd=.01,vertical_kp=2.,vertical_kd=.01','horizontal_kp=100.,horizontal_kd=.01,vertical_kp=0.,vertical_kd=0.,x_target_speed=.03')
p=R/'run_continuous_slide_probe.py';assert not p.exists();ast.parse(src);p.write_text(src,encoding='utf-8')
