"""Test the documented TGS external-force iteration setting independently."""
from pathlib import Path
root=Path(__file__).resolve().parent
for base,out in [('run_contact_probe.py','run_tgs_drop_force_iteration.py'),('run_friction_cone_probe.py','run_tgs_friction_force_iteration.py')]:
    src=(root/base).read_text(encoding='utf-8')
    old='AppLauncher.add_app_launcher_args(p);';assert src.count(old)==1
    src=src.replace(old,"p.add_argument('--external-iter',type=int,choices=[0,1],default=0)\n"+old)
    src=src.replace('cfg.physx.solver_type=0','cfg.physx.solver_type=1')
    old='    sim=su.SimulationContext(cfg)';assert src.count(old)==1
    src=src.replace(old,old+"\n    scene=next(p for p in sim.stage.Traverse() if p.IsA(UsdPhysics.Scene));api=PhysxSchema.PhysxSceneAPI.Apply(scene);api.CreateEnableExternalForcesEveryIterationAttr(bool(a.external_iter));s.write(folder/'external_iteration_readback.json',dict(enabled=api.GetEnableExternalForcesEveryIterationAttr().Get(),solver='TGS'))")
    (root/out).write_text(src,encoding='utf-8')
print('Prepared TGS external-force iteration probes.')
