from pathlib import Path
root=Path(__file__).resolve().parent;src=(root/'run_drive_probe.py').read_text(encoding='utf-8')
old='    sim=su.SimulationContext(cfg)';assert src.count(old)==1
src=src.replace(old,old+"\n    scene=next(p for p in sim.stage.Traverse() if p.IsA(UsdPhysics.Scene));api=PhysxSchema.PhysxSceneAPI.Apply(scene);api.CreateEnableExternalForcesEveryIterationAttr(True);s.write(folder/'tgs_external_iteration.json',dict(enabled=api.GetEnableExternalForcesEveryIterationAttr().Get(),solver='TGS'))")
(root/'run_drive_probe_tgs_ext.py').write_text(src,encoding='utf-8')
print('Prepared TGS drive probe with current scene setting.')
