from pathlib import Path
root=Path(__file__).resolve().parent;src=(root/'run_native_v6.py').read_text(encoding='utf-8')
old='cfg.physx.solver_type=0';assert src.count(old)==1;src=src.replace(old,'cfg.physx.solver_type=1')
old='    sim=sim_utils.SimulationContext(cfg)';assert src.count(old)==1
src=src.replace(old,old+"\n    scene=next(p for p in sim.stage.Traverse() if p.IsA(UsdPhysics.Scene));scene_api=PhysxSchema.PhysxSceneAPI.Apply(scene);scene_api.CreateEnableExternalForcesEveryIterationAttr(True);s.write(folder/'tgs_external_iteration.json',dict(enabled=scene_api.GetEnableExternalForcesEveryIterationAttr().Get(),solver='TGS',scene=str(scene.GetPath())))")
src=src.replace("solver='PGS'","solver='TGS'")
(root/'run_native_v13.py').write_text(src,encoding='utf-8')
print('Prepared TGS with external forces each iteration.')
