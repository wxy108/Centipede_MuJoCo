from pathlib import Path
root=Path(__file__).resolve().parent;src=(root/'run_native_v7.py').read_text(encoding='utf-8')
replacements={
"AppLauncher.add_app_launcher_args(p);":"p.add_argument('--contact-offset',type=float,default=.00002);p.add_argument('--friction-offset',type=float,default=.00004);p.add_argument('--correlation',type=float,default=.00005)\nAppLauncher.add_app_launcher_args(p);",
"cfg.physx.friction_offset_threshold=s.C['candidate_numerics']['native_friction_offset_threshold_m']":"cfg.physx.friction_offset_threshold=a.friction_offset",
"cfg.physx.friction_correlation_distance=s.C['candidate_numerics']['native_friction_correlation_distance_m']":"cfg.physx.friction_correlation_distance=a.correlation",
"robot=Articulation(robot_cfg);sim.reset()":"robot=Articulation(robot_cfg)\n    offset_values={}\n    for prim in sim.stage.Traverse():\n        if prim.HasAPI(UsdPhysics.CollisionAPI):\n            pc=PhysxSchema.PhysxCollisionAPI.Apply(prim);pc.CreateContactOffsetAttr(a.contact_offset);offset_values[str(prim.GetPath())]=pc.GetContactOffsetAttr().Get()\n    assert len(offset_values)==N*249+1\n    s.write(folder/'contact_scale_readback.json',dict(values=offset_values,contact_offset=a.contact_offset,friction_offset=a.friction_offset,correlation=a.correlation,stage='authored before native bootstrap; backend offsets are separately saved'))\n    sim.reset()",
}
for old,new in replacements.items():
    assert src.count(old)==1,(old,src.count(old));src=src.replace(old,new)
(root/'run_native_v12.py').write_text(src,encoding='utf-8')
print('Prepared contact-scale diagnostics.')
