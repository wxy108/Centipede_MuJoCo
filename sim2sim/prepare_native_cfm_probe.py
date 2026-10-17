from pathlib import Path
root=Path(__file__).resolve().parent;src=(root/'run_native_v7.py').read_text(encoding='utf-8')
replacements={
"AppLauncher.add_app_launcher_args(p);":"p.add_argument('--cfm-scale',type=float,default=0.)\nAppLauncher.add_app_launcher_args(p);",
"robot=Articulation(robot_cfg);sim.reset()":"robot=Articulation(robot_cfg)\n    cfm_values={}\n    for prim in sim.stage.Traverse():\n        if prim.HasAPI(UsdPhysics.RigidBodyAPI):\n            rb_api=PhysxSchema.PhysxRigidBodyAPI.Apply(prim);rb_api.CreateCfmScaleAttr(a.cfm_scale);cfm_values[str(prim.GetPath())]=rb_api.GetCfmScaleAttr().Get()\n    assert len(cfm_values)==N*251 and all(abs(v-a.cfm_scale)<1e-8 for v in cfm_values.values())\n    s.write(folder/'cfm_readback.json',dict(values=cfm_values,requested=a.cfm_scale,stage='authored before native scene bootstrap'))\n    sim.reset()",
}
for old,new in replacements.items():
    assert src.count(old)==1,(old,src.count(old));src=src.replace(old,new)
(root/'run_native_v11.py').write_text(src,encoding='utf-8')
print('Prepared explicit-CFM native diagnostic.')
