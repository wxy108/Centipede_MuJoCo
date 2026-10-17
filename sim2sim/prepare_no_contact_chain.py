"""Isolate the complete articulated PD dynamics from gravity and contact."""
from pathlib import Path
root=Path(__file__).resolve().parent
for base,out,replacements in [
    ('run_native_v7.py','run_native_no_contact.py',{
        'gravity=(0,0,-9.81)':'gravity=(0.,0.,0.)',
        'robot=Articulation(robot_cfg);sim.reset()':"robot=Articulation(robot_cfg);UsdPhysics.CollisionAPI(plane.GetPrim()).CreateCollisionEnabledAttr(False);sim.reset()",
    }),
    ('run_mujoco_v8.py','run_mujoco_no_contact.py',{
        'm.opt.timestep=a.dt':'m.opt.timestep=a.dt;m.opt.gravity[:]=0.;m.geom_contype[:]=0;m.geom_conaffinity[:]=0',
    }),
]:
    src=(root/base).read_text(encoding='utf-8')
    for old,new in replacements.items():
        assert src.count(old)==1,(old,src.count(old));src=src.replace(old,new)
    (root/out).write_text(src,encoding='utf-8')
print('Prepared no-contact chain diagnostics.')
