from pathlib import Path
R=Path(__file__).resolve().parent
src=(R/'run_native_contact_timing_probe.py').read_text()
src=src.replace('from common_contact_v2 import CommonContact,quat_matrix','from common_contact_fast import CommonContact,quat_matrix')
src=src.replace("ROOT/'common_contact_v2.py'", "ROOT/'common_contact_v2.py',ROOT/'common_contact_fast.py'")
src=src.replace('robot=Articulation(robot_cfg);sim.reset()', '''robot=Articulation(robot_cfg)
    # Ground, self and inter-environment collision were already disabled/filtered.
    # No native collision pair can contribute to this external-contact benchmark.
    disabled=[]
    for prim in sim.stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False);disabled.append(str(prim.GetPath()))
    s.write(folder/'native_collision_disabled.json',dict(count=len(disabled),paths=disabled))
    sim.reset()''')
(R/'run_native_common_contact_fast.py').write_text(src)
