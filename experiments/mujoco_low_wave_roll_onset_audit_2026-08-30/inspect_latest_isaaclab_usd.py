"""Read-only inventory of the hash-locked clean IsaacLab physics USD."""

from collections import Counter
from pathlib import Path
import json

from isaaclab.app import AppLauncher

# USD/PhysX schemas are supplied by the Isaac Sim runtime, so initialize it
# before importing pxr.  This remains a read-only, headless stage inspection.
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from pxr import Usd, UsdPhysics


ASSET = Path(
    r"C:\Users\wxy22\Documents\Centipede_IsaacLab_RL_Clean"
) / "assets/centipede_rebuild_v1_renamed.usd"
OUT = Path(__file__).resolve().parent / "outputs/data/latest_isaaclab_usd_inventory.json"


def authored(attribute):
    return bool(attribute and attribute.HasAuthoredValueOpinion())


def value(attribute):
    if not authored(attribute):
        return None
    result = attribute.Get()
    if not isinstance(result, str) and hasattr(result, "__len__") and hasattr(result, "__getitem__"):
        return [float(result[index]) for index in range(len(result))]
    return float(result) if isinstance(result, (int, float)) else str(result)


def body_family(name):
    if name.startswith("foot_"):
        return "foot"
    if name.startswith("link_leg_"):
        for suffix in ("_0", "_1", "_2", "_3"):
            if name.endswith(suffix):
                return f"leg{suffix}"
        return "leg_other"
    if name.startswith("link_pitch_body_") and name.endswith("__split0"):
        return "body_pitch_split"
    if name.startswith("link_pitch_body_"):
        return "body_pitch_roll"
    if name.startswith("link_body_"):
        return "body_main"
    return "other"


def main():
    stage = Usd.Stage.Open(str(ASSET), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Could not open {ASSET}")
    type_counts = Counter()
    rigid_bodies = []
    rigid_body_dynamics = []
    colliders = []
    joints = []
    authored_mass_sum = 0.0
    authored_mass_count = 0

    for prim in stage.Traverse():
        type_name = prim.GetTypeName() or "<typeless>"
        type_counts[type_name] += 1
        path = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_bodies.append(path)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            colliders.append(path)
        mass_api = UsdPhysics.MassAPI(prim)
        mass = value(mass_api.GetMassAttr()) if mass_api else None
        inertia = value(mass_api.GetDiagonalInertiaAttr()) if mass_api else None
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_body_dynamics.append(
                {
                    "path": path,
                    "name": prim.GetName(),
                    "family": body_family(prim.GetName()),
                    "mass_kg": mass,
                    "diagonal_inertia_kg_m2": inertia,
                }
            )
        if mass is not None:
            authored_mass_count += 1
            authored_mass_sum += float(mass)
        if type_name.endswith("Joint"):
            drive = {}
            for attribute in prim.GetAttributes():
                name = attribute.GetName()
                if "drive:" in name or "limit:" in name or "joint" in name.lower():
                    if attribute.HasAuthoredValueOpinion():
                        raw = attribute.Get()
                        if hasattr(raw, "__iter__") and not isinstance(raw, str):
                            raw = [float(item) for item in raw]
                        elif isinstance(raw, (int, float)):
                            raw = float(raw)
                        else:
                            raw = str(raw)
                        drive[name] = raw
            joints.append(
                {
                    "path": path,
                    "name": prim.GetName(),
                    "type": type_name,
                    "body0": [str(target) for target in UsdPhysics.Joint(prim).GetBody0Rel().GetTargets()],
                    "body1": [str(target) for target in UsdPhysics.Joint(prim).GetBody1Rel().GetTargets()],
                    "authored": drive,
                }
            )
    joint_family_counts = Counter()
    for joint in joints:
        name = joint["name"]
        family = next(
            (
                prefix
                for prefix in (
                    "body_yaw", "body_pitch", "body_roll", "legbase_yaw",
                    "legbase_pitch", "legbase_roll", "knee", "foot",
                )
                if name.startswith(prefix)
            ),
            "other",
        )
        joint_family_counts[family] += 1

    family_mass = {}
    for record in rigid_body_dynamics:
        family = record["family"]
        aggregate = family_mass.setdefault(family, {"count": 0, "authored_mass_count": 0, "mass_kg": 0.0})
        aggregate["count"] += 1
        if record["mass_kg"] is not None:
            aggregate["authored_mass_count"] += 1
            aggregate["mass_kg"] += record["mass_kg"]

    output = {
        "asset": str(ASSET),
        "default_prim": str(stage.GetDefaultPrim().GetPath()),
        "meters_per_unit": float(stage.GetMetadata("metersPerUnit") or 1.0),
        "up_axis": str(stage.GetMetadata("upAxis")),
        "type_counts": dict(sorted(type_counts.items())),
        "rigid_body_count": len(rigid_bodies),
        "collider_count": len(colliders),
        "joint_count": len(joints),
        "joint_family_counts": dict(sorted(joint_family_counts.items())),
        "authored_mass_count": authored_mass_count,
        "authored_mass_sum_kg": authored_mass_sum,
        "family_mass": dict(sorted(family_mass.items())),
        "rigid_body_paths": rigid_bodies,
        "rigid_body_dynamics": rigid_body_dynamics,
        "collider_paths": colliders,
        "joints": joints,
        "model_or_controller_modified": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({key: output[key] for key in (
        "asset", "default_prim", "meters_per_unit", "up_axis", "rigid_body_count",
        "collider_count", "joint_count", "joint_family_counts",
        "authored_mass_count", "authored_mass_sum_kg",
    )}, indent=2))
    print(f"[inventory] {OUT}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
