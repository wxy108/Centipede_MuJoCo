"""Read-only audit of the clean Robot USD used only as a dynamics reference."""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ROBOT_ROOT = Path(r"C:\Users\wxy22\Documents\Centipede_IsaacLab_RL_Clean")
USD_PATH = ROBOT_ROOT / "assets" / "centipede_rebuild_v1_renamed.usd"
OUT = ROOT / "outputs" / "robot_reference_readonly_audit.json"
USD_LIBS = Path(
    r"C:\Users\wxy22\anaconda3\envs\isaaclab_50\Lib\site-packages\isaacsim\extscache"
    r"\omni.usd.libs-1.0.1+8131b85d.wx64.r.cp311"
)

os.add_dll_directory(str(USD_LIBS / "bin"))
sys.path.insert(0, str(USD_LIBS))
from pxr import Usd, UsdPhysics  # noqa: E402


KEYWORDS = ("mass", "inertia", "armature", "damping", "stiffness", "gravity")


def serializable(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    try:
        return [serializable(item) for item in value]
    except TypeError:
        return str(value)


def main() -> None:
    stage = Usd.Stage.Open(str(USD_PATH))
    if stage is None:
        raise RuntimeError(f"Could not open {USD_PATH}")

    prim_types = Counter()
    authored_dynamics = []
    mass_api = []
    joints = []
    for prim in stage.Traverse():
        prim_types[prim.GetTypeName() or "<untyped>"] += 1
        attrs = {}
        for attr in prim.GetAttributes():
            name = attr.GetName()
            if any(keyword in name.lower() for keyword in KEYWORDS) and attr.HasAuthoredValueOpinion():
                attrs[name] = serializable(attr.Get())
        if attrs:
            authored_dynamics.append({"prim": str(prim.GetPath()), "type": prim.GetTypeName(), "attributes": attrs})

        if prim.HasAPI(UsdPhysics.MassAPI):
            api = UsdPhysics.MassAPI(prim)
            mass_api.append(
                {
                    "prim": str(prim.GetPath()),
                    "mass": serializable(api.GetMassAttr().Get()),
                    "density": serializable(api.GetDensityAttr().Get()),
                    "center_of_mass": serializable(api.GetCenterOfMassAttr().Get()),
                    "diagonal_inertia": serializable(api.GetDiagonalInertiaAttr().Get()),
                    "principal_axes": serializable(api.GetPrincipalAxesAttr().Get()),
                }
            )

        if "Joint" in (prim.GetTypeName() or ""):
            joints.append({"prim": str(prim.GetPath()), "type": prim.GetTypeName(), "attributes": attrs})

    named_joint_paths = [row["prim"] for row in joints]
    payload = {
        "scope": "Robot repository read-only reference; no Robot files written",
        "usd": str(USD_PATH),
        "prim_count": int(sum(prim_types.values())),
        "prim_types": dict(sorted(prim_types.items())),
        "mass_api_prim_count": len(mass_api),
        "joint_prim_count": len(joints),
        "contains_legacy_joint_roll_leg_name": any("joint_roll_leg" in path for path in named_joint_paths),
        "contains_legbase_pitch_name": any("legbase_pitch" in path for path in named_joint_paths),
        "mass_api": mass_api,
        "joints": joints,
        "authored_dynamics": authored_dynamics,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "prim_count": payload["prim_count"],
                "mass_api_prim_count": payload["mass_api_prim_count"],
                "joint_prim_count": payload["joint_prim_count"],
                "contains_legacy_joint_roll_leg_name": payload[
                    "contains_legacy_joint_roll_leg_name"
                ],
                "contains_legbase_pitch_name": payload["contains_legbase_pitch_name"],
                "output": str(OUT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
