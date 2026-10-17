"""Build the evidence-supported FARMS MuJoCo reliability model.

The source MJCF is never edited in place.  This builder replaces the global
``1e-4`` armature with the smallest tested stable order, ``1e-5``, then
removes passive stiffness/damping only from the 211 joints that are actively
impedance controlled.  This is explicitly a MuJoCo collision-chain numerical
regularizer, not a claimed physical rotor inertia.  The 38
passive leg-roll hinges and 38 passive foot hinges retain their physical
spring/damper terms because they are not members of the active Isaac-Lab
joint families.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import mujoco
import numpy as np


DEFAULT_SOURCE = Path(
    r"C:\Users\wxy22\Documents\Centipede_MUJOCO-main\models\farms\centipede.xml"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "models" / "farms" / "centipede_reliability_refined.xml"
NUMERICAL_ARMATURE = 1.0e-5


ACTIVE_PATTERNS = (
    re.compile(r"joint_body_\d+$"),
    re.compile(r"joint_pitch_body_\d+$"),
    re.compile(r"joint_roll_body_\d+$"),
    re.compile(r"joint_leg_\d+_[LR]_[0-3]$"),
)


def is_active(name: str) -> bool:
    return any(pattern.fullmatch(name) for pattern in ACTIVE_PATTERNS)


def set_attribute(tag: str, attribute: str, value: str) -> str:
    pattern = re.compile(rf'\b{re.escape(attribute)}="[^"]*"')
    replacement = f'{attribute}="{value}"'
    if pattern.search(tag):
        return pattern.sub(replacement, tag)
    return tag[:-2] + f' {replacement}/>' if tag.endswith("/>") else tag


def rewrite_joint(match: re.Match[str]) -> str:
    tag = match.group(0)
    name_match = re.search(r'\bname="([^"]+)"', tag)
    if not name_match or not is_active(name_match.group(1)):
        return tag
    tag = set_attribute(tag, "armature", f"{NUMERICAL_ARMATURE:.1e}")
    tag = set_attribute(tag, "damping", "0")
    tag = set_attribute(tag, "stiffness", "0")
    return tag


def joint_scalar(model: mujoco.MjModel, joint_id: int, values: np.ndarray) -> np.ndarray:
    start = int(model.jnt_dofadr[joint_id])
    count = 6 if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE) else 1
    return np.asarray(values[start : start + count], dtype=float)


def validate(source: Path, output: Path) -> None:
    old = mujoco.MjModel.from_xml_path(str(source))
    new = mujoco.MjModel.from_xml_path(str(output))
    if (old.nbody, old.njnt, old.ngeom, old.nu) != (new.nbody, new.njnt, new.ngeom, new.nu):
        raise RuntimeError("Topology changed while refining the model")
    if not np.isclose(old.body_mass.sum(), new.body_mass.sum(), rtol=0.0, atol=1e-15):
        raise RuntimeError("Total physical mass changed while refining the model")

    active_ids: list[int] = []
    passive_ids: list[int] = []
    for joint_id in range(new.njnt):
        name = mujoco.mj_id2name(new, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
        if is_active(name):
            active_ids.append(joint_id)
        elif name.startswith("joint_roll_leg_") or name.startswith("joint_foot_"):
            passive_ids.append(joint_id)
    if len(active_ids) != 211:
        raise RuntimeError(f"Expected 211 active joints, found {len(active_ids)}")
    if len(passive_ids) != 76:
        raise RuntimeError(f"Expected 76 preserved passive hinges, found {len(passive_ids)}")

    for joint_id in active_ids:
        if np.any(joint_scalar(new, joint_id, new.dof_armature) != NUMERICAL_ARMATURE):
            raise RuntimeError(f"Wrong active numerical armature at joint id {joint_id}")
        if np.any(joint_scalar(new, joint_id, new.dof_damping) != 0.0):
            raise RuntimeError(f"Nonzero active damping at joint id {joint_id}")
        if float(new.jnt_stiffness[joint_id]) != 0.0:
            raise RuntimeError(f"Nonzero active stiffness at joint id {joint_id}")

    for new_joint_id in passive_ids:
        name = mujoco.mj_id2name(new, mujoco.mjtObj.mjOBJ_JOINT, new_joint_id)
        old_joint_id = mujoco.mj_name2id(old, mujoco.mjtObj.mjOBJ_JOINT, name)
        if np.any(joint_scalar(new, new_joint_id, new.dof_armature) != NUMERICAL_ARMATURE):
            raise RuntimeError(f"Wrong passive numerical armature for {name}")
        old_values = joint_scalar(old, old_joint_id, old.dof_damping)
        new_values = joint_scalar(new, new_joint_id, new.dof_damping)
        if not np.array_equal(old_values, new_values):
            raise RuntimeError(f"Passive damping changed for {name}")
        if float(old.jnt_stiffness[old_joint_id]) != float(new.jnt_stiffness[new_joint_id]):
            raise RuntimeError(f"Passive stiffness changed for {name}")

    print(f"PASS: {output}")
    print(f"  topology: nbody={new.nbody}, njnt={new.njnt}, ngeom={new.ngeom}, nu={new.nu}")
    print(f"  total mass: {new.body_mass.sum():.12f} kg")
    print(f"  all hinge armatures: {NUMERICAL_ARMATURE:.1e} kg m^2 numerical regularizer")
    print("  active joints: 211 with damping=stiffness=0")
    print("  passive hinges: 76 physical spring/damper terms preserved")


def build(source: Path, output: Path) -> None:
    text = source.read_text(encoding="utf-8")
    text = text.replace(
        '<joint damping="4.0754e-06" armature="0.0001" frictionloss="0.0"/>',
        f'<joint damping="0" armature="{NUMERICAL_ARMATURE:.1e}" frictionloss="0.0"/>',
        1,
    )
    # The clean Isaac-Lab plant uses a 1 ms physics step.  The refined model
    # adopts that same step after a six-second finite-state gate passed.
    text = text.replace('timestep="0.0005"', 'timestep="0.001"', 1)
    marker = '<mujoco model="centipede_robot">'
    provenance = (
        marker
        + "\n  <!-- Reliability refinement: old 1e-4 armature replaced by a tested "
        + "1e-5 MuJoCo collision-chain regularizer (not physical inertia); active "
        + "passive-PD removed; passive "
        + "leg-roll and foot spring/damper terms preserved. -->"
    )
    text = text.replace(marker, provenance, 1)
    text = re.sub(r"<joint\b[^>]*?/>", rewrite_joint, text)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    validate(source, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
