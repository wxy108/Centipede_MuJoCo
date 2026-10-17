"""Build single-factor FARMS plant variants for the pre-noise stability gate."""

from __future__ import annotations

import re
from pathlib import Path

import mujoco
import numpy as np

import refine_model


ROOT = Path(__file__).resolve().parent
SOURCE = refine_model.DEFAULT_SOURCE
MODEL_DIR = ROOT / "models" / "farms"
SELF_OFF_ONLY = MODEL_DIR / "centipede_self_collision_off_original_pd_dt0p5.xml"
SELF_OFF_ACTIVE_PD_REMOVED = MODEL_DIR / "centipede_self_collision_off_active_pd_removed_dt0p5.xml"


def compile_model(path: Path) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(path))


def set_self_collision_off(text: str) -> str:
    return re.sub(r"<geom\b[^>]*?/>", refine_model.rewrite_geom, text)


def remove_active_passive_pd(text: str) -> str:
    text = text.replace(
        '<joint damping="4.0754e-06" armature="0.0001" frictionloss="0.0"/>',
        '<joint damping="0" armature="1.0e-04" frictionloss="0.0"/>',
        1,
    )
    return re.sub(r"<joint\b[^>]*?/>", refine_model.rewrite_joint, text)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    compile_model(path)


def exact(label: str, left: np.ndarray, right: np.ndarray) -> None:
    if not np.array_equal(left, right):
        raise RuntimeError(f"Unexpected change in {label}")


def main() -> None:
    source_text = SOURCE.read_text(encoding="utf-8")
    write(SELF_OFF_ONLY, set_self_collision_off(source_text))
    write(
        SELF_OFF_ACTIVE_PD_REMOVED,
        remove_active_passive_pd(set_self_collision_off(source_text)),
    )

    source = compile_model(SOURCE)
    self_off = compile_model(SELF_OFF_ONLY)
    pd_removed = compile_model(SELF_OFF_ACTIVE_PD_REMOVED)
    for label, model in (("self_off", self_off), ("pd_removed", pd_removed)):
        if (model.nbody, model.njnt, model.ngeom, model.nu) != (
            source.nbody, source.njnt, source.ngeom, source.nu
        ):
            raise RuntimeError(f"Topology drift in {label}")
        exact(f"{label} mass", source.body_mass, model.body_mass)
        exact(f"{label} inertia", source.body_inertia, model.body_inertia)
        exact(f"{label} armature", source.dof_armature, model.dof_armature)
        if model.opt.timestep != source.opt.timestep:
            raise RuntimeError(f"Timestep drift in {label}")

    # Variant B must differ from source only in the collision affinity mask.
    for label, left, right in (
        ("stiffness", source.jnt_stiffness, self_off.jnt_stiffness),
        ("damping", source.dof_damping, self_off.dof_damping),
        ("friction", source.geom_friction, self_off.geom_friction),
        ("contype", source.geom_contype, self_off.geom_contype),
    ):
        exact(f"self_off {label}", left, right)
    changed_affinity = np.flatnonzero(source.geom_conaffinity != self_off.geom_conaffinity)
    if len(changed_affinity) != 211 or not np.all(self_off.geom_conaffinity[changed_affinity] == 0):
        raise RuntimeError("Self-collision-only mask is not the expected 211 geoms")

    # Variant C adds only active-joint passive stiffness/damping removal.
    for label, left, right in (
        ("armature", self_off.dof_armature, pd_removed.dof_armature),
        ("geom affinity", self_off.geom_conaffinity, pd_removed.geom_conaffinity),
        ("geom friction", self_off.geom_friction, pd_removed.geom_friction),
    ):
        exact(f"pd_removed {label}", left, right)
    changed_damping = np.flatnonzero(self_off.dof_damping != pd_removed.dof_damping)
    changed_stiffness = np.flatnonzero(self_off.jnt_stiffness != pd_removed.jnt_stiffness)

    print("PASS: stability variants built as single-factor deltas")
    print(f"  A source: self collision ON, original PD, dt={source.opt.timestep:g}")
    print(f"  B: self collision OFF only ({len(changed_affinity)} geom affinities)")
    print(
        "  C: B + active passive-PD removal "
        f"({len(changed_damping)} damping, {len(changed_stiffness)} stiffness entries)"
    )
    print("  armature unchanged at 1e-4 kg m^2 in A/B/C")


if __name__ == "__main__":
    main()
