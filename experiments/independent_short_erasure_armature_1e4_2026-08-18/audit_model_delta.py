"""Audit the final pre-noise FARMS model against the untouched source MJCF."""

from pathlib import Path

import mujoco
import numpy as np

import refine_model


ROOT = Path(__file__).resolve().parent
SOURCE = refine_model.DEFAULT_SOURCE
NEW = ROOT / "models" / "farms" / "centipede_reliability_refined.xml"


def exact(label: str, old: np.ndarray, new: np.ndarray) -> None:
    if not np.array_equal(old, new):
        difference = np.max(np.abs(np.asarray(old, dtype=float) - np.asarray(new, dtype=float)))
        raise RuntimeError(f"Unexpected change in {label}; max absolute difference={difference:.6g}")


def main() -> None:
    source = mujoco.MjModel.from_xml_path(str(SOURCE))
    new = mujoco.MjModel.from_xml_path(str(NEW))
    if (source.nq, source.nv, source.nu, source.nbody, source.njnt, source.ngeom) != (
        new.nq, new.nv, new.nu, new.nbody, new.njnt, new.ngeom
    ):
        raise RuntimeError("Topology changed")

    unchanged = {
        "body_mass": (source.body_mass, new.body_mass),
        "body_inertia": (source.body_inertia, new.body_inertia),
        "dof_armature": (source.dof_armature, new.dof_armature),
        "dof_frictionloss": (source.dof_frictionloss, new.dof_frictionloss),
        "geom_pos": (source.geom_pos, new.geom_pos),
        "geom_quat": (source.geom_quat, new.geom_quat),
        "geom_size": (source.geom_size, new.geom_size),
        "geom_friction": (source.geom_friction, new.geom_friction),
        "geom_contype": (source.geom_contype, new.geom_contype),
        "actuator_gainprm": (source.actuator_gainprm, new.actuator_gainprm),
        "actuator_biasprm": (source.actuator_biasprm, new.actuator_biasprm),
        "actuator_ctrlrange": (source.actuator_ctrlrange, new.actuator_ctrlrange),
    }
    for label, pair in unchanged.items():
        exact(label, *pair)
    if source.opt.timestep != new.opt.timestep or source.opt.integrator != new.opt.integrator:
        raise RuntimeError("Timestep or integrator changed")
    if not np.all(new.dof_armature[6:] == 1.0e-4):
        raise RuntimeError("Hinge armature is not the source 1e-4 value")

    active_ids: list[int] = []
    passive_ids: list[int] = []
    robot_collision_geoms: list[int] = []
    for joint_id in range(new.njnt):
        name = mujoco.mj_id2name(new, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
        if refine_model.is_active(name):
            active_ids.append(joint_id)
        elif name.startswith("joint_roll_leg_") or name.startswith("joint_foot_"):
            passive_ids.append(joint_id)
    for geom_id in range(new.ngeom):
        name = mujoco.mj_id2name(new, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name.endswith("_collision"):
            robot_collision_geoms.append(geom_id)

    if (len(active_ids), len(passive_ids), len(robot_collision_geoms)) != (211, 76, 211):
        raise RuntimeError("Expected active/passive/collision counts changed")
    active_dofs = np.asarray([new.jnt_dofadr[joint_id] for joint_id in active_ids], dtype=int)
    if np.any(new.dof_damping[active_dofs] != 0.0) or np.any(new.jnt_stiffness[active_ids] != 0.0):
        raise RuntimeError("Active passive-PD was not fully removed")
    for joint_id in passive_ids:
        name = mujoco.mj_id2name(new, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        source_id = mujoco.mj_name2id(source, mujoco.mjtObj.mjOBJ_JOINT, name)
        source_dof = int(source.jnt_dofadr[source_id])
        new_dof = int(new.jnt_dofadr[joint_id])
        if source.dof_damping[source_dof] != new.dof_damping[new_dof]:
            raise RuntimeError(f"Passive damping changed for {name}")
        if source.jnt_stiffness[source_id] != new.jnt_stiffness[joint_id]:
            raise RuntimeError(f"Passive stiffness changed for {name}")

    changed_affinity = np.flatnonzero(source.geom_conaffinity != new.geom_conaffinity)
    if not np.array_equal(np.sort(changed_affinity), np.sort(robot_collision_geoms)):
        raise RuntimeError("Unexpected geometry-affinity delta")
    if np.any(new.geom_conaffinity[robot_collision_geoms] != 0):
        raise RuntimeError("Robot self-collision is not fully disabled")

    print("PASS: final model contract")
    print("  armature: unchanged at 1e-4 kg m^2")
    print(f"  timestep/integrator: unchanged at {new.opt.timestep:g} s / {new.opt.integrator}")
    print("  active passive-PD removed: 211 joints")
    print("  passive spring/damper preserved: 76 hinges")
    print("  robot-floor ON; robot self-collision OFF: 211 collision geoms")
    print("  topology, mass, inertia, geometry, friction, and actuators unchanged")


if __name__ == "__main__":
    main()
