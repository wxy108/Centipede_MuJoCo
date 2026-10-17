"""Build a true one-body distal leg while preserving reach and projected inertia."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
SOURCE = PROJECT / "experiments" / "mujoco_template_locked_head_2026-08-27" / "models" / "centipede_flat_locked_head_m29p0.xml"
MODEL = ROOT / "models" / "centipede_flat_locked_head_m29p0_rigid_tip.xml"
OUT = ROOT / "outputs"
TIP_BODY = re.compile(r"^link_leg_(\d+)_(L|R)_tip$")
TIP_JOINT = re.compile(r"^joint_tip_(\d+)_(L|R)_roll$")
FOOT_BODY = re.compile(r"^foot_\d+_[01]$")
KNEE_JOINT = re.compile(r"^joint_leg_\d+_[LR]_[23]$")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numbers(text: str | None, count: int, default=0.0) -> np.ndarray:
    if text is None:
        return np.full(count, default, dtype=float)
    value = np.fromstring(text, sep=" ", dtype=float)
    if value.size != count:
        raise RuntimeError(f"expected {count} values, got {value.size}: {text}")
    return value


def rotation_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)))


def rotation_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)))


def rotation_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


def quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat / np.linalg.norm(quat)
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
    )


def element_rotation(element: ET.Element) -> np.ndarray:
    if element.get("quat"):
        return quat_to_matrix(numbers(element.get("quat"), 4))
    euler = numbers(element.get("euler"), 3)
    return rotation_x(euler[0]) @ rotation_y(euler[1]) @ rotation_z(euler[2])


def matrix_quat(matrix: np.ndarray) -> str:
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.asarray(matrix, dtype=float).reshape(9))
    return " ".join(f"{value:.17g}" for value in quat)


def vector_text(value: np.ndarray) -> str:
    return " ".join(f"{number:.17g}" for number in value)


def promote_fixed_subtree(source_body, destination_body, parent_rotation, parent_translation):
    """Move geom/site elements into destination in destination coordinates."""
    geom_count = site_count = 0
    for child in list(source_body):
        if child.tag == "joint":
            if not TIP_JOINT.fullmatch(child.get("name", "")):
                raise RuntimeError(f"unexpected joint in distal subtree: {child.get('name')}")
            continue
        if child.tag == "body":
            child_translation = parent_translation + parent_rotation @ numbers(child.get("pos"), 3)
            child_rotation = parent_rotation @ element_rotation(child)
            geoms, sites = promote_fixed_subtree(child, destination_body, child_rotation, child_translation)
            geom_count += geoms
            site_count += sites
            continue
        if child.tag not in ("geom", "site"):
            raise RuntimeError(f"unexpected distal element {child.tag}")
        promoted = copy.deepcopy(child)
        if promoted.get("fromto"):
            endpoints = numbers(promoted.get("fromto"), 6).reshape(2, 3)
            endpoints = (parent_rotation @ endpoints.T).T + parent_translation
            promoted.set("fromto", vector_text(endpoints.reshape(-1)))
            for attribute in ("pos", "quat", "euler"):
                promoted.attrib.pop(attribute, None)
        else:
            position = parent_translation + parent_rotation @ numbers(promoted.get("pos"), 3)
            promoted.set("pos", vector_text(position))
            if promoted.get("quat") or promoted.get("euler"):
                promoted.set("quat", matrix_quat(parent_rotation @ element_rotation(promoted)))
                promoted.attrib.pop("euler", None)
            elif promoted.tag == "geom" and promoted.get("type", "sphere") != "sphere":
                promoted.set("quat", matrix_quat(parent_rotation))
        destination_body.append(promoted)
        geom_count += child.tag == "geom"
        site_count += child.tag == "site"
    return int(geom_count), int(site_count)


def object_names(model, objtype, count):
    return [mujoco.mj_id2name(model, objtype, index) or "" for index in range(count)]


def qpos_width(joint_type: int) -> int:
    return 7 if joint_type == mujoco.mjtJoint.mjJNT_FREE else (4 if joint_type == mujoco.mjtJoint.mjJNT_BALL else 1)


def dof_width(joint_type: int) -> int:
    return 6 if joint_type == mujoco.mjtJoint.mjJNT_FREE else (3 if joint_type == mujoco.mjtJoint.mjJNT_BALL else 1)


def common_joint_maps(source, rigid):
    source_by_name = {mujoco.mj_id2name(source, mujoco.mjtObj.mjOBJ_JOINT, joint): joint for joint in range(source.njnt)}
    qpos_pairs, dof_pairs = [], []
    for rigid_joint in range(rigid.njnt):
        name = mujoco.mj_id2name(rigid, mujoco.mjtObj.mjOBJ_JOINT, rigid_joint)
        source_joint = source_by_name[name]
        source_type, rigid_type = int(source.jnt_type[source_joint]), int(rigid.jnt_type[rigid_joint])
        if source_type != rigid_type:
            raise RuntimeError(f"joint type changed for {name}")
        for offset in range(qpos_width(rigid_type)):
            qpos_pairs.append((int(source.jnt_qposadr[source_joint]) + offset, int(rigid.jnt_qposadr[rigid_joint]) + offset))
        for offset in range(dof_width(rigid_type)):
            dof_pairs.append((int(source.jnt_dofadr[source_joint]) + offset, int(rigid.jnt_dofadr[rigid_joint]) + offset))
    return qpos_pairs, dof_pairs


def build() -> dict[str, object]:
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(SOURCE)
    root = tree.getroot()
    root.set("model", "centipede_locked_head_m29p0_onepiece_leg")
    parent = {child: node for node in root.iter() for child in node}
    tip_bodies = [body for body in root.iter("body") if TIP_BODY.fullmatch(body.get("name", ""))]
    if len(tip_bodies) != 38:
        raise RuntimeError(f"expected 38 tip bodies, found {len(tip_bodies)}")
    removed_tip_joints, removed_foot_bodies = [], []
    promoted_geoms = promoted_sites = 0
    for tip_body in tip_bodies:
        destination = parent[tip_body]
        joints = [child for child in list(tip_body) if child.tag == "joint"]
        if len(joints) != 1 or not TIP_JOINT.fullmatch(joints[0].get("name", "")):
            raise RuntimeError(f"unexpected tip topology in {tip_body.get('name')}")
        removed_tip_joints.append(joints[0].get("name"))
        removed_foot_bodies.extend(child.get("name") for child in list(tip_body) if child.tag == "body" and FOOT_BODY.fullmatch(child.get("name", "")))
        geoms, sites = promote_fixed_subtree(tip_body, destination, element_rotation(tip_body), numbers(tip_body.get("pos"), 3))
        promoted_geoms += geoms
        promoted_sites += sites
        destination.remove(tip_body)
    if len(removed_tip_joints) != 38 or len(removed_foot_bodies) != 38:
        raise RuntimeError("tip/foot removal count mismatch")
    tree.write(MODEL, encoding="utf-8", xml_declaration=True)

    source = mujoco.MjModel.from_xml_path(str(SOURCE))
    rigid = mujoco.MjModel.from_xml_path(str(MODEL))
    rigid_joint_names = object_names(rigid, mujoco.mjtObj.mjOBJ_JOINT, rigid.njnt)
    rigid_body_names = object_names(rigid, mujoco.mjtObj.mjOBJ_BODY, rigid.nbody)
    if any(TIP_JOINT.fullmatch(name) for name in rigid_joint_names):
        raise RuntimeError("a tip joint survived")
    if any(KNEE_JOINT.fullmatch(name) for name in rigid_joint_names):
        raise RuntimeError("a knee joint was reintroduced")
    if any(TIP_BODY.fullmatch(name) or FOOT_BODY.fullmatch(name) for name in rigid_body_names):
        raise RuntimeError("a distal child body survived")
    source_geom_names = object_names(source, mujoco.mjtObj.mjOBJ_GEOM, source.ngeom)
    rigid_geom_names = object_names(rigid, mujoco.mjtObj.mjOBJ_GEOM, rigid.ngeom)
    if set(source_geom_names) != set(rigid_geom_names):
        raise RuntimeError("geom set changed")
    source_geom = {name: index for index, name in enumerate(source_geom_names)}
    rigid_geom = {name: index for index, name in enumerate(rigid_geom_names)}
    geometry_fields = ("geom_type", "geom_size", "geom_friction", "geom_solref", "geom_solimp", "geom_contype", "geom_conaffinity")
    geometry_equal = {}
    for field in geometry_fields:
        left, right = getattr(source, field), getattr(rigid, field)
        geometry_equal[field] = all(
            np.allclose(left[source_geom[name]], right[rigid_geom[name]], rtol=1e-13, atol=1e-15)
            if np.issubdtype(left.dtype, np.floating)
            else np.array_equal(left[source_geom[name]], right[rigid_geom[name]])
            for name in source_geom
        )
    if not all(geometry_equal.values()):
        changed = {}
        for field, equal in geometry_equal.items():
            if equal:
                continue
            left, right = getattr(source, field), getattr(rigid, field)
            changed[field] = [
                name
                for name in source_geom
                if not (
                    np.allclose(left[source_geom[name]], right[rigid_geom[name]], rtol=1e-13, atol=1e-15)
                    if np.issubdtype(left.dtype, np.floating)
                    else np.array_equal(left[source_geom[name]], right[rigid_geom[name]])
                )
            ]
        raise RuntimeError(f"a non-pose geometry/contact field changed: {changed}")
    if not np.isclose(source.body_mass.sum(), rigid.body_mass.sum(), rtol=0, atol=1e-15):
        raise RuntimeError("total mass changed")

    qpos_pairs, dof_pairs = common_joint_maps(source, rigid)
    source_dofs = np.asarray([pair[0] for pair in dof_pairs], dtype=int)
    rigid_dofs = np.asarray([pair[1] for pair in dof_pairs], dtype=int)
    rng = np.random.default_rng(9102026)
    max_geom_position_error = max_geom_orientation_error = max_foot_site_error = max_mass_matrix_error = 0.0
    source_data, rigid_data = mujoco.MjData(source), mujoco.MjData(rigid)
    for trial in range(7):
        mujoco.mj_resetData(source, source_data)
        mujoco.mj_resetData(rigid, rigid_data)
        if trial:
            for joint in range(rigid.njnt):
                if int(rigid.jnt_type[joint]) != mujoco.mjtJoint.mjJNT_HINGE:
                    continue
                address = int(rigid.jnt_qposadr[joint])
                if bool(rigid.jnt_limited[joint]):
                    low, high = rigid.jnt_range[joint]
                    margin = 0.2 * (high - low)
                    rigid_data.qpos[address] = rng.uniform(low + margin, high - margin)
                else:
                    rigid_data.qpos[address] = rng.uniform(-0.15, 0.15)
        for source_qpos, rigid_qpos in qpos_pairs:
            source_data.qpos[source_qpos] = rigid_data.qpos[rigid_qpos]
        mujoco.mj_forward(source, source_data)
        mujoco.mj_forward(rigid, rigid_data)
        for name in source_geom:
            gs, gr = source_geom[name], rigid_geom[name]
            max_geom_position_error = max(max_geom_position_error, float(np.abs(source_data.geom_xpos[gs] - rigid_data.geom_xpos[gr]).max()))
            geom_type = int(source.geom_type[gs])
            if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                orientation_error = 0.0
            elif geom_type in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
                # Axially symmetric geoms can legitimately have a different
                # arbitrary roll basis after recompilation. Compare their
                # physical symmetry axis, allowing equivalent sign reversal.
                source_axis = source_data.geom_xmat[gs].reshape(3, 3)[:, 2]
                rigid_axis = rigid_data.geom_xmat[gr].reshape(3, 3)[:, 2]
                orientation_error = min(
                    float(np.abs(source_axis - rigid_axis).max()),
                    float(np.abs(source_axis + rigid_axis).max()),
                )
            else:
                orientation_error = float(np.abs(source_data.geom_xmat[gs] - rigid_data.geom_xmat[gr]).max())
            max_geom_orientation_error = max(max_geom_orientation_error, orientation_error)
        for segment in range(19):
            for side in range(2):
                name = f"site_foot_{segment}_{side}"
                ss = mujoco.mj_name2id(source, mujoco.mjtObj.mjOBJ_SITE, name)
                sr = mujoco.mj_name2id(rigid, mujoco.mjtObj.mjOBJ_SITE, name)
                max_foot_site_error = max(max_foot_site_error, float(np.abs(source_data.site_xpos[ss] - rigid_data.site_xpos[sr]).max()))
        source_matrix, rigid_matrix = np.zeros((source.nv, source.nv)), np.zeros((rigid.nv, rigid.nv))
        mujoco.mj_fullM(source, source_matrix, source_data.qM)
        mujoco.mj_fullM(rigid, rigid_matrix, rigid_data.qM)
        max_mass_matrix_error = max(max_mass_matrix_error, float(np.abs(source_matrix[np.ix_(source_dofs, source_dofs)] - rigid_matrix[np.ix_(rigid_dofs, rigid_dofs)]).max()))
    if max(max_geom_position_error, max_foot_site_error, max_geom_orientation_error) > 1e-11:
        raise RuntimeError(
            "geometry or leg reach changed under random joint poses: "
            f"geom_pos={max_geom_position_error}, geom_rot={max_geom_orientation_error}, "
            f"foot={max_foot_site_error}"
        )
    if max_mass_matrix_error > 1e-14:
        raise RuntimeError("projected active mass matrix changed")

    result = {
        "scope": "Centipede Model only",
        "source_model": str(SOURCE),
        "source_sha256": sha256(SOURCE),
        "rigid_tip_model": str(MODEL),
        "rigid_tip_sha256": sha256(MODEL),
        "removed_tip_joint_count": len(removed_tip_joints),
        "removed_tip_body_count": len(tip_bodies),
        "removed_foot_body_count": len(removed_foot_bodies),
        "promoted_geom_count": promoted_geoms,
        "promoted_site_count": promoted_sites,
        "tip_roll_dof_count": 0,
        "knee_joint_count": sum(bool(KNEE_JOINT.fullmatch(name)) for name in rigid_joint_names),
        "active_leg_dof_per_leg": 2,
        "distal_fixed_child_body_count": 0,
        "nominal_leg_geometry_unchanged": True,
        "random_pose_trials": 7,
        "all_geom_world_position_max_error_m": max_geom_position_error,
        "all_geom_world_orientation_max_error": max_geom_orientation_error,
        "foot_site_max_error_m": max_foot_site_error,
        "projected_active_mass_matrix_max_error": max_mass_matrix_error,
        "geometry_and_contact_fields_equal_by_name": geometry_equal,
        "source_total_mass_kg": float(source.body_mass.sum()),
        "rigid_total_mass_kg": float(rigid.body_mass.sum()),
        "source_nq_nv_njnt_nbody": [int(source.nq), int(source.nv), int(source.njnt), int(source.nbody)],
        "rigid_nq_nv_njnt_nbody": [int(rigid.nq), int(rigid.nv), int(rigid.njnt), int(rigid.nbody)],
        "removed_generalized_dofs": int(source.nv - rigid.nv),
        "one_piece_interpretation": "distal capsule and foot geoms/sites are directly owned by the leg-pitch body; no distal, knee, fixed-tip, or fixed-foot body",
    }
    (OUT / "topology_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
