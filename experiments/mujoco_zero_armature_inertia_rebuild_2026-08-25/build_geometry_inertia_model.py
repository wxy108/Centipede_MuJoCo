"""Build a zero-armature Centipede Model whose inertia comes from collision geometry.

The source MJCF is read-only.  All generated files stay in this isolated Model
experiment.  Collision geoms (group 0) are the sole inertial source; visual
duplicates (group 2) are excluded.  A single material density is solved so the
rebuilt model preserves the source model's total mass.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE_MODEL = (
    ROOT.parent
    / "independent_short_erasure_2026-08-18"
    / "models"
    / "farms"
    / "centipede_reliability_refined.xml"
)
OUTPUT_MODEL = ROOT / "models" / "centipede_geometry_inertia_armature0.xml"
AUDIT_PATH = ROOT / "outputs" / "geometry_inertia_build_audit.json"
REFERENCE_DENSITY_KG_M3 = 1000.0


def collapse_zero_offset_spine_carriers(root: ET.Element) -> list[dict[str, str]]:
    """Merge each massless pitch/roll carrier with its physical child link.

    The carrier and child frames are coincident.  Keeping three orthogonal
    hinges on the now-physical child preserves the pitch/roll/yaw kinematics
    while ensuring those DOFs are backed by the segment's mesh inertia.
    """

    collapsed = []
    for index in range(20):
        carrier = root.find(f".//body[@name='link_pitch_body_{index}']")
        child_name = f"link_body_{index + 1}"
        if carrier is None:
            raise RuntimeError(f"Missing spine carrier link_pitch_body_{index}")
        child = carrier.find(f"body[@name='{child_name}']")
        if child is None:
            raise RuntimeError(f"{carrier.get('name')} does not directly contain {child_name}")
        if child.get("pos", "0 0 0") != "0 0 0" or child.get("quat") is not None or child.get("euler") is not None:
            raise RuntimeError(f"Cannot safely collapse non-coincident body {child_name}")

        carrier.set("name", child_name)
        for element in list(child):
            carrier.append(element)
        carrier.remove(child)
        collapsed.append({"carrier": f"link_pitch_body_{index}", "physical_body": child_name})
    return collapsed


def remove_redundant_leg_roll_joints(root: ET.Element) -> list[str]:
    """Remove the passive joint exactly coaxial with active legbase pitch.

    In every leg the legacy ``joint_roll_leg`` hinge is at the same point and
    on the same axis (opposite sign on the right) as ``joint_leg_*_1``.  It is
    not one of the 211 active Robot/Model coordinates.  With zero armature the
    pair creates a duplicate generalized direction rather than a physical
    additional link.
    """

    parent = {child: node for node in root.iter() for child in node}
    removed = []
    for joint in list(root.findall(".//joint")):
        name = joint.get("name", "")
        if not name.startswith("joint_roll_leg_"):
            continue
        parent[joint].remove(joint)
        removed.append(name)
    if len(removed) != 38:
        raise RuntimeError(f"Expected 38 redundant leg-roll joints, found {len(removed)}")
    return removed


def set_geometry_inertia_contract(tree: ET.ElementTree, density: float) -> None:
    root = tree.getroot()
    collapse_zero_offset_spine_carriers(root)
    remove_redundant_leg_roll_joints(root)
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("inertiafromgeom", "true")
    compiler.set("inertiagrouprange", "0 0")
    compiler.set("balanceinertia", "false")
    compiler.set("meshdir", SOURCE_MODEL.parent.joinpath("meshes").as_posix())

    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(1, option)
    option.set("timestep", "0.001")
    option.set("integrator", "implicitfast")

    for body in root.findall(".//body"):
        for inertial in list(body.findall("inertial")):
            body.remove(inertial)

    for joint in root.findall(".//joint"):
        joint.set("armature", "0")

    for default_joint in root.findall(".//default/joint"):
        default_joint.set("armature", "0")

    for geom in root.findall(".//geom"):
        group = int(geom.get("group", "0"))
        if group == 0:
            geom.attrib.pop("mass", None)
            geom.set("density", f"{density:.17g}")


def write_candidate(density: float) -> None:
    tree = ET.parse(SOURCE_MODEL)
    set_geometry_inertia_contract(tree, density)
    ET.indent(tree, space="  ")
    OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    tree.write(OUTPUT_MODEL, encoding="utf-8", xml_declaration=True)


def mass_matrix_diagnostics(model: mujoco.MjModel) -> dict[str, float | int | bool]:
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_crb(model, data)
    mujoco.mj_makeM(model, data)
    dense = np.empty((model.nv, model.nv), dtype=float)
    mujoco.mj_fullM(model, data, dense)
    eigenvalues = np.linalg.eigvalsh(dense)
    return {
        "minimum_eigenvalue": float(eigenvalues[0]),
        "maximum_eigenvalue": float(eigenvalues[-1]),
        "condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        "positive_definite": bool(eigenvalues[0] > 0.0),
        "rank": int(np.linalg.matrix_rank(dense)),
        "nv": int(model.nv),
    }


def body_rows(model: mujoco.MjModel) -> list[dict[str, object]]:
    rows = []
    for body_id in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
        rows.append(
            {
                "name": name,
                "mass_kg": float(model.body_mass[body_id]),
                "diaginertia_kg_m2": [float(value) for value in model.body_inertia[body_id]],
                "joint_count": int(model.body_jntnum[body_id]),
                "geom_count": int(model.body_geomnum[body_id]),
            }
        )
    return rows


def main() -> None:
    source = mujoco.MjModel.from_xml_path(str(SOURCE_MODEL))
    target_mass = float(np.sum(source.body_mass))

    write_candidate(REFERENCE_DENSITY_KG_M3)
    provisional = mujoco.MjModel.from_xml_path(str(OUTPUT_MODEL))
    provisional_mass = float(np.sum(provisional.body_mass))
    if not math.isfinite(provisional_mass) or provisional_mass <= 0.0:
        raise RuntimeError(f"Invalid geometry-derived provisional mass: {provisional_mass}")

    solved_density = REFERENCE_DENSITY_KG_M3 * target_mass / provisional_mass
    write_candidate(solved_density)
    rebuilt = mujoco.MjModel.from_xml_path(str(OUTPUT_MODEL))

    try:
        forward_data = mujoco.MjData(rebuilt)
        mujoco.mj_forward(rebuilt, forward_data)
        forward_status = "pass"
    except Exception as exc:  # evidence is written before re-raising below
        forward_status = f"fail: {type(exc).__name__}: {exc}"

    source_rows = body_rows(source)
    rebuilt_rows = body_rows(rebuilt)
    source_by_name = {row["name"]: row for row in source_rows}
    comparisons = []
    for row in rebuilt_rows:
        prior = source_by_name[row["name"]]
        comparisons.append(
            {
                "name": row["name"],
                "joint_count": row["joint_count"],
                "geom_count": row["geom_count"],
                "source_mass_kg": prior["mass_kg"],
                "rebuilt_mass_kg": row["mass_kg"],
                "source_diaginertia_kg_m2": prior["diaginertia_kg_m2"],
                "rebuilt_diaginertia_kg_m2": row["diaginertia_kg_m2"],
            }
        )

    joint_bodies = [row for row in rebuilt_rows if int(row["joint_count"]) > 0]
    audit = {
        "scope": "Centipede Model only; Robot repositories are read-only references",
        "source_model": str(SOURCE_MODEL),
        "output_model": str(OUTPUT_MODEL),
        "method": {
            "inertiafromgeom": True,
            "inertiagrouprange": [0, 0],
            "visual_geometries_excluded": True,
            "uniform_density_kg_m3": solved_density,
            "total_mass_preserved": True,
            "armature_kg_m2": 0.0,
            "collapsed_zero_offset_spine_carriers": 20,
            "removed_coaxial_passive_leg_roll_joints": 38,
            "active_joint_coordinates_preserved": 211,
        },
        "source_total_mass_kg": target_mass,
        "provisional_geometry_mass_at_1000_kg_m3": provisional_mass,
        "rebuilt_total_mass_kg": float(np.sum(rebuilt.body_mass)),
        "mass_error_kg": float(np.sum(rebuilt.body_mass)) - target_mass,
        "model_counts": {
            "nbody": int(rebuilt.nbody),
            "njnt": int(rebuilt.njnt),
            "nv": int(rebuilt.nv),
            "ngeom": int(rebuilt.ngeom),
        },
        "joint_bodies_with_zero_mass": [
            row["name"] for row in joint_bodies if float(row["mass_kg"]) == 0.0
        ],
        "minimum_positive_body_mass_kg": min(
            float(row["mass_kg"]) for row in rebuilt_rows if float(row["mass_kg"]) > 0.0
        ),
        "maximum_armature_kg_m2": float(np.max(rebuilt.dof_armature)),
        "mass_matrix_at_reset": mass_matrix_diagnostics(rebuilt),
        "mj_forward_at_reset": forward_status,
        "body_comparison": comparisons,
    }
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in audit.items() if key != "body_comparison"}, indent=2))

    if forward_status != "pass":
        raise RuntimeError(forward_status)


if __name__ == "__main__":
    main()
