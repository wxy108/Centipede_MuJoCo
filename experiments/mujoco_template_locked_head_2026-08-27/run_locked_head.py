"""Parameterized fixed-head export for the abstract Centipede Model.

The selected distal-tip model has two co-located joints between the head
(link_body_0) and link_body_1.  This experiment removes both joints and their
actuator/sensor references, writes the requested pitch as a fixed body-frame
rotation, and exports a combined head/link1 STL for visualization and CAD use.

The combined STL is intentionally visual-only.  The original link_0/link_1
collision meshes remain separate, so MuJoCo still derives mass and inertia
from the original physical geometry instead of silently changing dynamics.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import struct
import sys
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parent
TIP_ROOT = ROOT.parent / "mujoco_template_tip_compliance_2026-08-27"
TIP_SCRIPT = TIP_ROOT / "run_tip_compliance.py"
MODELS = ROOT / "models"
ASSETS = ROOT / "assets"
OUTPUTS = ROOT / "outputs"
DATA = OUTPUTS / "data"
FIGURES = OUTPUTS / "figures"
VIDEOS = OUTPUTS / "videos"

DEFAULT_HEAD_PITCH_DEG = -29.0
SOURCE_MODELS = {
    "flat": TIP_ROOT / "models" / "centipede_flat_tip_k4p5e-6_r38_bodyk6_distal_roll.xml",
    "rough": TIP_ROOT / "models" / "centipede_rough_tip_k4p5e-6_r38_bodyk6_distal_roll.xml",
}
REMOVED_JOINTS = {"joint_pitch_body_0", "joint_roll_body_0"}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tip = load_module(TIP_SCRIPT, "centipede_locked_head_tip_source")


def ensure_directories() -> None:
    for path in (MODELS, ASSETS, DATA, FIGURES, VIDEOS):
        path.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def fmt(value: float) -> str:
    return f"{value:.12g}"


def angle_tag(angle_deg: float) -> str:
    sign = "m" if angle_deg < 0 else "p"
    return sign + f"{abs(angle_deg):.1f}".replace(".", "p")


def parse_vec(text: str, count: int) -> np.ndarray:
    values = np.asarray([float(item) for item in text.split()], dtype=float)
    if len(values) != count:
        raise ValueError(f"Expected {count} values, got {text!r}")
    return values


def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    triangles: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
            elif line.startswith("f "):
                face = []
                for token in line.split()[1:]:
                    raw = int(token.split("/", 1)[0])
                    face.append(raw - 1 if raw > 0 else len(vertices) + raw)
                for index in range(1, len(face) - 1):
                    triangles.append([face[0], face[index], face[index + 1]])
    if not vertices or not triangles:
        raise ValueError(f"OBJ contains no usable mesh: {path}")
    return np.asarray(vertices, dtype=float), np.asarray(triangles, dtype=np.int32)


def mesh_topology(triangles: np.ndarray) -> dict[str, int | bool]:
    edges: Counter[tuple[int, int]] = Counter()
    for triangle in triangles:
        for a, b in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
            edges[tuple(sorted((int(a), int(b))))] += 1
    boundary = sum(count == 1 for count in edges.values())
    nonmanifold = sum(count > 2 for count in edges.values())
    return {
        "unique_edge_count": len(edges),
        "boundary_edge_count": boundary,
        "nonmanifold_edge_count": nonmanifold,
        "watertight_by_index": boundary == 0 and nonmanifold == 0,
    }


def write_binary_stl(path: Path, vertices: np.ndarray, triangles: np.ndarray, header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = header.encode("ascii", errors="replace")[:80].ljust(80, b" ")
    with path.open("wb") as handle:
        handle.write(encoded)
        handle.write(struct.pack("<I", len(triangles)))
        for ids in triangles:
            points = vertices[ids]
            normal = np.cross(points[1] - points[0], points[2] - points[0])
            norm = float(np.linalg.norm(normal))
            if norm > 0.0:
                normal /= norm
            else:
                normal[:] = 0.0
            handle.write(struct.pack("<12fH", *normal, *points.reshape(-1), 0))


def source_mesh_info(source_xml: Path, mesh_name: str) -> tuple[Path, np.ndarray]:
    root = ET.parse(source_xml).getroot()
    compiler = root.find("compiler")
    asset = root.find("asset")
    if compiler is None or asset is None:
        raise RuntimeError("Source XML lacks compiler/asset")
    mesh = asset.find(f"mesh[@name='{mesh_name}']")
    if mesh is None:
        raise RuntimeError(f"Missing mesh asset {mesh_name}")
    meshdir = Path(compiler.get("meshdir", str(source_xml.parent)))
    file_path = Path(mesh.get("file", ""))
    if not file_path.is_absolute():
        file_path = meshdir / file_path
    scale = parse_vec(mesh.get("scale", "1 1 1"), 3)
    return file_path.resolve(), scale


def build_combined_stl(source_xml: Path, angle_deg: float, output: Path) -> dict[str, object]:
    path0, scale0 = source_mesh_info(source_xml, "link_0")
    path1, scale1 = source_mesh_info(source_xml, "link_1")
    vertices0, triangles0 = read_obj(path0)
    vertices1, triangles1 = read_obj(path1)
    vertices0 *= scale0
    vertices1 *= scale1

    angle = math.radians(angle_deg)
    rotation_y = np.asarray(
        [
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ],
        dtype=float,
    )
    vertices1 = vertices1 @ rotation_y.T + np.asarray([0.0075000003, 0.0, 0.0])
    triangles1 = triangles1 + len(vertices0)
    vertices = np.vstack((vertices0, vertices1))
    triangles = np.vstack((triangles0, triangles1))
    write_binary_stl(
        output,
        vertices,
        triangles,
        f"Centipede Model head+link1 fixed pitch {angle_deg:+.3f} deg; visual/export only",
    )
    topology0 = mesh_topology(triangles0)
    topology1 = mesh_topology(triangles1 - len(vertices0))
    triangle_points = vertices[triangles]
    areas = 0.5 * np.linalg.norm(
        np.cross(triangle_points[:, 1] - triangle_points[:, 0], triangle_points[:, 2] - triangle_points[:, 0]),
        axis=1,
    )
    return {
        "path": str(output),
        "sha256": sha256(output),
        "size_bytes": output.stat().st_size,
        "source_meshes": [str(path0), str(path1)],
        "vertex_count": int(len(vertices)),
        "triangle_count": int(len(triangles)),
        "degenerate_triangle_count": int(np.count_nonzero(areas <= 1e-18)),
        "bounds_min_m": vertices.min(axis=0).tolist(),
        "bounds_max_m": vertices.max(axis=0).tolist(),
        "link_0_topology": topology0,
        "link_1_topology": topology1,
        "note": "Disconnected union is valid; each source shell is audited separately.",
    }


def remove_references(root: ET.Element) -> dict[str, list[str]]:
    removed: dict[str, list[str]] = {"actuators": [], "sensors": []}
    actuator = root.find("actuator")
    if actuator is not None:
        for element in list(actuator):
            if element.get("joint") in REMOVED_JOINTS:
                removed["actuators"].append(element.get("name", ""))
                actuator.remove(element)
    sensor = root.find("sensor")
    if sensor is not None:
        for element in list(sensor):
            if element.get("joint") in REMOVED_JOINTS:
                removed["sensors"].append(element.get("name", ""))
                sensor.remove(element)
    return removed


def transform_xml(source: Path, output: Path, stl_path: Path, angle_deg: float) -> dict[str, object]:
    tree = ET.parse(source)
    root = tree.getroot()
    root.set("model", f"centipede_locked_head_{angle_tag(angle_deg)}")
    asset = root.find("asset")
    body0 = root.find(".//body[@name='link_body_0']")
    body1 = root.find(".//body[@name='link_body_1']")
    if asset is None or body0 is None or body1 is None:
        raise RuntimeError("Source XML lacks expected asset/head bodies")

    removed_joint_elements = []
    for element in list(body1):
        if element.tag == "joint" and element.get("name") in REMOVED_JOINTS:
            removed_joint_elements.append(element.get("name", ""))
            body1.remove(element)
    if set(removed_joint_elements) != REMOVED_JOINTS:
        raise RuntimeError(f"Expected to remove {REMOVED_JOINTS}, got {removed_joint_elements}")
    body1.set("euler", f"0 {fmt(math.radians(angle_deg))} 0")

    removed_refs = remove_references(root)
    for element in root.iter():
        if any(value in REMOVED_JOINTS for value in element.attrib.values()):
            raise RuntimeError(f"Dangling removed-joint reference in <{element.tag}>: {element.attrib}")

    for body, geom_name in ((body0, "link_0_visual"), (body1, "link_1_visual")):
        geom = body.find(f"geom[@name='{geom_name}']")
        if geom is None:
            raise RuntimeError(f"Missing expected visual geom {geom_name}")
        body.remove(geom)

    mesh_name = f"head_link1_locked_{angle_tag(angle_deg)}"
    ET.SubElement(
        asset,
        "mesh",
        {
            "name": mesh_name,
            "file": str(stl_path.resolve()).replace("\\", "/"),
            "scale": "1 1 1",
        },
    )
    combined = ET.Element(
        "geom",
        {
            "type": "mesh",
            "mesh": mesh_name,
            "pos": "0 0 0",
            "contype": "0",
            "conaffinity": "0",
            "group": "2",
            "name": "head_link1_locked_visual",
            "rgba": "0.10 0.015 0.015 1",
        },
    )
    collision0 = body0.find("geom[@name='link_0_collision']")
    insert_at = list(body0).index(collision0) + 1 if collision0 is not None else 1
    body0.insert(insert_at, combined)

    custom = root.find("custom")
    if custom is None:
        custom = ET.SubElement(root, "custom")
    ET.SubElement(
        custom,
        "numeric",
        {"name": "locked_head_pitch_deg", "data": fmt(angle_deg)},
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return {
        "model": str(output),
        "sha256": sha256(output),
        "removed_joints": sorted(removed_joint_elements),
        "removed_actuators": sorted(removed_refs["actuators"]),
        "removed_sensors": sorted(removed_refs["sensors"]),
        "fixed_body_euler_rad": [0.0, math.radians(angle_deg), 0.0],
        "combined_visual_mesh": mesh_name,
        "physics_collision_meshes_retained": ["link_0_collision", "link_1_collision"],
    }


def set_joint_qpos(model: mujoco.MjModel, data: mujoco.MjData, name: str, value: float) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise RuntimeError(f"Missing joint {name}")
    data.qpos[int(model.jnt_qposadr[joint_id])] = value


def geometry_equivalence(source_path: Path, locked_path: Path, angle_deg: float) -> dict[str, object]:
    source = mujoco.MjModel.from_xml_path(str(source_path))
    locked = mujoco.MjModel.from_xml_path(str(locked_path))
    source_data = mujoco.MjData(source)
    locked_data = mujoco.MjData(locked)
    mujoco.mj_resetData(source, source_data)
    mujoco.mj_resetData(locked, locked_data)
    set_joint_qpos(source, source_data, "joint_pitch_body_0", math.radians(angle_deg))
    set_joint_qpos(source, source_data, "joint_roll_body_0", 0.0)
    mujoco.mj_forward(source, source_data)
    mujoco.mj_forward(locked, locked_data)

    position_errors = []
    rotation_errors = []
    compared = []
    for source_id in range(source.nbody):
        name = mujoco.mj_id2name(source, mujoco.mjtObj.mjOBJ_BODY, source_id)
        if name is None or name == "world":
            continue
        locked_id = mujoco.mj_name2id(locked, mujoco.mjtObj.mjOBJ_BODY, name)
        if locked_id < 0:
            raise RuntimeError(f"Locked model lost body {name}")
        position_errors.append(float(np.linalg.norm(source_data.xpos[source_id] - locked_data.xpos[locked_id])))
        rotation_errors.append(float(np.max(np.abs(source_data.xmat[source_id] - locked_data.xmat[locked_id]))))
        compared.append(name)
    return {
        "body_count_compared": len(compared),
        "max_body_position_error_m": max(position_errors, default=float("nan")),
        "max_body_rotation_matrix_abs_error": max(rotation_errors, default=float("nan")),
        "equivalent_within_1e-10": bool(
            max(position_errors, default=1.0) <= 1e-10
            and max(rotation_errors, default=1.0) <= 1e-10
        ),
    }


def mass_matrix_audit(model: mujoco.MjModel) -> dict[str, float | bool]:
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    dense = np.zeros((model.nv, model.nv), dtype=float)
    mujoco.mj_fullM(model, dense, data.qM)
    eigenvalues = np.linalg.eigvalsh(0.5 * (dense + dense.T))
    return {
        "finite": bool(np.isfinite(dense).all()),
        "symmetric_max_abs_error": float(np.max(np.abs(dense - dense.T))),
        "minimum_eigenvalue": float(eigenvalues[0]),
        "maximum_eigenvalue": float(eigenvalues[-1]),
        "condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        "positive_definite": bool(eigenvalues[0] > 0.0),
    }


def compiled_audit(source_path: Path, locked_path: Path, angle_deg: float) -> dict[str, object]:
    source = mujoco.MjModel.from_xml_path(str(source_path))
    locked = mujoco.MjModel.from_xml_path(str(locked_path))
    source_mass = np.asarray(source.body_mass)
    locked_mass = np.asarray(locked.body_mass)
    source_inertia = np.asarray(source.body_inertia)
    locked_inertia = np.asarray(locked.body_inertia)
    head_pitch_id = mujoco.mj_name2id(locked, mujoco.mjtObj.mjOBJ_JOINT, "joint_pitch_body_0")
    head_roll_id = mujoco.mj_name2id(locked, mujoco.mjtObj.mjOBJ_JOINT, "joint_roll_body_0")
    link0_geom = mujoco.mj_name2id(locked, mujoco.mjtObj.mjOBJ_GEOM, "link_0_collision")
    link1_geom = mujoco.mj_name2id(locked, mujoco.mjtObj.mjOBJ_GEOM, "link_1_collision")
    visual_geom = mujoco.mj_name2id(locked, mujoco.mjtObj.mjOBJ_GEOM, "head_link1_locked_visual")
    index = tip.LeanIndex(locked)
    tip.configure_model(locked, index)
    return {
        "source": str(source_path),
        "locked": str(locked_path),
        "source_nq_nv_nu": [int(source.nq), int(source.nv), int(source.nu)],
        "locked_nq_nv_nu": [int(locked.nq), int(locked.nv), int(locked.nu)],
        "expected_deltas_nq_nv_nu": [-2, -2, -2],
        "actual_deltas_nq_nv_nu": [
            int(locked.nq - source.nq),
            int(locked.nv - source.nv),
            int(locked.nu - source.nu),
        ],
        "head_pitch_joint_absent": head_pitch_id < 0,
        "head_roll_joint_absent": head_roll_id < 0,
        "controller_detects_fixed_head": bool(index.locked_head),
        "active_joint_count": len(index.active_joint_ids),
        "passive_body_roll_joint_count": len(index.roll_joint_ids),
        "total_mass_source_kg": float(np.sum(source_mass)),
        "total_mass_locked_kg": float(np.sum(locked_mass)),
        "total_mass_abs_delta_kg": float(abs(np.sum(locked_mass) - np.sum(source_mass))),
        "body_mass_max_abs_delta_kg": float(np.max(np.abs(locked_mass - source_mass))),
        "body_inertia_max_abs_delta_kg_m2": float(np.max(np.abs(locked_inertia - source_inertia))),
        "armature_nonzero_count": int(np.count_nonzero(np.abs(locked.dof_armature) > 1e-16)),
        "collision_geoms_retained": link0_geom >= 0 and link1_geom >= 0,
        "combined_visual_geom_present": visual_geom >= 0,
        "combined_visual_excluded_from_inertia": bool(visual_geom >= 0 and locked.geom_group[visual_geom] == 2),
        "geometry_equivalence": geometry_equivalence(source_path, locked_path, angle_deg),
        "mass_matrix": mass_matrix_audit(locked),
    }


def build_models(angle_deg: float) -> dict[str, object]:
    ensure_directories()
    before = tip.robot_sentinels()
    tag = angle_tag(angle_deg)
    stl_path = ASSETS / f"centipede_head_link1_locked_{tag}.stl"
    stl = build_combined_stl(SOURCE_MODELS["flat"], angle_deg, stl_path)
    variants = {}
    for kind, source in SOURCE_MODELS.items():
        output = MODELS / f"centipede_{kind}_locked_head_{tag}.xml"
        variant = transform_xml(source, output, stl_path, angle_deg)
        variant["compiled_audit"] = compiled_audit(source, output, angle_deg)
        variants[kind] = variant
    result = {
        "purpose": "fixed head/link1 topology with adjustable build-time pitch; no high-stiffness surrogate",
        "default_head_pitch_deg": DEFAULT_HEAD_PITCH_DEG,
        "head_pitch_deg": angle_deg,
        "source_dynamic_target_deg": math.degrees(-0.57),
        "parameter_command": f"run_locked_head.py all --head-pitch-deg {angle_deg:g} --duration 6",
        "combined_stl": stl,
        "variants": variants,
        "robot_sentinels_before": before,
        "robot_sentinels_after": tip.robot_sentinels(),
    }
    result["robot_sentinels_unchanged"] = result["robot_sentinels_before"] == result["robot_sentinels_after"]
    path = DATA / f"locked_head_{tag}_build_manifest.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def load_manifest(angle_deg: float) -> dict[str, object]:
    path = DATA / f"locked_head_{angle_tag(angle_deg)}_build_manifest.json"
    if not path.exists():
        return build_models(angle_deg)
    return json.loads(path.read_text(encoding="utf-8"))


def baseline_rows() -> dict[str, dict[str, object]]:
    audit = json.loads((tip.DATA / "tip_demo_audit.json").read_text(encoding="utf-8"))
    rows: dict[str, dict[str, object]] = {}
    for row in audit["rows"]:
        rows.setdefault(str(row["kind"]), row)
    return rows


def dynamic_gates(row: dict[str, object], baseline: dict[str, object]) -> dict[str, bool]:
    warnings = row["warning_counts"]
    return {
        "finite": bool(row["finite"]),
        "no_mujoco_warnings": all(int(value) == 0 for value in warnings.values()),
        "head_connection_is_fixed": row["head_connection"] == "fixed",
        "armature_is_zero": int(row["armature_nonzero_count"]) == 0,
        "forward_displacement_retains_80pct": float(row["forward_displacement_m"])
        >= 0.8 * float(baseline["forward_displacement_m"]),
        "minimum_upright_cos_at_least_0p75": float(row["minimum_root_upright_cos"]) >= 0.75,
        "penetration_p95_at_most_0p10mm": float(row["contact_penetration_p95_mm"]) <= 0.10,
        "world_roll_not_over_baseline_plus_2deg": float(row["chassis_world_roll_abs_p95_deg"])
        <= float(baseline["chassis_world_roll_abs_p95_deg"]) + 2.0,
    }


def inherited_tip_gates(row: dict[str, object], reference: dict[str, object]) -> dict[str, bool]:
    """Preserve the distal-tip experiment's gates, adapting only the -2 DoF topology."""
    return {
        "finite_warning_free": bool(row["finite"]) and not bool(row["warning_counts"]),
        "topology_fixed_head_3dof_per_leg": int(row["passive_tip_roll_joint_count"]) == 38
        and int(row["nv"]) == 177
        and row["head_connection"] == "fixed",
        "armature_zero": int(row["armature_nonzero_count"]) == 0,
        "waves_coupled": bool(row["wave_numbers_coupled"]),
        "forward_retention": float(row["forward_displacement_m"])
        >= 0.80 * float(reference["forward_displacement_m"]),
        "upright": float(row["minimum_root_upright_cos"]) >= 0.75,
        "spine_tortuosity": float(row["maximum_spine_tortuosity"]) <= 1.30,
        "tip_compliance_visible": float(row["tip_roll_abs_p95_deg"]) >= 0.5,
        "tip_p95_bounded": float(row["tip_roll_abs_p95_deg"]) <= 30.0,
        "tip_stop_rare": float(row["tip_roll_stop_fraction"]) <= 0.01,
        "world_roll_not_worse": float(row["chassis_world_roll_abs_p95_deg"])
        <= float(reference["chassis_world_roll_abs_p95_deg"]) + 1.5,
        "contacting_feet": 6.0 <= float(row["mean_contacting_feet"]) <= 30.0,
        "contact_penetration": float(row["contact_penetration_p95_mm"])
        <= 1.10 * float(reference["penetration_p95_mm"]),
    }


def run_angle_sweep(angles_deg: list[float], duration_s: float) -> dict[str, object]:
    """Choose the smallest pitch reduction that preserves every inherited gate."""
    before = tip.robot_sentinels()
    references = tip.baseline_rows()
    rows = []
    progress_path = DATA / "locked_head_angle_sweep_progress.json"
    for angle_deg in angles_deg:
        manifest = load_manifest(angle_deg)
        for kind in ("flat", "rough"):
            print(f"[head-angle sweep] START {angle_deg:+.1f} deg {kind}", flush=True)
            row = tip.run_case(
                f"head pitch {angle_deg:+.1f} deg fixed",
                kind,
                Path(manifest["variants"][kind]["model"]),
                duration_s,
            )
            row["head_pitch_deg"] = angle_deg
            row["gates"] = inherited_tip_gates(row, references[kind])
            row["all_inherited_gates"] = bool(all(row["gates"].values()))
            rows.append(row)
            progress_path.write_text(
                json.dumps(
                    {
                        "duration_s": duration_s,
                        "requested_angles_deg": angles_deg,
                        "completed_rows": rows,
                        "robot_sentinels_before": before,
                        "robot_sentinels_current": tip.robot_sentinels(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(
                f"[head-angle sweep] DONE  {angle_deg:+.1f} deg {kind} "
                f"disp={1000.0 * float(row['forward_displacement_m']):.2f}mm "
                f"tipStop={100.0 * float(row['tip_roll_stop_fraction']):.3f}% "
                f"pass={row['all_inherited_gates']}",
                flush=True,
            )
    passing = []
    for angle_deg in angles_deg:
        angle_rows = [row for row in rows if math.isclose(float(row["head_pitch_deg"]), angle_deg)]
        if len(angle_rows) == 2 and all(row["all_inherited_gates"] for row in angle_rows):
            passing.append(angle_deg)
    previous = math.degrees(-0.57)
    selected = min(passing, key=lambda value: abs(value - previous)) if passing else None
    result = {
        "purpose": "head pitch sweep with inherited distal-tip gates; no scalar reward",
        "duration_s": duration_s,
        "angles_deg": angles_deg,
        "previous_dynamic_target_deg": previous,
        "selection_rule": "closest fixed angle to the previous target among candidates passing every flat/rough inherited gate",
        "passing_angles_deg": passing,
        "selected_angle_deg": selected,
        "rows": rows,
        "robot_sentinels_before": before,
        "robot_sentinels_after": tip.robot_sentinels(),
    }
    result["robot_sentinels_unchanged"] = result["robot_sentinels_before"] == result["robot_sentinels_after"]
    (DATA / "locked_head_angle_sweep.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def render_demos(angle_deg: float, duration_s: float) -> dict[str, object]:
    ensure_directories()
    manifest = load_manifest(angle_deg)
    refs = baseline_rows()
    inherited_refs = tip.baseline_rows()
    tag = angle_tag(angle_deg)
    cases = [
        ("flat", "side", 90.0, -12.0),
        ("flat", "oblique", 125.0, -35.0),
        ("flat", "top", 90.0, -89.0),
        ("rough", "side", 90.0, -12.0),
        ("rough", "oblique", 125.0, -35.0),
        ("rough", "top", 90.0, -89.0),
    ]
    before = tip.robot_sentinels()
    rows = []
    paths: list[Path] = []
    for kind, view, azimuth, elevation in cases:
        model_path = Path(manifest["variants"][kind]["model"])
        video = VIDEOS / f"locked_head_{tag}_{kind}_{view}.mp4"
        print(f"[locked-head demo] START {kind} {view}", flush=True)
        row = tip.run_case(
            f"head pitch {angle_deg:+.1f} deg fixed",
            kind,
            model_path,
            duration_s,
            video=video,
            camera_azimuth=azimuth,
            camera_elevation=elevation,
        )
        row["head_pitch_deg"] = angle_deg
        row["view"] = view
        stability_gates = dynamic_gates(row, refs[kind])
        tip_gates = inherited_tip_gates(row, inherited_refs[kind])
        row["stability_gates"] = stability_gates
        row["inherited_tip_gates"] = tip_gates
        row["gates"] = {
            **{f"stability::{name}": value for name, value in stability_gates.items()},
            **{f"inherited::{name}": value for name, value in tip_gates.items()},
        }
        row["all_gates"] = bool(all(row["gates"].values()))
        rows.append(row)
        paths.append(video)
        print(
            f"[locked-head demo] DONE  {kind} {view} "
            f"disp={1000.0 * float(row['forward_displacement_m']):.2f}mm "
            f"rollP95={float(row['chassis_world_roll_abs_p95_deg']):.2f}deg",
            flush=True,
        )
    sheet = FIGURES / f"locked_head_{tag}_contact_sheet.png"
    tip.write_contact_sheet(paths, sheet)
    result = {
        "head_pitch_deg": angle_deg,
        "duration_s": duration_s,
        "videos": [str(path) for path in paths],
        "contact_sheet": str(sheet),
        "rows": rows,
        "all_demo_rows_pass": bool(all(row["all_gates"] for row in rows)),
        "baseline_reference": refs,
        "robot_sentinels_before": before,
        "robot_sentinels_after": tip.robot_sentinels(),
    }
    result["robot_sentinels_unchanged"] = result["robot_sentinels_before"] == result["robot_sentinels_after"]
    (DATA / f"locked_head_{tag}_demo_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def write_report(angle_deg: float) -> Path:
    tag = angle_tag(angle_deg)
    build = load_manifest(angle_deg)
    demo_path = DATA / f"locked_head_{tag}_demo_audit.json"
    if not demo_path.exists():
        raise FileNotFoundError("Run the demo command before writing the report")
    demo = json.loads(demo_path.read_text(encoding="utf-8"))
    representative = {}
    for row in demo["rows"]:
        representative.setdefault(row["kind"], row)
    lines = [
        "# Fixed head/link1 audit",
        "",
        "This experiment belongs only to the biological/template **Centipede Model**. It does not modify Centipede Robot Step 18/19.",
        "",
        "## Implemented topology",
        "",
        f"- Build-time head pitch: **{angle_deg:+.1f} deg** (previous controller target {build['source_dynamic_target_deg']:+.2f} deg).",
        "- `joint_pitch_body_0` and `joint_roll_body_0` are absent; their actuators and head-pitch sensors are absent.",
        "- `link_body_1` is a fixed child of the head. There is no high-stiffness lock and no armature surrogate.",
        "- The combined STL is visual/export-only. Original link0/link1 collision meshes still determine mass and inertia.",
        f"- Regenerate at another angle with: `{build['parameter_command']}`.",
        "",
        "## Structural evidence",
        "",
    ]
    flat_audit = build["variants"]["flat"]["compiled_audit"]
    lines.extend(
        [
            f"- nq/nv/nu: {flat_audit['source_nq_nv_nu']} -> {flat_audit['locked_nq_nv_nu']} (exactly -2/-2/-2).",
            f"- Active joints: {flat_audit['active_joint_count']}; passive body-roll joints: {flat_audit['passive_body_roll_joint_count']}.",
            f"- Total mass: {flat_audit['total_mass_locked_kg']:.15g} kg; delta {flat_audit['total_mass_abs_delta_kg']:.3e} kg.",
            f"- Max per-body inertia delta: {flat_audit['body_inertia_max_abs_delta_kg_m2']:.3e} kg m^2.",
            f"- Fixed geometry equivalence max position error: {flat_audit['geometry_equivalence']['max_body_position_error_m']:.3e} m.",
            f"- Mass matrix positive definite: {flat_audit['mass_matrix']['positive_definite']}; minimum eigenvalue {flat_audit['mass_matrix']['minimum_eigenvalue']:.3e}.",
            f"- Robot sentinels unchanged: {build['robot_sentinels_unchanged'] and demo['robot_sentinels_unchanged']}.",
            "",
            "## 6 s locomotion evidence",
            "",
            "| Terrain | Forward displacement | World roll P95 | Local body roll P95 | Tip roll P95 | Tip-stop fraction | Penetration P95 | Gates |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for kind in ("flat", "rough"):
        row = representative[kind]
        lines.append(
            f"| {kind} | {1000.0 * row['forward_displacement_m']:.2f} mm | "
            f"{row['chassis_world_roll_abs_p95_deg']:.2f} deg | "
            f"{row['body_roll_local_abs_p95_deg']:.2f} deg | "
            f"{row['tip_roll_abs_p95_deg']:.2f} deg | "
            f"{100.0 * row['tip_roll_stop_fraction']:.3f}% | "
            f"{row['contact_penetration_p95_mm']:.4f} mm | "
            f"{'PASS' if row['all_gates'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            f"All six view rows pass both the locked-head stability gates and inherited distal-tip gates: **{demo['all_demo_rows_pass']}**.",
            "",
            "The STL is not substituted for collision/inertia because a visual union can change mesh preprocessing and therefore mass properties. This separation keeps the requested CAD asset while preserving evidence-traceable dynamics.",
        ]
    )
    report = ROOT / "LOCKED_HEAD_REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def compact(result: dict[str, object]) -> dict[str, object]:
    output = dict(result)
    if "rows" in output:
        output["row_count"] = len(output.pop("rows"))
    for key in ("robot_sentinels_before", "robot_sentinels_after", "baseline_reference", "variants"):
        output.pop(key, None)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "sweep", "demo", "report", "all"))
    parser.add_argument("--head-pitch-deg", type=float, default=DEFAULT_HEAD_PITCH_DEG)
    parser.add_argument("--angles", default="-31,-30,-29,-28")
    parser.add_argument("--duration", type=float, default=6.0)
    args = parser.parse_args()
    if not -60.0 <= args.head_pitch_deg <= 30.0:
        raise ValueError("head pitch must remain within the audited build range [-60, +30] deg")
    ensure_directories()
    results: dict[str, object] = {}
    if args.command in ("build", "all"):
        results["build"] = compact(build_models(args.head_pitch_deg))
    if args.command == "sweep":
        angles = [float(value) for value in args.angles.split(",")]
        if any(not -60.0 <= value <= 30.0 for value in angles):
            raise ValueError("every sweep angle must remain within [-60, +30] deg")
        results["sweep"] = compact(run_angle_sweep(angles, args.duration))
    if args.command in ("demo", "all"):
        results["demo"] = compact(render_demos(args.head_pitch_deg, args.duration))
    if args.command in ("report", "all"):
        results["report"] = str(write_report(args.head_pitch_deg))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
