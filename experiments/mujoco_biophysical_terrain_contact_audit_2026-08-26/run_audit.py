"""Biophysical and terrain-contact audit for the abstract Centipede Model.

This experiment is deliberately isolated from Centipede Robot Step 18/19.
It treats the latest zero-armature geometry-inertia plant and optimized
controller as frozen inputs, builds Model-only diagnostic variants, and
measures signed contact penetration and terrain-normal discontinuities.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import struct
import sys
from dataclasses import replace
from pathlib import Path
from xml.etree import ElementTree as ET

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[2]
CONTROLLER_EXPERIMENT = ROOT.parent / "mujoco_zero_armature_controller_optimization_2026-08-26"
PLANT_EXPERIMENT = ROOT.parent / "mujoco_zero_armature_inertia_rebuild_2026-08-25"
BASE_MODEL = PLANT_EXPERIMENT / "models" / "centipede_geometry_inertia_armature0.xml"
CONTROLLER_SCRIPT = CONTROLLER_EXPERIMENT / "controller_optimization.py"
BASELINE_MANIFEST = CONTROLLER_EXPERIMENT / "outputs" / "data" / "terrain_manifest.json"

MODELS = ROOT / "models"
OUTPUT = ROOT / "outputs"
DATA = OUTPUT / "data"
FIGURES = OUTPUT / "figures"
TERRAINS = OUTPUT / "terrains"
VIDEOS = OUTPUT / "videos"

ARENA_SIZE_M = 3.0
TERRAIN_SEED = 244949
BIOPHYSICAL_HEIGHTFIELD_PIXELS = 1536
FOOT_DIAMETER_M = 0.0009
BODY_LENGTH_M = 0.1025

TERRAIN_SPECS = {
    "l90a8": (90.0, 8.0, 2.0),
    "l60a12": (60.0, 12.0, 3.0),
}

CONTACTS = {
    "legacy": {"solref": "0.005 1.5", "solimp": "0.9 0.95 0.001"},
    "medium": {"solref": "0.003 1.2", "solimp": "0.93 0.98 0.0003"},
    "firm": {"solref": "0.002 1", "solimp": "0.95 0.99 0.0001"},
}

ROBOT_SENTINELS = {
    "step18_run_configuration": WORKSPACE / "Centipede_robot_MUJOCO" / "exp_data" / "reliability"
    / "step18_body_wave_only_random_pitch" / "run_configuration.json",
    "step18_trials": WORKSPACE / "Centipede_robot_MUJOCO" / "exp_data" / "reliability"
    / "step18_body_wave_only_random_pitch" / "trials.csv",
    "step19_run_configuration": WORKSPACE / "Centipede_robot_MUJOCO" / "exp_data" / "reliability"
    / "step19_phi_pi_over_2_body_wave_only_random_pitch" / "run_configuration.json",
    "step19_trials": WORKSPACE / "Centipede_robot_MUJOCO" / "exp_data" / "reliability"
    / "step19_phi_pi_over_2_body_wave_only_random_pitch" / "trials.csv",
}


def load_controller_module():
    spec = importlib.util.spec_from_file_location("controller_optimization_audit_source", CONTROLLER_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {CONTROLLER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ctrl = load_controller_module()
base = ctrl.base


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def robot_sentinels() -> dict[str, dict[str, object]]:
    return {
        name: {
            "path": str(path),
            "exists": path.exists(),
            "sha256": sha256(path) if path.exists() else "",
            "size_bytes": path.stat().st_size if path.exists() else 0,
        }
        for name, path in ROBOT_SENTINELS.items()
    }


def spectral_terrain(size: int, wavelength_mm: float, amplitude_mm: float, seed: int) -> np.ndarray:
    amplitude_m = float(amplitude_mm) * 1e-3
    wavelength_m = float(wavelength_mm) * 1e-3
    frequency = 1.0 / wavelength_m
    axis = np.linspace(-ARENA_SIZE_M / 2.0, ARENA_SIZE_M / 2.0, size)
    xx, yy = np.meshgrid(axis, axis)
    rng = np.random.default_rng(int(seed))
    height = np.zeros((size, size), dtype=np.float64)
    for _ in range(8):
        component_frequency = rng.uniform(0.9 * frequency, 1.1 * frequency)
        angle = rng.uniform(0.0, np.pi)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        kx = 2.0 * np.pi * component_frequency * np.cos(angle)
        ky = 2.0 * np.pi * component_frequency * np.sin(angle)
        height += np.sin(kx * xx + ky * yy + phase)
    height /= max(float(height.std()), 1e-12)
    height *= amplitude_m
    pixels_per_cycle = wavelength_m / ARENA_SIZE_M * size
    height = gaussian_filter(height, sigma=max(0.25 * pixels_per_cycle, 0.5))
    height -= height.mean()
    peak = max(abs(float(height.min())), abs(float(height.max())))
    return height * (amplitude_m / peak)


def write_mujoco_hfield(path: Path, normalized: np.ndarray) -> None:
    """Write MuJoCo's custom hfield format: int32 rows/cols + float32 samples."""

    rows, cols = normalized.shape
    payload = struct.pack("ii", rows, cols) + np.asarray(normalized, dtype=np.float32).tobytes(order="C")
    path.write_bytes(payload)


def write_smooth_visual_obj(
    path: Path,
    loaded_height: np.ndarray,
    z_max: float,
    visual_pixels: int = 257,
) -> None:
    """Write a smooth-normal visual surface; it never participates in contact."""

    source_rows, source_cols = loaded_height.shape
    row_ids = np.round(np.linspace(0, source_rows - 1, visual_pixels)).astype(int)
    col_ids = np.round(np.linspace(0, source_cols - 1, visual_pixels)).astype(int)
    height = loaded_height[np.ix_(row_ids, col_ids)] * z_max
    axis_x = np.linspace(-ARENA_SIZE_M / 2.0, ARENA_SIZE_M / 2.0, visual_pixels)
    axis_y = np.linspace(-ARENA_SIZE_M / 2.0, ARENA_SIZE_M / 2.0, visual_pixels)
    dy, dx = np.gradient(height, axis_y, axis_x)
    normals = np.stack((-dx, -dy, np.ones_like(height)), axis=-1)
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-12)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("# Smooth visual-only terrain generated from the collision hfield\n")
        for row, y in enumerate(axis_y):
            for col, x in enumerate(axis_x):
                handle.write(f"v {x:.9g} {y:.9g} {height[row, col]:.9g}\n")
        for row in range(visual_pixels):
            for col in range(visual_pixels):
                nx, ny, nz = normals[row, col]
                handle.write(f"vn {nx:.9g} {ny:.9g} {nz:.9g}\n")
        for row in range(visual_pixels - 1):
            for col in range(visual_pixels - 1):
                v00 = row * visual_pixels + col + 1
                v01 = v00 + 1
                v10 = v00 + visual_pixels
                v11 = v10 + 1
                handle.write(f"f {v00}//{v00} {v01}//{v01} {v11}//{v11}\n")
                handle.write(f"f {v00}//{v00} {v11}//{v11} {v10}//{v10}\n")


def remove_foot_spin_dofs(root: ET.Element) -> dict[str, int]:
    parent = {child: owner for owner in root.iter() for child in owner}
    removed_joints = []
    for joint in list(root.iter("joint")):
        name = joint.get("name", "")
        if name.startswith("joint_foot_"):
            parent[joint].remove(joint)
            removed_joints.append(name)
    removed_references: dict[str, int] = {}
    for element in list(root.iter()):
        if element.get("joint", "") not in removed_joints:
            continue
        parent[element].remove(element)
        removed_references[element.tag] = removed_references.get(element.tag, 0) + 1
    return {
        "removed_foot_joints": len(removed_joints),
        "removed_foot_joint_actuators": sum(
            count for tag, count in removed_references.items() if tag not in {"jointpos", "jointvel"}
        ),
        "removed_foot_joint_sensors": sum(
            count for tag, count in removed_references.items() if tag in {"jointpos", "jointvel"}
        ),
        "removed_reference_elements_by_tag": removed_references,
    }


def set_contact_parameters(root: ET.Element, contact_name: str) -> None:
    settings = CONTACTS[contact_name]
    default_geom = root.find("./default/geom")
    if default_geom is None:
        raise RuntimeError("MJCF has no default geom")
    default_geom.set("solref", settings["solref"])
    default_geom.set("solimp", settings["solimp"])
    for geom_name in ("floor", "terrain_geom"):
        geom = next((item for item in root.iter("geom") if item.get("name") == geom_name), None)
        if geom is not None:
            geom.set("solref", settings["solref"])
            geom.set("solimp", settings["solimp"])


def build_flat_variant(contact_name: str) -> tuple[Path, dict[str, object]]:
    tree = ET.parse(BASE_MODEL)
    root = tree.getroot()
    removal = remove_foot_spin_dofs(root)
    set_contact_parameters(root, contact_name)
    path = MODELS / f"centipede_flat_foot_weld_{contact_name}.xml"
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(path))
    return path, {
        **removal,
        "model": str(path),
        "sha256": sha256(path),
        "nq": int(model.nq),
        "nv": int(model.nv),
        "njnt": int(model.njnt),
        "mass_kg": float(np.sum(model.body_mass)),
        "armature_max_kg_m2": float(np.max(model.dof_armature)),
    }


def build_terrain_variant(
    terrain_name: str,
    contact_name: str,
    pixels: int = BIOPHYSICAL_HEIGHTFIELD_PIXELS,
) -> tuple[Path, dict[str, object]]:
    wavelength_mm, amplitude_mm, wave = TERRAIN_SPECS[terrain_name]
    source = spectral_terrain(pixels, wavelength_mm, amplitude_mm, TERRAIN_SEED)
    low, high = float(source.min()), float(source.max())
    peak = max(abs(low), abs(high))
    z_max = max(2.0 * peak, 0.005)
    normalized = (source - low) / max(high - low, 1e-15)
    terrain_dir = TERRAINS / f"{terrain_name}_{pixels}_float"
    terrain_dir.mkdir(parents=True, exist_ok=True)
    hfield_path = terrain_dir / "heightfield.bin"
    npy_path = terrain_dir / "heightfield_normalized.npy"
    write_mujoco_hfield(hfield_path, normalized)
    np.save(npy_path, normalized)

    tree = ET.parse(BASE_MODEL)
    root = tree.getroot()
    removal = remove_foot_spin_dofs(root)
    set_contact_parameters(root, contact_name)
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise RuntimeError("Base MJCF is missing asset/worldbody")
    ET.SubElement(
        asset,
        "hfield",
        {
            "name": "terrain",
            "file": hfield_path.as_posix(),
            "size": f"{ARENA_SIZE_M / 2:.9g} {ARENA_SIZE_M / 2:.9g} {z_max:.9g} 0.001",
        },
    )
    floor = worldbody.find("geom[@name='floor']")
    if floor is not None:
        worldbody.remove(floor)
    terrain_geom = ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "terrain_geom",
            "type": "hfield",
            "hfield": "terrain",
            "pos": "0 0 0",
            "conaffinity": "1",
            "condim": "3",
            "friction": "1.6 0.005 0.0001",
        },
    )
    terrain_geom.set("solref", CONTACTS[contact_name]["solref"])
    terrain_geom.set("solimp", CONTACTS[contact_name]["solimp"])

    free_body = next((body for body in root.iter("body") if body.find("freejoint") is not None), None)
    if free_body is None:
        raise RuntimeError("Base MJCF has no free-root body")
    radius = 0.075
    axis = np.linspace(-ARENA_SIZE_M / 2.0, ARENA_SIZE_M / 2.0, pixels)
    mask = np.logical_and(np.abs(axis)[None, :] <= radius, np.abs(axis)[:, None] <= radius)
    local_max = float(np.max(normalized[mask]) * z_max)
    base_pos = np.fromstring(free_body.get("pos", "0 0 0"), sep=" ")
    base_pos[2] += local_max
    free_body.set("pos", " ".join(f"{value:.12g}" for value in base_pos))

    path = MODELS / f"centipede_{terrain_name}_{pixels}_float_foot_weld_{contact_name}.xml"
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(path))
    actual_hfield_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain")
    if actual_hfield_id < 0:
        raise RuntimeError("Generated terrain did not load")
    start = int(model.hfield_adr[actual_hfield_id])
    loaded_height = model.hfield_data[
        start : start + int(model.hfield_nrow[actual_hfield_id] * model.hfield_ncol[actual_hfield_id])
    ].reshape(int(model.hfield_nrow[actual_hfield_id]), int(model.hfield_ncol[actual_hfield_id]))
    visual_obj = terrain_dir / "smooth_visual.obj"
    write_smooth_visual_obj(visual_obj, loaded_height, z_max)
    visual_mesh_name = f"{terrain_name}_smooth_visual"
    ET.SubElement(
        asset,
        "mesh",
        {
            "name": visual_mesh_name,
            "file": visual_obj.as_posix(),
            # Force vertex-normal interpolation even if the OBJ importer would
            # otherwise retain per-face normals.  This mesh is visual-only; the
            # float32 hfield above remains the sole collision surface.
            "smoothnormal": "true",
        },
    )
    terrain_geom.set("rgba", "0.35 0.38 0.42 0")
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "terrain_visual_geom",
            "type": "mesh",
            "mesh": visual_mesh_name,
            "contype": "0",
            "conaffinity": "0",
            "group": "2",
            "rgba": "0.42 0.45 0.50 1",
        },
    )
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(path))
    actual_hfield_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain")
    return path, {
        **removal,
        "model": str(path),
        "sha256": sha256(path),
        "terrain_name": terrain_name,
        "contact_name": contact_name,
        "wave": wave,
        "pixels": pixels,
        "cell_size_mm": 1000.0 * ARENA_SIZE_M / (pixels - 1),
        "cell_to_foot_diameter_ratio": ARENA_SIZE_M / (pixels - 1) / FOOT_DIAMETER_M,
        "height_storage": "float32 custom MuJoCo hfield",
        "visual_surface": "257x257 OBJ with shared vertex normals; contact disabled",
        "height_levels": "continuous float32",
        "z_max_m": z_max,
        "spawn_local_max_m": local_max,
        "loaded_nrow": int(model.hfield_nrow[actual_hfield_id]),
        "loaded_ncol": int(model.hfield_ncol[actual_hfield_id]),
        "nq": int(model.nq),
        "nv": int(model.nv),
        "njnt": int(model.njnt),
        "mass_kg": float(np.sum(model.body_mass)),
        "armature_max_kg_m2": float(np.max(model.dof_armature)),
    }


def build_all() -> dict[str, object]:
    for folder in (MODELS, DATA, FIGURES, TERRAINS, VIDEOS):
        folder.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "scope": "Centipede Model only; Centipede Robot Step 18/19 are read-only sentinels",
        "base_model": str(BASE_MODEL),
        "base_model_sha256": sha256(BASE_MODEL),
        "robot_sentinels_before": robot_sentinels(),
        "variants": {},
    }
    for contact_name in CONTACTS:
        flat_path, flat_row = build_flat_variant(contact_name)
        manifest["variants"][f"flat_{contact_name}"] = flat_row
    for terrain_name in TERRAIN_SPECS:
        for contact_name in CONTACTS:
            path, row = build_terrain_variant(terrain_name, contact_name)
            manifest["variants"][f"{terrain_name}_{contact_name}"] = row
    manifest["robot_sentinels_after"] = robot_sentinels()
    manifest["robot_sentinels_unchanged"] = (
        manifest["robot_sentinels_before"] == manifest["robot_sentinels_after"]
    )
    (DATA / "model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def model_mass_matrix(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_crb(model, data)
    mujoco.mj_makeM(model, data)
    dense = np.empty((model.nv, model.nv), dtype=float)
    mujoco.mj_fullM(model, dense, data.qM)
    return dense, np.linalg.eigvalsh(dense)


def plant_audit(
    path: Path,
    impedance: dict[str, tuple[float, float]] | None = None,
) -> dict[str, object]:
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    impedance = ctrl.final_impedance() if impedance is None else impedance
    index = base.ModelIndex(model, impedance=impedance)
    base.configure_native_implicit_pd(model, index, 0.0)
    dense, eigenvalues = model_mass_matrix(model)
    inverse = np.linalg.inv(dense)
    effective_inertia = 1.0 / np.diag(inverse)[index.dof_addresses]

    family_positions = {
        "body_yaw": index.body_positions,
        "head_pitch": index.pitch_positions[:1],
        "body_pitch": index.pitch_positions[1:],
        "body_roll": index.roll_positions,
        "legbase_yaw": index.leg_positions[:, :, 0].reshape(-1),
        "legbase_pitch": index.leg_positions[:, :, 1].reshape(-1),
        "legbase_roll": index.leg_positions[:, :, 2].reshape(-1),
        "knee": index.leg_positions[:, :, 3].reshape(-1),
    }
    frequencies = {}
    gains = impedance
    for family, positions in family_positions.items():
        values = effective_inertia[np.asarray(positions, dtype=int)]
        kp, kv = gains[family]
        fn = np.sqrt(kp / values) / (2.0 * np.pi)
        zeta = kv / (2.0 * np.sqrt(kp * values))
        frequencies[family] = {
            "kp_nm_rad": kp,
            "kv_nms_rad": kv,
            "effective_inertia_min_kg_m2": float(np.min(values)),
            "effective_inertia_median_kg_m2": float(np.median(values)),
            "effective_inertia_max_kg_m2": float(np.max(values)),
            "natural_frequency_min_hz": float(np.min(fn)),
            "natural_frequency_median_hz": float(np.median(fn)),
            "natural_frequency_max_hz": float(np.max(fn)),
            "damping_ratio_min": float(np.min(zeta)),
            "damping_ratio_median": float(np.median(zeta)),
            "damping_ratio_max": float(np.max(zeta)),
        }

    joint_rows = []
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
        dof = int(model.jnt_dofadr[joint_id])
        joint_rows.append(
            {
                "name": name,
                "stiffness": float(model.jnt_stiffness[joint_id]),
                "damping": float(model.dof_damping[dof]),
                "armature": float(model.dof_armature[dof]),
            }
        )
    top_damping = sorted(joint_rows, key=lambda row: row["damping"], reverse=True)[:40]

    masses = {"chassis": 0.0, "legs": 0.0, "feet": 0.0, "other": 0.0}
    triangle_failures = []
    for body_id in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        mass = float(model.body_mass[body_id])
        if name.startswith("link_body_"):
            masses["chassis"] += mass
        elif name.startswith("link_leg_"):
            masses["legs"] += mass
        elif name.startswith("foot_"):
            masses["feet"] += mass
        else:
            masses["other"] += mass
        inertia = np.sort(model.body_inertia[body_id])
        if float(inertia[2]) > float(inertia[0] + inertia[1]) + 1e-20:
            triangle_failures.append(name)

    xml_root = ET.parse(path).getroot()
    explicit_inertials = len(list(xml_root.iter("inertial")))
    collision_geoms = [geom for geom in xml_root.iter("geom") if geom.get("group", "0") == "0"]
    mass_or_inertia_overrides = [
        geom.get("name", "") for geom in collision_geoms if "mass" in geom.attrib
    ]
    foot_spin = [row for row in joint_rows if row["name"].startswith("joint_foot_")]
    chassis_positions = data.xpos[index.chassis_body_ids]
    foot_body_ids = np.asarray(
        [
            base.name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"foot_{segment}_{side}")
            for segment in range(base.N_LEG_PAIRS)
            for side in range(2)
        ],
        dtype=int,
    )
    foot_positions = data.xpos[foot_body_ids]
    return {
        "model": str(path),
        "sha256": sha256(path),
        "nbody": int(model.nbody),
        "njnt": int(model.njnt),
        "nq": int(model.nq),
        "nv": int(model.nv),
        "ngeom": int(model.ngeom),
        "total_mass_kg": float(np.sum(model.body_mass)),
        "mass_by_component_kg": masses,
        "mass_fraction_by_component": {
            name: value / float(np.sum(model.body_mass)) for name, value in masses.items()
        },
        "minimum_nonworld_body_mass_kg": float(np.min(model.body_mass[1:])),
        "maximum_nonworld_body_mass_kg": float(np.max(model.body_mass[1:])),
        "explicit_inertial_elements": explicit_inertials,
        "collision_geom_mass_overrides": mass_or_inertia_overrides,
        "inertia_triangle_failures": triangle_failures,
        "armature_max_kg_m2": float(np.max(model.dof_armature)),
        "joint_stiffness_max_nm_rad": float(np.max(model.jnt_stiffness)),
        "joint_damping_max_nms_rad": float(np.max(model.dof_damping)),
        "foot_spin_joint_count": len(foot_spin),
        "foot_spin_joint_damping_nms_rad": sorted({row["damping"] for row in foot_spin}),
        "top_joint_damping": top_damping,
        "mass_matrix_min_eigenvalue": float(eigenvalues[0]),
        "mass_matrix_max_eigenvalue": float(eigenvalues[-1]),
        "mass_matrix_condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        "mass_matrix_rank": int(np.linalg.matrix_rank(dense)),
        "mass_matrix_positive_definite": bool(eigenvalues[0] > 0.0),
        "compiled_chassis_centroid_span_m": np.ptp(chassis_positions, axis=0).tolist(),
        "compiled_foot_center_span_m": np.ptp(foot_positions, axis=0).tolist(),
        "controller_dynamics": frequencies,
    }


def derive_bandwidth_impedance(
    model_path: Path,
    body_hz: float,
    leg_hz: float,
) -> dict[str, tuple[float, float]]:
    """Derive family gains from geometry-only effective inertia.

    Body and leg bandwidths are explicit audit variables.  The leg roll target
    is capped because that axial coordinate has the smallest inertia and is not
    commanded by the gait; extra damping, not a kHz spring, handles re-entry.
    """

    target_hz = {
        "body_yaw": body_hz,
        "head_pitch": body_hz,
        "body_pitch": body_hz,
        "body_roll": body_hz,
        "legbase_yaw": leg_hz,
        "legbase_pitch": leg_hz,
        "legbase_roll": min(0.6 * leg_hz, 120.0),
        "knee": leg_hz,
    }
    target_zeta = {
        "body_yaw": 0.8,
        "head_pitch": 1.0,
        "body_pitch": 1.0,
        "body_roll": 1.0,
        "legbase_yaw": 1.0,
        "legbase_pitch": 1.0,
        "legbase_roll": 2.0,
        "knee": 1.2,
    }
    return derive_frequency_map_impedance(model_path, target_hz, target_zeta)


def derive_frequency_map_impedance(
    model_path: Path,
    target_hz: dict[str, float],
    target_zeta: dict[str, float],
) -> dict[str, tuple[float, float]]:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    provisional = base.ModelIndex(model, impedance=ctrl.final_impedance())
    dense, _ = model_mass_matrix(model)
    inverse = np.linalg.inv(dense)
    effective = 1.0 / np.diag(inverse)[provisional.dof_addresses]
    positions = {
        "body_yaw": provisional.body_positions,
        "head_pitch": provisional.pitch_positions[:1],
        "body_pitch": provisional.pitch_positions[1:],
        "body_roll": provisional.roll_positions,
        "legbase_yaw": provisional.leg_positions[:, :, 0].reshape(-1),
        "legbase_pitch": provisional.leg_positions[:, :, 1].reshape(-1),
        "legbase_roll": provisional.leg_positions[:, :, 2].reshape(-1),
        "knee": provisional.leg_positions[:, :, 3].reshape(-1),
    }
    result = {}
    for family, family_positions in positions.items():
        inertia = float(np.median(effective[np.asarray(family_positions, dtype=int)]))
        omega = 2.0 * math.pi * target_hz[family]
        kp = inertia * omega * omega
        kv = 2.0 * target_zeta[family] * inertia * omega
        result[family] = (kp, kv)
    return result


def selected_impedance() -> dict[str, tuple[float, float]]:
    path = DATA / "controller_selection.json"
    if not path.exists():
        return ctrl.final_impedance()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        family: (float(values[0]), float(values[1]))
        for family, values in payload["selected_impedance"].items()
    }


def controller_capacity(impedance: dict[str, tuple[float, float]]) -> dict[str, float]:
    """Static scale checks tied to the 7 mm leg and measured 2.536 g body."""

    lever_m = 0.007
    rough_pitch_amplitude = base.LEG_PITCH_AMPLITUDE_RAD * ctrl.ROUGH_LEG_PITCH_AMPLITUDE_SCALE
    rough_yaw_amplitude = base.LEG_YAW_AMPLITUDE_RAD * ctrl.ROUGH_LEG_YAW_AMPLITUDE_SCALE
    weight_n = 0.00253603109178 * 9.81
    pitch_force = impedance["legbase_pitch"][0] * rough_pitch_amplitude / lever_m
    yaw_force = impedance["legbase_yaw"][0] * rough_yaw_amplitude / lever_m
    return {
        "rough_leg_pitch_static_force_at_full_amplitude_mn": 1000.0 * pitch_force,
        "rough_leg_yaw_static_force_at_full_amplitude_mn": 1000.0 * yaw_force,
        "pitch_force_as_body_weight_fraction": pitch_force / weight_n,
        "yaw_force_as_body_weight_fraction": yaw_force / weight_n,
        "reference_vertical_force_per_contacting_leg_mn": 1.0,
    }


def terrain_geom_id(model: mujoco.MjModel) -> int:
    for name in ("terrain_geom", "floor"):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id >= 0:
            return int(geom_id)
    raise RuntimeError("No terrain/floor geom")


def percentile(values: list[float], q: float, default: float = float("nan")) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q)) if values else default


def warning_counts(data: mujoco.MjData) -> dict[str, int]:
    return {
        mujoco.mjtWarning(warning_id).name: int(item.number)
        for warning_id, item in enumerate(data.warning)
        if int(item.number) > 0
    }


def run_contact_diagnostic(
    label: str,
    model_path: Path,
    wave: float,
    duration_s: float = 4.5,
    eval_start_s: float = 1.5,
    video: Path | None = None,
    camera_azimuth: float = 90.0,
    camera_elevation: float = -12.0,
    camera_distance: float = 0.19,
    impedance: dict[str, tuple[float, float]] | None = None,
) -> dict[str, object]:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    impedance = ctrl.final_impedance() if impedance is None else impedance
    index = base.ModelIndex(model, impedance=impedance)
    base.configure_native_implicit_pd(model, index, 0.0)
    data = mujoco.MjData(model)
    controller = ctrl.FloatWaveController(model, index, wave)
    mujoco.mj_resetData(model, data)
    knees = index.leg_positions[:, :, 3].reshape(-1)
    data.qpos[index.qpos_addresses[knees]] = ctrl.KNEE_DC_RAD
    mujoco.mj_forward(model, data)

    ground = terrain_geom_id(model)
    foot_geoms = np.asarray(
        [
            base.name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{segment}_{side}_collision")
            for segment in range(base.N_LEG_PAIRS)
            for side in range(2)
        ],
        dtype=int,
    )
    foot_lookup = {int(geom): index for index, geom in enumerate(foot_geoms)}
    penetrations_mm: list[float] = []
    contact_forces_mn: list[float] = []
    normal_jumps_deg: list[float] = []
    contact_counts: list[int] = []
    previous_normals: dict[int, np.ndarray] = {}
    previous_contact_time: dict[int, float] = {}
    q_min = np.full(len(index.qpos_addresses), np.inf)
    q_max = np.full(len(index.qpos_addresses), -np.inf)
    peak_speed = 0.0
    peak_speed_name = ""
    peak_speed_time_s = 0.0
    peak_acceleration = 0.0
    peak_torque = 0.0
    min_upright = float("inf")
    max_tortuosity = 0.0
    start = end = previous_center = None
    path_length = 0.0
    finite = True

    renderer = writer = camera = scene_option = None
    next_frame = 0.0
    if video is not None:
        os.environ.setdefault("MUJOCO_GL", "wgl")
        video.parent.mkdir(parents=True, exist_ok=True)
        visual_terrain_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "terrain_visual_geom"
        )
        direct_hfield_visual = (
            visual_terrain_id >= 0
            and os.environ.get("CENTIPEDE_DIRECT_HFIELD_VISUAL", "1") != "0"
        )
        if direct_hfield_visual:
            # Diagnostic option: render the collision hfield itself in an
            # isolated display group, hiding the approximate visual mesh.
            # This is useful for measuring whether display downsampling can
            # masquerade as foot penetration.
            model.geom_group[ground] = 3
            model.geom_rgba[ground] = (0.42, 0.45, 0.50, 1.0)
            model.geom_rgba[visual_terrain_id, 3] = 0.0
        renderer = mujoco.Renderer(model, height=720, width=1280)
        # The terrain is an open, single-sided surface. Shadow mapping on such
        # a dense triangulation produces dark triangular acne that looks like
        # collision discontinuities, so disable shadows in audit videos.
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        writer = imageio.get_writer(video, fps=30, codec="libx264", quality=8, macro_block_size=None)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance = camera_distance
        camera.azimuth = camera_azimuth
        camera.elevation = camera_elevation
        if visual_terrain_id >= 0:
            # Terrain variants contain a high-resolution collision hfield in
            # group 0 and a smooth, non-colliding display mesh in group 2.
            # Rendering both creates false triangle seams and self-shadowing
            # that are not present in the contact geometry.
            scene_option = mujoco.MjvOption()
            scene_option.geomgroup[:] = 0
            scene_option.geomgroup[2] = 1
            if direct_hfield_visual:
                scene_option.geomgroup[3] = 1

    dt = float(model.opt.timestep)
    try:
        while data.time < duration_s - 0.5 * dt:
            torque, _ = controller.step(data)
            peak_torque = max(peak_torque, float(np.max(np.abs(torque))))
            mujoco.mj_step(model, data)
            active_q = data.qpos[index.qpos_addresses]
            q_min = np.minimum(q_min, active_q)
            q_max = np.maximum(q_max, active_q)
            active_speed = np.abs(data.qvel[index.dof_addresses])
            active_speed_position = int(np.argmax(active_speed))
            if float(active_speed[active_speed_position]) > peak_speed:
                peak_speed = float(active_speed[active_speed_position])
                peak_speed_name = index.active_joint_names[active_speed_position]
                peak_speed_time_s = float(data.time)
            peak_acceleration = max(
                peak_acceleration, float(np.max(np.abs(data.qacc[index.dof_addresses])))
            )
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                finite = False
                break

            if data.time >= eval_start_s:
                center = base.chassis_centroid(data, index)
                posture = base.posture_metrics(data, index)
                min_upright = min(min_upright, posture["root_upright_cos"])
                max_tortuosity = max(max_tortuosity, posture["spine_tortuosity"])
                if start is None:
                    start = center.copy()
                if previous_center is not None:
                    path_length += float(np.linalg.norm(center[:2] - previous_center[:2]))
                previous_center = center.copy()
                end = center.copy()

                current_normals: dict[int, list[tuple[float, np.ndarray]]] = {}
                feet_in_contact = set()
                for contact_id in range(data.ncon):
                    contact = data.contact[contact_id]
                    geom1, geom2 = int(contact.geom1), int(contact.geom2)
                    if geom1 == ground and geom2 in foot_lookup:
                        foot_position = foot_lookup[geom2]
                    elif geom2 == ground and geom1 in foot_lookup:
                        foot_position = foot_lookup[geom1]
                    else:
                        continue
                    feet_in_contact.add(foot_position)
                    penetrations_mm.append(max(0.0, -1000.0 * float(contact.dist)))
                    force = np.zeros(6, dtype=float)
                    mujoco.mj_contactForce(model, data, contact_id, force)
                    contact_forces_mn.append(1000.0 * abs(float(force[0])))
                    normal = np.asarray(contact.frame[:3], dtype=float)
                    if normal[2] < 0.0:
                        normal = -normal
                    normal /= max(float(np.linalg.norm(normal)), 1e-12)
                    current_normals.setdefault(foot_position, []).append((abs(float(force[0])), normal))
                contact_counts.append(len(feet_in_contact))
                for foot_position, force_normals in current_normals.items():
                    # Multiple triangular prisms can report contacts for one sphere.
                    # Follow the force-dominant contact so appearance/disappearance of
                    # negligible auxiliary contacts does not masquerade as a normal jump.
                    normal = max(force_normals, key=lambda item: item[0])[1]
                    if (
                        foot_position in previous_normals
                        and math.isclose(
                            float(data.time) - previous_contact_time[foot_position], dt,
                            abs_tol=0.25 * dt,
                        )
                    ):
                        dot = float(np.clip(np.dot(normal, previous_normals[foot_position]), -1.0, 1.0))
                        normal_jumps_deg.append(math.degrees(math.acos(dot)))
                    previous_normals[foot_position] = normal
                    previous_contact_time[foot_position] = float(data.time)

            if renderer is not None and writer is not None and data.time + 1e-12 >= next_frame:
                center = base.chassis_centroid(data, index)
                camera.lookat[:] = (float(center[0]), float(center[1]), 0.013)
                renderer.update_scene(data, camera=camera, scene_option=scene_option)
                canvas = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(canvas)
                draw.rectangle((16, 16, 745, 62), fill=(25, 25, 25))
                draw.text(
                    (28, 26),
                    f"{label} | armature=0 | signed contact audit | t={data.time:4.1f}s",
                    fill=(255, 255, 255),
                )
                writer.append_data(np.asarray(canvas))
                next_frame += 1.0 / 30.0
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()

    joint_range = model.jnt_range[index.active_joint_ids]
    center_range = np.mean(joint_range, axis=1)
    half_range = np.maximum(0.5 * np.ptp(joint_range, axis=1), 1e-12)
    normalized_excursion = np.maximum(abs(q_min - center_range), abs(q_max - center_range)) / half_range
    maximum_range_position = int(np.argmax(normalized_excursion))
    eval_duration = max(duration_s - eval_start_s, 1e-12)
    displacement = (
        base.FORWARD_SIGN * float(end[0] - start[0]) if start is not None and end is not None else float("nan")
    )
    jumps_nontrivial = [value for value in normal_jumps_deg if value > 1e-8]
    return {
        "label": label,
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "duration_s": duration_s,
        "eval_start_s": eval_start_s,
        "wave": wave,
        "finite": finite,
        "warning_counts": warning_counts(data),
        "forward_displacement_m": displacement,
        "forward_speed_bl_cycle": displacement / eval_duration / BODY_LENGTH_M,
        "path_efficiency": displacement / max(path_length, 1e-12),
        "minimum_root_upright_cos": min_upright,
        "maximum_spine_tortuosity": max_tortuosity,
        "peak_active_joint_speed_rad_s": peak_speed,
        "peak_active_joint_speed_name": peak_speed_name,
        "peak_active_joint_speed_time_s": peak_speed_time_s,
        "peak_active_joint_acceleration_rad_s2": peak_acceleration,
        "peak_active_torque_nm": peak_torque,
        "maximum_active_joint_range_fraction": float(np.max(normalized_excursion)),
        "maximum_active_joint_range_name": index.active_joint_names[maximum_range_position],
        "maximum_active_joint_range_observed_rad": [
            float(q_min[maximum_range_position]),
            float(q_max[maximum_range_position]),
        ],
        "maximum_active_joint_range_limit_rad": joint_range[maximum_range_position].tolist(),
        "joint_limit_contact_risk": bool(np.max(normalized_excursion) >= 0.98),
        "contact_sample_count": len(penetrations_mm),
        "penetration_median_mm": percentile(penetrations_mm, 0.5),
        "penetration_p95_mm": percentile(penetrations_mm, 0.95),
        "penetration_p99_mm": percentile(penetrations_mm, 0.99),
        "penetration_max_mm": max(penetrations_mm, default=float("nan")),
        "penetration_p95_as_foot_radius_fraction": percentile(penetrations_mm, 0.95) / 0.45,
        "contact_normal_jump_all_p95_deg_per_ms": percentile(normal_jumps_deg, 0.95),
        "contact_normal_jump_nonzero_p95_deg_per_ms": percentile(jumps_nontrivial, 0.95, 0.0),
        "contact_normal_jump_max_deg_per_ms": max(normal_jumps_deg, default=float("nan")),
        "contact_force_median_mn": percentile(contact_forces_mn, 0.5),
        "contact_force_p95_mn": percentile(contact_forces_mn, 0.95),
        "contact_force_p99_mn": percentile(contact_forces_mn, 0.99),
        "mean_contacting_feet": float(np.mean(contact_counts)) if contact_counts else float("nan"),
        "video": str(video) if video is not None else "",
    }


def locomotion_row(
    label: str,
    model_path: Path,
    wave: float,
    duration_s: float = 6.0,
    impedance: dict[str, tuple[float, float]] | None = None,
) -> dict[str, object]:
    impedance = ctrl.final_impedance() if impedance is None else impedance
    config = replace(
        base.TrialConfig(wave=1, q=0.0, seed=0),
        wave=wave,
        duration_s=duration_s,
        eval_start_s=1.5,
    )
    row = base.run_trial(
        config,
        model_path=model_path,
        armature_override_kg_m2=0.0,
        impedance=impedance,
    )
    row["label"] = label
    row["speed_bl_cycle"] = row["absolute_forward_displacement_m"] / (duration_s - 1.5) / BODY_LENGTH_M
    row["stability_gate"] = bool(
        row["finite"]
        and row["warning_counts"] == "{}"
        and row["minimum_root_upright_cos"] >= 0.259
        and row["maximum_spine_tortuosity"] <= 1.30
        and row["peak_abs_active_joint_speed_rad_s"] <= 100.0
    )
    return row


def comparison_figure(rows: list[dict[str, object]]) -> None:
    labels = [str(row["label"]) for row in rows]
    sink = np.asarray([float(row["penetration_p95_mm"]) for row in rows])
    jump = np.asarray([float(row["contact_normal_jump_nonzero_p95_deg_per_ms"]) for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    colors = ["#a24a3a" if "baseline" in label else "#3478a8" for label in labels]
    axes[0].barh(labels, sink, color=colors)
    axes[0].set_xlabel("P95 signed penetration (mm)")
    axes[0].set_title("Foot sink")
    axes[1].barh(labels, jump, color=colors)
    axes[1].set_xlabel("P95 nonzero normal jump (deg/ms)")
    axes[1].set_title("Facet/contact discontinuity")
    for axis in axes:
        axis.grid(axis="x", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIGURES / "terrain_contact_comparison.png", dpi=220)
    plt.close(fig)


def run_sweep(duration_s: float = 4.5) -> dict[str, object]:
    manifest_path = DATA / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else build_all()
    baseline = json.loads(BASELINE_MANIFEST.read_text(encoding="utf-8"))
    before = robot_sentinels()
    rows = []
    rows.append(
        run_contact_diagnostic(
            "baseline 512 PNG + foot spin + legacy contact",
            Path(baseline["l60a12"]["model"]),
            3.0,
            duration_s=duration_s,
        )
    )
    for contact_name in CONTACTS:
        row = manifest["variants"][f"l60a12_{contact_name}"]
        rows.append(
            run_contact_diagnostic(
                f"1536 float + welded foot + {contact_name} contact",
                Path(row["model"]),
                3.0,
                duration_s=duration_s,
            )
        )
    comparison_figure(rows)
    result = {
        "duration_s": duration_s,
        "contact_rows": rows,
        "selection_rule": (
            "Select the least compliant contact that remains finite, warning-free, under the frozen "
            "100 rad/s speed gate and 1.30 tortuosity gate; report sink and normal-jump metrics separately."
        ),
        "robot_sentinels_before": before,
        "robot_sentinels_after": robot_sentinels(),
    }
    result["robot_sentinels_unchanged"] = (
        result["robot_sentinels_before"] == result["robot_sentinels_after"]
    )
    (DATA / "contact_sweep.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_controller_sweep(duration_s: float = 3.5) -> dict[str, object]:
    manifest_path = DATA / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else build_all()
    before = robot_sentinels()
    candidates = [
        ("body90_yaw400_pitch300_roll350z10_kneez20", 90.0, 400.0, 300.0, 350.0, 10.0, 20.0),
        ("body100_yaw450_pitch325_roll350z10_kneez20", 100.0, 450.0, 325.0, 350.0, 10.0, 20.0),
        ("body100_yaw450_pitch325_roll400z10_kneez20", 100.0, 450.0, 325.0, 400.0, 10.0, 20.0),
        ("body100_yaw450_pitch325_roll400z20_kneez20", 100.0, 450.0, 325.0, 400.0, 20.0, 20.0),
        ("body100_yaw500_pitch325_roll400z10_kneez20", 100.0, 500.0, 325.0, 400.0, 10.0, 20.0),
    ]
    model_path = Path(manifest["variants"]["l60a12_medium"]["model"])
    flat_reference = Path(manifest["variants"]["flat_medium"]["model"])
    rows = []
    impedance_by_label = {}
    for stem, body_hz, yaw_hz, pitch_hz, roll_hz, roll_zeta, knee_zeta in candidates:
        label = f"{stem}_medium"
        target_hz = {
            "body_yaw": body_hz,
            "head_pitch": min(body_hz, 65.0),
            "body_pitch": body_hz,
            "body_roll": min(1.1 * body_hz, 110.0),
            "legbase_yaw": yaw_hz,
            "legbase_pitch": pitch_hz,
            "legbase_roll": roll_hz,
            "knee": pitch_hz,
        }
        target_zeta = {
            "body_yaw": 1.5,
            "head_pitch": 1.0,
            "body_pitch": 1.5,
            "body_roll": 1.5,
            "legbase_yaw": 1.5,
            "legbase_pitch": 1.5,
            "legbase_roll": roll_zeta,
            "knee": knee_zeta,
        }
        impedance = derive_frequency_map_impedance(flat_reference, target_hz, target_zeta)
        impedance_by_label[label] = impedance
        row = run_contact_diagnostic(
            label,
            model_path,
            3.0,
            duration_s=duration_s,
            impedance=impedance,
        )
        row["body_bandwidth_hz"] = body_hz
        row["leg_yaw_bandwidth_hz"] = yaw_hz
        row["leg_pitch_knee_bandwidth_hz"] = pitch_hz
        row["leg_roll_bandwidth_hz"] = roll_hz
        row["leg_roll_damping_ratio"] = roll_zeta
        row["knee_damping_ratio"] = knee_zeta
        row["target_frequency_hz"] = target_hz
        row["target_damping_ratio"] = target_zeta
        row["capacity"] = controller_capacity(impedance)
        row["biophysical_gate"] = bool(
            row["finite"]
            and row["warning_counts"] == {}
            and row["minimum_root_upright_cos"] >= 0.259
            and row["maximum_spine_tortuosity"] <= 1.30
            and row["peak_active_joint_speed_rad_s"] <= 100.0
            and not row["joint_limit_contact_risk"]
            and row["penetration_p95_mm"] <= 0.45
            and row["contact_force_p95_mn"] <= 5.0
            and row["forward_speed_bl_cycle"] >= 0.03
        )
        rows.append(row)

    feasible = [row for row in rows if row["biophysical_gate"]]
    if feasible:
        selected = max(feasible, key=lambda row: float(row["forward_speed_bl_cycle"]))
        selection_status = "all biophysical and stability gates passed"
    else:
        def penalty(row: dict[str, object]) -> float:
            return (
                max(0.0, float(row["maximum_spine_tortuosity"]) - 1.30) * 10.0
                + max(0.0, float(row["peak_active_joint_speed_rad_s"]) / 100.0 - 1.0)
                + max(0.0, float(row["maximum_active_joint_range_fraction"]) - 0.98) * 5.0
                + max(0.0, float(row["penetration_p95_mm"]) / 0.45 - 1.0)
                + max(0.0, float(row["contact_force_p95_mn"]) / 5.0 - 1.0)
                + max(0.0, 0.03 - float(row["forward_speed_bl_cycle"])) * 10.0
            )
        selected = min(rows, key=lambda row: (penalty(row), -float(row["forward_speed_bl_cycle"])))
        selection_status = "least-violation candidate; final validation must not call this fully biophysical"

    selected_label = str(selected["label"])
    result = {
        "controller_before": ctrl.final_impedance(),
        "controller_before_dynamics": plant_audit(BASE_MODEL)["controller_dynamics"],
        "contact_variant": "medium",
        "candidates": rows,
        "selected_label": selected_label,
        "selected_status": selection_status,
        "selected_impedance": impedance_by_label[selected_label],
        "selected_capacity": controller_capacity(impedance_by_label[selected_label]),
        "robot_sentinels_before": before,
        "robot_sentinels_after": robot_sentinels(),
    }
    result["robot_sentinels_unchanged"] = (
        result["robot_sentinels_before"] == result["robot_sentinels_after"]
    )
    (DATA / "controller_selection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_selected_contact_sweep(duration_s: float = 4.5) -> dict[str, object]:
    manifest = json.loads((DATA / "model_manifest.json").read_text(encoding="utf-8"))
    selection_path = DATA / "controller_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    impedance = selected_impedance()
    rows = []
    for contact_name in CONTACTS:
        row = run_contact_diagnostic(
            f"selected controller + {contact_name} contact",
            Path(manifest["variants"][f"l60a12_{contact_name}"]["model"]),
            3.0,
            duration_s=duration_s,
            impedance=impedance,
        )
        row["contact_name"] = contact_name
        row["contact_gate"] = bool(
            row["finite"]
            and row["warning_counts"] == {}
            and row["minimum_root_upright_cos"] >= 0.259
            and row["maximum_spine_tortuosity"] <= 1.30
            and row["penetration_p95_mm"] <= 0.45
            and row["contact_force_p95_mn"] <= 5.0
        )
        rows.append(row)
    feasible = [row for row in rows if row["contact_gate"]]
    if feasible:
        chosen = min(
            feasible,
            key=lambda row: (
                float(row["penetration_p95_mm"]),
                float(row["contact_normal_jump_nonzero_p95_deg_per_ms"]),
            ),
        )
    else:
        chosen = min(rows, key=lambda row: float(row["penetration_p95_mm"]))
    selection["contact_variant"] = chosen["contact_name"]
    selection["selected_contact_evidence"] = chosen
    selection_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    result = {
        "rows": rows,
        "selected_contact": chosen["contact_name"],
        "selection_rule": "minimum P95 sink among finite, stable, <=5 mN P95 force candidates",
        "robot_sentinels_unchanged": robot_sentinels() == selection["robot_sentinels_after"],
    }
    (DATA / "selected_contact_sweep.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_final(duration_s: float = 9.6, render: bool = False) -> dict[str, object]:
    manifest_path = DATA / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else build_all()
    baseline = json.loads(BASELINE_MANIFEST.read_text(encoding="utf-8"))
    before = robot_sentinels()
    selection_path = DATA / "controller_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.exists() else {}
    contact_name = str(selection.get("contact_variant", "medium"))
    impedance = selected_impedance()
    cases = {
        "flat_final": (Path(manifest["variants"][f"flat_{contact_name}"]["model"]), 1.5),
        "l90a8_final": (Path(manifest["variants"][f"l90a8_{contact_name}"]["model"]), 2.0),
        "l60a12_final": (Path(manifest["variants"][f"l60a12_{contact_name}"]["model"]), 3.0),
    }
    contact_rows = []
    locomotion_rows = []
    for name, (path, wave) in cases.items():
        video = VIDEOS / f"{name}_side.mp4" if render else None
        contact_rows.append(
            run_contact_diagnostic(
                name,
                path,
                wave,
                duration_s=duration_s,
                video=video,
                impedance=impedance,
            )
        )
        locomotion_rows.append(
            locomotion_row(name, path, wave, duration_s=duration_s, impedance=impedance)
        )
    if render:
        run_contact_diagnostic(
            "l60a12 baseline 512 PNG",
            Path(baseline["l60a12"]["model"]),
            3.0,
            duration_s=duration_s,
            video=VIDEOS / "l60a12_baseline_512png_side.mp4",
        )
        run_contact_diagnostic(
            "l60a12 final 1536 float",
            cases["l60a12_final"][0],
            3.0,
            duration_s=duration_s,
            video=VIDEOS / "l60a12_final_1536float_oblique.mp4",
            camera_azimuth=125.0,
            camera_elevation=-35.0,
            camera_distance=0.19,
            impedance=impedance,
        )
    result = {
        "contact_rows": contact_rows,
        "locomotion_rows": locomotion_rows,
        "selected_impedance": impedance,
        "selected_contact": contact_name,
        "selected_controller_dynamics": plant_audit(cases["flat_final"][0], impedance)[
            "controller_dynamics"
        ],
        "selected_capacity": controller_capacity(impedance),
        "robot_sentinels_before": before,
        "robot_sentinels_after": robot_sentinels(),
    }
    result["robot_sentinels_unchanged"] = (
        result["robot_sentinels_before"] == result["robot_sentinels_after"]
    )
    (DATA / "final_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def render_smooth_demos(duration_s: float = 6.0) -> dict[str, object]:
    manifest = json.loads((DATA / "model_manifest.json").read_text(encoding="utf-8"))
    selection = json.loads((DATA / "controller_selection.json").read_text(encoding="utf-8"))
    contact_name = str(selection["contact_variant"])
    impedance = selected_impedance()
    l60 = Path(manifest["variants"][f"l60a12_{contact_name}"]["model"])
    l90 = Path(manifest["variants"][f"l90a8_{contact_name}"]["model"])
    before = robot_sentinels()
    rows = [
        run_contact_diagnostic(
            "l60a12 final exact collision surface side",
            l60,
            3.0,
            duration_s=duration_s,
            video=VIDEOS / "l60a12_final_exact_hfield_side.mp4",
            impedance=impedance,
        ),
        run_contact_diagnostic(
            "l60a12 final exact collision surface oblique",
            l60,
            3.0,
            duration_s=duration_s,
            video=VIDEOS / "l60a12_final_exact_hfield_oblique.mp4",
            camera_azimuth=125.0,
            camera_elevation=-35.0,
            camera_distance=0.19,
            impedance=impedance,
        ),
        run_contact_diagnostic(
            "l90a8 final exact collision surface side",
            l90,
            2.0,
            duration_s=duration_s,
            video=VIDEOS / "l90a8_final_exact_hfield_side.mp4",
            impedance=impedance,
        ),
    ]
    result = {
        "rows": rows,
        "collision_contract": "1536x1536 float32 hfield",
        "visual_contract": (
            "render the exact 1536x1536 collision hfield in isolated geom group 3; "
            "hide collision-body group 0 and disable shadow mapping"
        ),
        "robot_sentinels_before": before,
        "robot_sentinels_after": robot_sentinels(),
    }
    result["robot_sentinels_unchanged"] = (
        result["robot_sentinels_before"] == result["robot_sentinels_after"]
    )
    (DATA / "smooth_demo_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_plant_audit() -> dict[str, object]:
    manifest_path = DATA / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else build_all()
    before = robot_sentinels()
    corrected_path = Path(
        manifest["variants"].get("flat_medium", manifest["variants"]["flat_firm"])["model"]
    )
    result = {
        "base": plant_audit(BASE_MODEL),
        "corrected": plant_audit(corrected_path, selected_impedance()),
        "robot_sentinels_before": before,
        "robot_sentinels_after": robot_sentinels(),
    }
    result["robot_sentinels_unchanged"] = (
        result["robot_sentinels_before"] == result["robot_sentinels_after"]
    )
    (DATA / "plant_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "build",
            "plant",
            "sweep",
            "controller",
            "selected_contact",
            "final",
            "demo",
            "all",
        ),
        nargs="?",
        default="all",
    )
    parser.add_argument("--duration", type=float, default=4.5)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    output = {}
    if args.command in {"build", "all"}:
        output["build"] = build_all()
    if args.command in {"plant", "all"}:
        output["plant"] = run_plant_audit()
    if args.command in {"sweep", "all"}:
        output["sweep"] = run_sweep(duration_s=args.duration)
    if args.command in {"controller", "all"}:
        output["controller"] = run_controller_sweep(duration_s=args.duration)
    if args.command in {"selected_contact", "all"}:
        output["selected_contact"] = run_selected_contact_sweep(duration_s=args.duration)
    if args.command == "final":
        output["final"] = run_final(duration_s=args.duration, render=args.render)
    if args.command == "demo":
        output["demo"] = render_smooth_demos(duration_s=args.duration)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
