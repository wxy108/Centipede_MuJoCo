"""Stage-1 fidelity calibration for the abstract Centipede Model.

This experiment is isolated from Centipede Robot Step 18/19 and never writes
to their directories.  It freezes the latest Model controller, calibrates
small-scale foot contact, rebuilds rough terrain at foot-resolving spatial
resolution, and tests bounded effective leg-plane roll (joint family _2).

The script deliberately separates:
  * numerical/contact calibration (single-foot rig),
  * plant diagnostics (bounded-roll ablation), and
  * visual demonstrations (only after the diagnostic sweep).
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
from dataclasses import dataclass
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
EXPERIMENTS = ROOT.parent
PREVIOUS_AUDIT = EXPERIMENTS / "mujoco_biophysical_terrain_contact_audit_2026-08-26"
PREVIOUS_SCRIPT = PREVIOUS_AUDIT / "run_audit.py"
PREVIOUS_SELECTION = PREVIOUS_AUDIT / "outputs" / "data" / "controller_selection.json"

MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
DATA = OUTPUTS / "data"
FIGURES = OUTPUTS / "figures"
VIDEOS = OUTPUTS / "videos"
TERRAINS = OUTPUTS / "terrains"
RIGS = OUTPUTS / "single_foot_rigs"

ARENA_SIZE_M = 0.30
TERRAIN_PIXELS = 1536
TERRAIN_SEED = 244949
FOOT_RADIUS_M = 0.00045
FOOT_DIAMETER_M = 2.0 * FOOT_RADIUS_M
BODY_LENGTH_M = 0.1025
SUPPORTED_MASS_KG = 0.00253603109178 / 20.0
ROUGH_SPEC = {"name": "l60a12", "wavelength_mm": 60.0, "amplitude_mm": 12.0, "wave": 3.0}
Q2_LIMITS_DEG = (180.0, 30.0, 20.0, 10.0)
PASSIVE_Q2 = {
    # Round-2 explicit physical plant brackets.  The round-1 2.5--10 uN m/rad
    # sweep is preserved in passive_q2_sweep.json and showed that external
    # foot moments overwhelmed the restoring moment.  These sensitivity
    # points remain orders of magnitude below the old hidden high-gain servo;
    # they are not claimed biological constants.
    "restoring_k20_c12": (2.0e-5, 1.2e-7),
    "restoring_k30_c18": (3.0e-5, 1.8e-7),
    "restoring_k40_c24": (4.0e-5, 2.4e-7),
    "damped_k30_c30": (3.0e-5, 3.0e-7),
}

PASSIVE_SWEEP_PATH = DATA / "passive_q2_sweep_round2.json"
BODY_AMPLITUDE_SCALES = (0.45, 0.55, 0.65, 0.75)


@dataclass(frozen=True)
class ContactProfile:
    name: str
    timestep_s: float
    solref: str
    solimp: str


CONTACT_PROFILES = {
    "medium_dt1ms": ContactProfile("medium_dt1ms", 0.001, "0.003 1.2", "0.93 0.98 0.0003"),
    "medium_dt0p5ms": ContactProfile(
        "medium_dt0p5ms", 0.0005, "0.003 1.2", "0.93 0.98 0.0003"
    ),
    "scaled_dt0p5ms": ContactProfile(
        "scaled_dt0p5ms", 0.0005, "0.002 1.5", "0.95 0.99 0.0001"
    ),
    "scaled_dt0p25ms": ContactProfile(
        "scaled_dt0p25ms", 0.00025, "0.002 1.5", "0.95 0.99 0.0001"
    ),
    "stiff_dt0p5ms": ContactProfile(
        "stiff_dt0p5ms", 0.0005, "0.001 1.5", "0.95 0.99 0.0001"
    ),
    "stiff_dt0p25ms": ContactProfile(
        "stiff_dt0p25ms", 0.00025, "0.0005 1.5", "0.95 0.99 0.0001"
    ),
}


ROBOT_SENTINELS = {
    "step18_run_configuration": ROOT.parents[2]
    / "Centipede_robot_MUJOCO"
    / "exp_data"
    / "reliability"
    / "step18_body_wave_only_random_pitch"
    / "run_configuration.json",
    "step18_trials": ROOT.parents[2]
    / "Centipede_robot_MUJOCO"
    / "exp_data"
    / "reliability"
    / "step18_body_wave_only_random_pitch"
    / "trials.csv",
    "step19_run_configuration": ROOT.parents[2]
    / "Centipede_robot_MUJOCO"
    / "exp_data"
    / "reliability"
    / "step19_phi_pi_over_2_body_wave_only_random_pitch"
    / "run_configuration.json",
    "step19_trials": ROOT.parents[2]
    / "Centipede_robot_MUJOCO"
    / "exp_data"
    / "reliability"
    / "step19_phi_pi_over_2_body_wave_only_random_pitch"
    / "trials.csv",
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


audit = load_module(PREVIOUS_SCRIPT, "centipede_stage1_previous_audit")
ctrl = audit.ctrl
base = audit.base
BASE_MODEL = audit.BASE_MODEL


class Stage1FloatWaveController(ctrl.FloatWaveController):
    """Frozen controller with one explicit body-wave amplitude multiplier."""

    def __init__(
        self,
        model: mujoco.MjModel,
        index,
        wave: float,
        body_amplitude_scale: float,
        hold_first_body_yaw: bool,
    ):
        super().__init__(model, index, wave)
        self.body_amplitude_scale = float(body_amplitude_scale)
        self.hold_first_body_yaw = bool(hold_first_body_yaw)

    def step(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        # The inherited controller reads the amplitude constant at each step.
        # Temporarily scale only that target-generator term, then restore it;
        # plant stiffness, gains, leg motion, phase and frequency remain frozen.
        original = float(base.BODY_AMPLITUDE_RAD)
        base.BODY_AMPLITUDE_RAD = original * self.body_amplitude_scale
        try:
            torque, qdot = super().step(data)
        finally:
            base.BODY_AMPLITUDE_RAD = original
        if self.hold_first_body_yaw:
            position = int(self.index.body_positions[0])
            actuator_id = int(self.index.active_actuator_ids[position])
            q = float(data.qpos[self.index.qpos_addresses[position]])
            qvel = float(data.qvel[self.index.dof_addresses[position]])
            # This target is zero from the first step, so its target velocity
            # is also exactly zero.  The first leg pair is indexed separately
            # and retains normal yaw/pitch motion.
            self.target[position] = 0.0
            if self.previous_target is not None:
                self.previous_target[position] = 0.0
            data.ctrl[actuator_id] = 0.0
            torque[position] = -self.index.kp[position] * q - self.index.kv[position] * qvel
        return torque, qdot


def ensure_directories() -> None:
    for path in (MODELS, DATA, FIGURES, VIDEOS, TERRAINS, RIGS):
        path.mkdir(parents=True, exist_ok=True)


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


def warning_counts(data: mujoco.MjData) -> dict[str, int]:
    return {
        mujoco.mjtWarning(index).name: int(item.number)
        for index, item in enumerate(data.warning)
        if int(item.number) > 0
    }


def percentile(values: list[float], quantile: float, default: float = float("nan")) -> float:
    if not values:
        return default
    return float(np.quantile(np.asarray(values, dtype=float), quantile))


def selected_impedance() -> dict[str, tuple[float, float]]:
    raw = json.loads(PREVIOUS_SELECTION.read_text(encoding="utf-8"))["selected_impedance"]
    return {name: (float(values[0]), float(values[1])) for name, values in raw.items()}


def set_contact(root: ET.Element, profile: ContactProfile) -> None:
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", f"{profile.timestep_s:.9g}")
    option.set("integrator", "implicitfast")
    option.set("cone", "elliptic")

    default_geom = root.find("./default/geom")
    if default_geom is None:
        raise RuntimeError("Base model lacks ./default/geom")
    default_geom.set("solref", profile.solref)
    default_geom.set("solimp", profile.solimp)
    for geom in root.iter("geom"):
        if geom.get("name") in {"floor", "terrain_geom"}:
            geom.set("solref", profile.solref)
            geom.set("solimp", profile.solimp)


def set_q2_limit(root: ET.Element, limit_deg: float, profile: ContactProfile) -> int:
    limit_rad = math.radians(float(limit_deg))
    count = 0
    for joint in root.iter("joint"):
        name = joint.get("name", "")
        if not name.startswith("joint_leg_") or not name.endswith("_2"):
            continue
        joint.set("limited", "true")
        joint.set("range", f"{-limit_rad:.12g} {limit_rad:.12g}")
        # The anatomical stop uses the same explicitly calibrated solver scale
        # as foot contact.  This prevents the old 3 ms soft constraint from
        # permitting 2--3x nominal range before reacting.
        joint.set("solreflimit", profile.solref)
        joint.set("solimplimit", profile.solimp)
        count += 1
    if count != 38:
        raise RuntimeError(f"Expected 38 q2 roll joints, found {count}")
    return count


def set_q2_passive(root: ET.Element, stiffness: float, damping: float) -> int:
    count = 0
    for joint in root.iter("joint"):
        name = joint.get("name", "")
        if not name.startswith("joint_leg_") or not name.endswith("_2"):
            continue
        joint.set("stiffness", f"{stiffness:.12g}")
        joint.set("damping", f"{damping:.12g}")
        joint.set("springref", "0")
        count += 1
    if count != 38:
        raise RuntimeError(f"Expected 38 passive q2 roll joints, found {count}")
    return count


def remove_foot_spin_dofs(root: ET.Element) -> dict[str, int]:
    return audit.remove_foot_spin_dofs(root)


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


def write_hfield(path: Path, normalized: np.ndarray) -> None:
    rows, cols = normalized.shape
    path.write_bytes(
        struct.pack("ii", rows, cols)
        + np.asarray(normalized, dtype=np.float32).tobytes(order="C")
    )


def write_visual_obj(path: Path, loaded_height: np.ndarray, z_max: float, pixels: int = 257) -> None:
    rows, cols = loaded_height.shape
    row_ids = np.round(np.linspace(0, rows - 1, pixels)).astype(int)
    col_ids = np.round(np.linspace(0, cols - 1, pixels)).astype(int)
    height = loaded_height[np.ix_(row_ids, col_ids)] * z_max
    axis_x = np.linspace(-ARENA_SIZE_M / 2.0, ARENA_SIZE_M / 2.0, pixels)
    axis_y = np.linspace(-ARENA_SIZE_M / 2.0, ARENA_SIZE_M / 2.0, pixels)
    dy, dx = np.gradient(height, axis_y, axis_x)
    normals = np.stack((-dx, -dy, np.ones_like(height)), axis=-1)
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-12)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("# Stage-1 visual-only surface; the float32 hfield is the collision surface.\n")
        for row, y in enumerate(axis_y):
            for col, x in enumerate(axis_x):
                handle.write(f"v {x:.9g} {y:.9g} {height[row, col]:.9g}\n")
        for row in range(pixels):
            for col in range(pixels):
                nx, ny, nz = normals[row, col]
                handle.write(f"vn {nx:.9g} {ny:.9g} {nz:.9g}\n")
        for row in range(pixels - 1):
            for col in range(pixels - 1):
                v00 = row * pixels + col + 1
                v01 = v00 + 1
                v10 = v00 + pixels
                v11 = v10 + 1
                handle.write(f"f {v00}//{v00} {v01}//{v01} {v11}//{v11}\n")
                handle.write(f"f {v00}//{v00} {v11}//{v11} {v10}//{v10}\n")


def terrain_assets() -> dict[str, object]:
    terrain_dir = TERRAINS / f"{ROUGH_SPEC['name']}_{TERRAIN_PIXELS}_local300mm"
    terrain_dir.mkdir(parents=True, exist_ok=True)
    hfield_path = terrain_dir / "heightfield.bin"
    npy_path = terrain_dir / "heightfield_normalized.npy"
    visual_path = terrain_dir / "smooth_visual.obj"

    source = spectral_terrain(
        TERRAIN_PIXELS,
        float(ROUGH_SPEC["wavelength_mm"]),
        float(ROUGH_SPEC["amplitude_mm"]),
        TERRAIN_SEED,
    )
    low, high = float(source.min()), float(source.max())
    peak = max(abs(low), abs(high))
    z_max = max(2.0 * peak, 0.005)
    normalized = (source - low) / max(high - low, 1e-15)
    write_hfield(hfield_path, normalized)
    np.save(npy_path, normalized)
    write_visual_obj(visual_path, normalized, z_max)
    return {
        "directory": str(terrain_dir),
        "hfield": str(hfield_path),
        "normalized_npy": str(npy_path),
        "visual_obj": str(visual_path),
        "z_max_m": z_max,
        "source_low_m": low,
        "source_high_m": high,
        "cell_size_mm": 1000.0 * ARENA_SIZE_M / (TERRAIN_PIXELS - 1),
        "cells_per_foot_diameter": FOOT_DIAMETER_M / (ARENA_SIZE_M / (TERRAIN_PIXELS - 1)),
    }


def build_variant(
    kind: str,
    profile: ContactProfile,
    q2_limit_deg: float,
    passive_name: str | None = None,
) -> tuple[Path, dict[str, object]]:
    tree = ET.parse(BASE_MODEL)
    root = tree.getroot()
    removal = remove_foot_spin_dofs(root)
    set_contact(root, profile)
    q2_count = set_q2_limit(root, q2_limit_deg, profile)
    q2_stiffness = 0.0
    q2_damping = 0.0
    if passive_name is not None:
        q2_stiffness, q2_damping = PASSIVE_Q2[passive_name]
        set_q2_passive(root, q2_stiffness, q2_damping)

    terrain_meta: dict[str, object] = {}
    if kind == "rough":
        terrain_meta = terrain_assets()
        asset = root.find("asset")
        worldbody = root.find("worldbody")
        if asset is None or worldbody is None:
            raise RuntimeError("Base model lacks asset/worldbody")
        ET.SubElement(
            asset,
            "hfield",
            {
                "name": "terrain",
                "file": Path(str(terrain_meta["hfield"])).as_posix(),
                "size": (
                    f"{ARENA_SIZE_M / 2:.9g} {ARENA_SIZE_M / 2:.9g} "
                    f"{float(terrain_meta['z_max_m']):.9g} 0.001"
                ),
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
                "solref": profile.solref,
                "solimp": profile.solimp,
                "rgba": "0.35 0.38 0.42 0",
            },
        )
        visual_mesh_name = "stage1_smooth_visual"
        ET.SubElement(
            asset,
            "mesh",
            {
                "name": visual_mesh_name,
                "file": Path(str(terrain_meta["visual_obj"])).as_posix(),
                "smoothnormal": "true",
            },
        )
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

        normalized = np.load(str(terrain_meta["normalized_npy"]))
        z_max = float(terrain_meta["z_max_m"])
        axis = np.linspace(-ARENA_SIZE_M / 2.0, ARENA_SIZE_M / 2.0, TERRAIN_PIXELS)
        x_mask = np.logical_and(axis >= -0.015, axis <= 0.120)
        y_mask = np.logical_and(axis >= -0.030, axis <= 0.030)
        footprint = normalized[np.ix_(y_mask, x_mask)]
        local_max = float(np.max(footprint) * z_max)
        free_body = next((body for body in root.iter("body") if body.find("freejoint") is not None), None)
        if free_body is None:
            raise RuntimeError("Base model has no free-root body")
        base_pos = np.fromstring(free_body.get("pos", "0 0 0"), sep=" ")
        base_pos[2] += local_max
        free_body.set("pos", " ".join(f"{value:.12g}" for value in base_pos))
        terrain_meta["spawn_footprint_max_m"] = local_max
    elif kind != "flat":
        raise ValueError(f"Unknown variant kind {kind}")

    suffix = str(q2_limit_deg).replace(".", "p")
    passive_suffix = f"_{passive_name}" if passive_name is not None else ""
    path = MODELS / f"centipede_{kind}_{profile.name}_q2pm{suffix}deg{passive_suffix}.xml"
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(path))
    meta = {
        "kind": kind,
        "profile": profile.name,
        "q2_limit_deg": q2_limit_deg,
        "q2_joint_count": q2_count,
        "q2_mode": "passive" if passive_name is not None else "active_pd_reference",
        "q2_passive_name": passive_name or "",
        "q2_passive_stiffness_nm_rad": q2_stiffness,
        "q2_passive_damping_nms_rad": q2_damping,
        "model": str(path),
        "sha256": sha256(path),
        "timestep_s": float(model.opt.timestep),
        "nq": int(model.nq),
        "nv": int(model.nv),
        "njnt": int(model.njnt),
        "total_mass_kg": float(np.sum(model.body_mass)),
        "armature_max_kg_m2": float(np.max(model.dof_armature)),
        **removal,
        "terrain": terrain_meta,
    }
    return path, meta


def single_foot_xml(profile: ContactProfile, drop_height_m: float) -> str:
    start_z = FOOT_RADIUS_M + float(drop_height_m)
    return f"""<mujoco model=\"single_foot_stage1\">
  <option timestep=\"{profile.timestep_s:.9g}\" gravity=\"0 0 -9.81\" integrator=\"implicitfast\" cone=\"elliptic\"/>
  <size nconmax=\"32\" njmax=\"128\"/>
  <default>
    <joint damping=\"0\" armature=\"0\" frictionloss=\"0\"/>
    <geom condim=\"3\" friction=\"1.6 0.005 0.0001\" solref=\"{profile.solref}\" solimp=\"{profile.solimp}\"/>
  </default>
  <worldbody>
    <geom name=\"floor\" type=\"plane\" size=\"0.02 0.02 0.001\"/>
    <body name=\"supported_foot\" pos=\"0 0 {start_z:.12g}\">
      <joint name=\"vertical_slide\" type=\"slide\" axis=\"0 0 1\"/>
      <geom name=\"foot\" type=\"sphere\" size=\"{FOOT_RADIUS_M:.12g}\" mass=\"{SUPPORTED_MASS_KG:.12g}\"/>
    </body>
  </worldbody>
</mujoco>
"""


def run_single_foot_case(profile: ContactProfile, drop_height_m: float) -> dict[str, object]:
    rig_path = RIGS / f"foot_{profile.name}_drop{1000.0 * drop_height_m:.2f}mm.xml"
    rig_path.write_text(single_foot_xml(profile, drop_height_m), encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(rig_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "foot")
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    penetrations: list[float] = []
    settle_penetrations: list[float] = []
    forces: list[float] = []
    settle_speeds: list[float] = []
    peak_vertical_speed = 0.0
    first_contact_s: float | None = None
    finite = True
    duration_s = 0.60
    settle_start_s = 0.40
    dt = float(model.opt.timestep)
    while data.time < duration_s - 0.5 * dt:
        mujoco.mj_step(model, data)
        peak_vertical_speed = max(peak_vertical_speed, abs(float(data.qvel[0])))
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            finite = False
            break
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            pair = {int(contact.geom1), int(contact.geom2)}
            if pair != {foot_id, floor_id}:
                continue
            if first_contact_s is None:
                first_contact_s = float(data.time)
            penetration = max(0.0, -1000.0 * float(contact.dist))
            penetrations.append(penetration)
            force = np.zeros(6, dtype=float)
            mujoco.mj_contactForce(model, data, contact_id, force)
            forces.append(1000.0 * abs(float(force[0])))
            if data.time >= settle_start_s:
                settle_penetrations.append(penetration)
        if data.time >= settle_start_s:
            settle_speeds.append(abs(float(data.qvel[0])))

    expected_force_mn = 1000.0 * SUPPORTED_MASS_KG * 9.81
    return {
        "profile": profile.name,
        "timestep_s": profile.timestep_s,
        "solref": profile.solref,
        "solimp": profile.solimp,
        "drop_height_mm": 1000.0 * drop_height_m,
        "supported_mass_kg": SUPPORTED_MASS_KG,
        "expected_static_force_mn": expected_force_mn,
        "finite": finite,
        "warning_counts": warning_counts(data),
        "first_contact_s": first_contact_s,
        "penetration_p95_mm": percentile(penetrations, 0.95),
        "penetration_max_mm": max(penetrations, default=float("nan")),
        "settle_penetration_median_mm": percentile(settle_penetrations, 0.5),
        "settle_penetration_p95_mm": percentile(settle_penetrations, 0.95),
        "settle_vertical_speed_p95_m_s": percentile(settle_speeds, 0.95),
        "contact_force_settle_median_mn": percentile(
            forces[-max(1, int(0.15 / dt)) :], 0.5
        ),
        "peak_vertical_speed_m_s": peak_vertical_speed,
        "rig": str(rig_path),
        "rig_sha256": sha256(rig_path),
    }


def calibrate_contacts() -> dict[str, object]:
    before = robot_sentinels()
    rows = []
    for profile in CONTACT_PROFILES.values():
        for drop_height_m in (0.0, 0.0005, 0.0010):
            rows.append(run_single_foot_case(profile, drop_height_m))
    for row in rows:
        row["contact_gate"] = bool(
            row["finite"]
            and not row["warning_counts"]
            and float(row["settle_penetration_p95_mm"]) <= 0.09
            and float(row["penetration_max_mm"]) <= 0.225
            and float(row["settle_vertical_speed_p95_m_s"]) <= 0.002
        )
    result = {
        "purpose": "single-foot numerical/contact calibration; not a locomotion objective",
        "foot_radius_mm": 1000.0 * FOOT_RADIUS_M,
        "provisional_gates": {
            "settle_penetration_p95_mm_max": 0.09,
            "impact_penetration_max_mm_max": 0.225,
            "settle_vertical_speed_p95_m_s_max": 0.002,
        },
        "rows": rows,
        "robot_sentinels_before": before,
        "robot_sentinels_after": robot_sentinels(),
    }
    result["robot_sentinels_unchanged"] = (
        result["robot_sentinels_before"] == result["robot_sentinels_after"]
    )
    path = DATA / "single_foot_contact_calibration.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def ground_geom_id(model: mujoco.MjModel) -> int:
    for name in ("terrain_geom", "floor"):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id >= 0:
            return int(geom_id)
    raise RuntimeError("No ground geom")


def configure_native_implicit_pd(
    model: mujoco.MjModel,
    index,
    q2_positions: np.ndarray | None = None,
    q2_mode: str = "active_pd",
) -> None:
    """Stage-local copy of the validated native PD port.

    The predecessor intentionally hard-coded a 1 ms integration step.  This
    stage is a timestep/contact convergence experiment, so it retains every
    plant and actuator check while accepting the timestep compiled into the
    isolated variant.
    """

    actuator_ids = index.active_actuator_ids
    if len(np.unique(actuator_ids)) != len(actuator_ids):
        raise RuntimeError("Active actuator mapping is not one-to-one")
    if not np.all(
        model.actuator_trntype[actuator_ids] == int(mujoco.mjtTrn.mjTRN_JOINT)
    ):
        raise RuntimeError("Implicit-PD port requires direct joint transmissions")
    if not np.allclose(model.actuator_gear[actuator_ids, 0], 1.0):
        raise RuntimeError("Implicit-PD port requires unit actuator gear")
    if not np.allclose(model.actuator_gear[actuator_ids, 1:], 0.0):
        raise RuntimeError("Implicit-PD port found an unexpected multi-axis gear")
    if np.any(index.kp <= 0.0) or np.any(index.kv < 0.0):
        raise RuntimeError("Implicit-PD gains must satisfy kp>0 and kv>=0")
    if not np.allclose(
        model.dof_armature[index.dof_addresses], 0.0, atol=1.0e-16, rtol=1.0e-9
    ):
        raise RuntimeError("Stage-1 variants must remain exactly armature=0")
    passive_mask = np.zeros(len(index.active_joint_ids), dtype=bool)
    if q2_positions is not None:
        passive_mask[np.asarray(q2_positions, dtype=int)] = True
    if q2_mode == "active_pd":
        if not np.allclose(model.dof_damping[index.dof_addresses], 0.0):
            raise RuntimeError("Active-PD reference still contains passive damping")
        if not np.allclose(model.jnt_stiffness[index.active_joint_ids], 0.0):
            raise RuntimeError("Active-PD reference still contains passive stiffness")
    elif q2_mode == "passive":
        if q2_positions is None:
            raise RuntimeError("Passive q2 mode requires q2 positions")
        if not np.allclose(model.dof_damping[index.dof_addresses][~passive_mask], 0.0):
            raise RuntimeError("Non-q2 joints contain hidden passive damping")
        if not np.allclose(model.jnt_stiffness[index.active_joint_ids][~passive_mask], 0.0):
            raise RuntimeError("Non-q2 joints contain hidden passive stiffness")
        if np.any(model.dof_damping[index.dof_addresses][passive_mask] <= 0.0):
            raise RuntimeError("Passive q2 damping is not explicit and positive")
        if np.any(model.jnt_stiffness[index.active_joint_ids][passive_mask] <= 0.0):
            raise RuntimeError("Passive q2 stiffness is not explicit and positive")
    else:
        raise ValueError(f"Unknown q2 mode {q2_mode}")

    model.opt.integrator = int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
    model.actuator_gaintype[actuator_ids] = int(mujoco.mjtGain.mjGAIN_FIXED)
    model.actuator_biastype[actuator_ids] = int(mujoco.mjtBias.mjBIAS_AFFINE)
    model.actuator_gainprm[actuator_ids] = 0.0
    model.actuator_biasprm[actuator_ids] = 0.0
    model.actuator_gainprm[actuator_ids, 0] = index.kp
    model.actuator_biasprm[actuator_ids, 1] = -index.kp
    model.actuator_biasprm[actuator_ids, 2] = -index.kv
    if q2_mode == "passive":
        q2_actuators = actuator_ids[np.asarray(q2_positions, dtype=int)]
        model.actuator_gainprm[q2_actuators] = 0.0
        model.actuator_biasprm[q2_actuators] = 0.0


def run_diagnostic(
    label: str,
    model_path: Path,
    wave: float,
    q2_limit_deg: float,
    duration_s: float = 4.5,
    eval_start_s: float = 1.5,
    video: Path | None = None,
    camera_azimuth: float = 90.0,
    camera_elevation: float = -12.0,
    camera_distance: float = 0.19,
    q2_mode: str = "active_pd",
    q2_passive_name: str = "",
    body_amplitude_scale: float = 1.0,
    hold_first_body_yaw: bool = False,
) -> dict[str, object]:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    impedance = selected_impedance()
    index = base.ModelIndex(model, impedance=impedance)
    q2_positions = index.leg_positions[:, :, 2].reshape(-1)
    q3_positions = index.leg_positions[:, :, 3].reshape(-1)
    configure_native_implicit_pd(model, index, q2_positions=q2_positions, q2_mode=q2_mode)
    data = mujoco.MjData(model)
    controller = Stage1FloatWaveController(
        model, index, wave, body_amplitude_scale, hold_first_body_yaw
    )
    mujoco.mj_resetData(model, data)
    data.qpos[index.qpos_addresses[q3_positions]] = ctrl.KNEE_DC_RAD
    mujoco.mj_forward(model, data)

    ground = ground_geom_id(model)
    foot_geoms = np.asarray(
        [
            base.name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{segment}_{side}_collision")
            for segment in range(base.N_LEG_PAIRS)
            for side in range(2)
        ],
        dtype=int,
    )
    foot_lookup = {int(geom): position for position, geom in enumerate(foot_geoms)}
    penetrations_mm: list[float] = []
    contact_forces_mn: list[float] = []
    normal_jumps_deg: list[float] = []
    normal_rates_deg_ms: list[float] = []
    contact_counts: list[int] = []
    q2_abs_deg: list[float] = []
    q2_speed_abs: list[float] = []
    q2_passive_torque_abs: list[float] = []
    q3_deviation_abs_deg: list[float] = []
    previous_normals: dict[int, np.ndarray] = {}
    previous_contact_time: dict[int, float] = {}
    q_min = np.full(len(index.qpos_addresses), np.inf)
    q_max = np.full(len(index.qpos_addresses), -np.inf)
    peak_speed = 0.0
    peak_speed_name = ""
    peak_speed_time_s = 0.0
    peak_acceleration = 0.0
    peak_torque = 0.0
    q2_peak_angle_deg = 0.0
    q2_peak_angle_name = ""
    q2_peak_angle_time_s = 0.0
    min_upright = float("inf")
    max_tortuosity = 0.0
    start = end = previous_center = None
    path_length = 0.0
    finite = True

    renderer = writer = camera = scene_option = None
    next_frame = 0.0
    live_contact_count = 0
    if video is not None:
        os.environ.setdefault("MUJOCO_GL", "wgl")
        video.parent.mkdir(parents=True, exist_ok=True)
        visual_terrain_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "terrain_visual_geom"
        )
        model.geom_group[ground] = 3
        model.geom_rgba[ground] = (0.42, 0.45, 0.50, 1.0)
        if visual_terrain_id >= 0:
            model.geom_rgba[visual_terrain_id, 3] = 0.0
        renderer = mujoco.Renderer(model, height=720, width=1280)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        writer = imageio.get_writer(
            video, fps=30, codec="libx264", quality=8, macro_block_size=None
        )
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance = camera_distance
        camera.azimuth = camera_azimuth
        camera.elevation = camera_elevation
        scene_option = mujoco.MjvOption()
        scene_option.geomgroup[:] = 0
        scene_option.geomgroup[2] = 1
        scene_option.geomgroup[3] = 1

    dt = float(model.opt.timestep)
    try:
        while data.time < duration_s - 0.5 * dt:
            torque, _ = controller.step(data)
            if q2_mode == "passive":
                torque[np.asarray(q2_positions, dtype=int)] = 0.0
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

                q2 = data.qpos[index.qpos_addresses[q2_positions]]
                q2_speed = data.qvel[index.dof_addresses[q2_positions]]
                q3 = data.qpos[index.qpos_addresses[q3_positions]]
                q2_abs_deg.extend(np.degrees(np.abs(q2)).tolist())
                q2_speed_abs.extend(np.abs(q2_speed).tolist())
                q2_abs_now_deg = np.degrees(np.abs(q2))
                q2_peak_position = int(np.argmax(q2_abs_now_deg))
                if float(q2_abs_now_deg[q2_peak_position]) > q2_peak_angle_deg:
                    q2_peak_angle_deg = float(q2_abs_now_deg[q2_peak_position])
                    q2_peak_angle_name = index.active_joint_names[
                        int(q2_positions[q2_peak_position])
                    ]
                    q2_peak_angle_time_s = float(data.time)
                if q2_mode == "passive":
                    q2_joint_ids = np.asarray(index.active_joint_ids, dtype=int)[q2_positions]
                    q2_dof_ids = index.dof_addresses[q2_positions]
                    passive_torque = (
                        -model.jnt_stiffness[q2_joint_ids] * q2
                        - model.dof_damping[q2_dof_ids] * q2_speed
                    )
                    q2_passive_torque_abs.extend(np.abs(passive_torque).tolist())
                q3_deviation_abs_deg.extend(
                    np.degrees(np.abs(q3 - ctrl.KNEE_DC_RAD)).tolist()
                )

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
                    magnitude = abs(float(force[0]))
                    contact_forces_mn.append(1000.0 * magnitude)
                    # MuJoCo exposes contact.frame as mutable internal storage.
                    # Copy it before the next mj_step; retaining the view makes
                    # even a flat plane appear to have changing normals.
                    normal = np.asarray(contact.frame[:3], dtype=float).copy()
                    if normal[2] < 0.0:
                        normal = -normal
                    normal /= max(float(np.linalg.norm(normal)), 1e-12)
                    current_normals.setdefault(foot_position, []).append((magnitude, normal))
                live_contact_count = len(feet_in_contact)
                contact_counts.append(live_contact_count)
                for foot_position, force_normals in current_normals.items():
                    normal = max(force_normals, key=lambda item: item[0])[1]
                    if foot_position in previous_normals:
                        elapsed = float(data.time) - previous_contact_time[foot_position]
                        if math.isclose(elapsed, dt, abs_tol=0.25 * dt):
                            dot = float(
                                np.clip(np.dot(normal, previous_normals[foot_position]), -1.0, 1.0)
                            )
                            jump_deg = math.degrees(math.acos(dot))
                            normal_jumps_deg.append(jump_deg)
                            normal_rates_deg_ms.append(jump_deg / max(1000.0 * elapsed, 1e-12))
                    previous_normals[foot_position] = normal
                    previous_contact_time[foot_position] = float(data.time)

            if renderer is not None and writer is not None and data.time + 1e-12 >= next_frame:
                center = base.chassis_centroid(data, index)
                camera.lookat[:] = (float(center[0]), float(center[1]), 0.013)
                renderer.update_scene(data, camera=camera, scene_option=scene_option)
                canvas = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(canvas)
                q2_now = data.qpos[index.qpos_addresses[q2_positions]]
                draw.rectangle((16, 16, 890, 82), fill=(25, 25, 25))
                draw.text(
                    (28, 25),
                    f"{label} | body={body_amplitude_scale:.2f}x | "
                    f"head1={'hold' if hold_first_body_yaw else 'feedback'} | "
                    f"q2=+/-{q2_limit_deg:g} deg | armature=0 | t={data.time:4.1f}s",
                    fill=(255, 255, 255),
                )
                draw.text(
                    (28, 51),
                    f"max |q2|={np.degrees(np.max(np.abs(q2_now))):5.1f} deg | feet contact={live_contact_count:2d}",
                    fill=(255, 230, 120),
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
        base.FORWARD_SIGN * float(end[0] - start[0])
        if start is not None and end is not None
        else float("nan")
    )
    q2_limit_excess = percentile(q2_abs_deg, 1.0) / max(q2_limit_deg, 1e-12)
    return {
        "label": label,
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "duration_s": duration_s,
        "eval_start_s": eval_start_s,
        "wave": wave,
        "timestep_s": dt,
        "q2_limit_deg": q2_limit_deg,
        "q2_mode": q2_mode,
        "q2_passive_name": q2_passive_name,
        "body_amplitude_scale": float(body_amplitude_scale),
        "body_amplitude_rad": float(base.BODY_AMPLITUDE_RAD) * float(body_amplitude_scale),
        "hold_first_body_yaw": bool(hold_first_body_yaw),
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
        "q2_abs_rms_deg": float(np.sqrt(np.mean(np.square(q2_abs_deg)))) if q2_abs_deg else float("nan"),
        "q2_abs_p95_deg": percentile(q2_abs_deg, 0.95),
        "q2_abs_p99_deg": percentile(q2_abs_deg, 0.99),
        "q2_abs_max_deg": percentile(q2_abs_deg, 1.0),
        "q2_above_nominal_limit_fraction": float(
            np.mean(np.asarray(q2_abs_deg, dtype=float) > q2_limit_deg)
        )
        if q2_abs_deg
        else float("nan"),
        "q2_peak_angle_name": q2_peak_angle_name,
        "q2_peak_angle_time_s": q2_peak_angle_time_s,
        "q2_limit_excess_ratio": q2_limit_excess,
        "q2_speed_p95_rad_s": percentile(q2_speed_abs, 0.95),
        "q2_speed_max_rad_s": percentile(q2_speed_abs, 1.0),
        "q2_passive_torque_p95_nm": percentile(q2_passive_torque_abs, 0.95, 0.0),
        "q2_passive_torque_max_nm": percentile(q2_passive_torque_abs, 1.0, 0.0),
        "q3_deviation_p95_deg": percentile(q3_deviation_abs_deg, 0.95),
        "q3_deviation_max_deg": percentile(q3_deviation_abs_deg, 1.0),
        "contact_sample_count": len(penetrations_mm),
        "penetration_median_mm": percentile(penetrations_mm, 0.5),
        "penetration_p95_mm": percentile(penetrations_mm, 0.95),
        "penetration_p99_mm": percentile(penetrations_mm, 0.99),
        "penetration_max_mm": max(penetrations_mm, default=float("nan")),
        "penetration_p95_as_foot_radius_fraction": percentile(penetrations_mm, 0.95)
        / (1000.0 * FOOT_RADIUS_M),
        "contact_normal_rate_nonzero_p95_deg_ms": percentile(
            [value for value in normal_rates_deg_ms if value > 1e-8], 0.95, 0.0
        ),
        "contact_normal_rate_max_deg_ms": max(normal_rates_deg_ms, default=float("nan")),
        "contact_normal_jump_nonzero_p95_deg": percentile(
            [value for value in normal_jumps_deg if value > 1e-8], 0.95, 0.0
        ),
        "contact_normal_jump_max_deg": max(normal_jumps_deg, default=float("nan")),
        "contact_force_median_mn": percentile(contact_forces_mn, 0.5),
        "contact_force_p95_mn": percentile(contact_forces_mn, 0.95),
        "contact_force_p99_mn": percentile(contact_forces_mn, 0.99),
        "mean_contacting_feet": float(np.mean(contact_counts)) if contact_counts else float("nan"),
        "video": str(video) if video is not None else "",
    }


def build_all(profile: ContactProfile) -> dict[str, object]:
    before = robot_sentinels()
    variants = {}
    for limit in Q2_LIMITS_DEG:
        for kind in ("flat", "rough"):
            path, meta = build_variant(kind, profile, limit)
            variants[f"{kind}_q2pm{limit:g}"] = meta
    result = {
        "source_model": str(BASE_MODEL),
        "source_model_sha256": sha256(BASE_MODEL),
        "source_controller_selection": str(PREVIOUS_SELECTION),
        "source_controller_selection_sha256": sha256(PREVIOUS_SELECTION),
        "profile": profile.name,
        "arena_size_m": ARENA_SIZE_M,
        "terrain_pixels": TERRAIN_PIXELS,
        "cell_size_mm": 1000.0 * ARENA_SIZE_M / (TERRAIN_PIXELS - 1),
        "cells_per_foot_diameter": FOOT_DIAMETER_M / (ARENA_SIZE_M / (TERRAIN_PIXELS - 1)),
        "variants": variants,
        "robot_sentinels_before": before,
        "robot_sentinels_after": robot_sentinels(),
    }
    result["robot_sentinels_unchanged"] = (
        result["robot_sentinels_before"] == result["robot_sentinels_after"]
    )
    (DATA / "model_manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def build_passive_all(profile: ContactProfile, q2_limit_deg: float = 30.0) -> dict[str, object]:
    before = robot_sentinels()
    variants = {}
    for passive_name in PASSIVE_Q2:
        for kind in ("flat", "rough"):
            path, meta = build_variant(
                kind,
                profile,
                q2_limit_deg,
                passive_name=passive_name,
            )
            variants[f"{kind}_{passive_name}"] = meta
    result = {
        "purpose": (
            "explicit passive q2 plant variants; q2 active PD is disabled at runtime "
            "while all other controller parameters remain frozen"
        ),
        "source_model": str(BASE_MODEL),
        "source_model_sha256": sha256(BASE_MODEL),
        "source_controller_selection": str(PREVIOUS_SELECTION),
        "source_controller_selection_sha256": sha256(PREVIOUS_SELECTION),
        "profile": profile.name,
        "q2_limit_deg": q2_limit_deg,
        "passive_brackets": {
            name: {"stiffness_nm_rad": values[0], "damping_nms_rad": values[1]}
            for name, values in PASSIVE_Q2.items()
        },
        "arena_size_m": ARENA_SIZE_M,
        "terrain_pixels": TERRAIN_PIXELS,
        "cell_size_mm": 1000.0 * ARENA_SIZE_M / (TERRAIN_PIXELS - 1),
        "cells_per_foot_diameter": FOOT_DIAMETER_M / (ARENA_SIZE_M / (TERRAIN_PIXELS - 1)),
        "variants": variants,
        "robot_sentinels_before": before,
        "robot_sentinels_after": robot_sentinels(),
    }
    result["robot_sentinels_unchanged"] = (
        result["robot_sentinels_before"] == result["robot_sentinels_after"]
    )
    (DATA / "passive_model_manifest.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def diagnostic_gates(row: dict[str, object]) -> dict[str, bool]:
    return {
        "finite_warning_free": bool(row["finite"] and not row["warning_counts"]),
        "forward": bool(float(row["forward_displacement_m"]) > 0.0),
        "upright": bool(float(row["minimum_root_upright_cos"]) >= 0.75),
        "spine_tortuosity": bool(float(row["maximum_spine_tortuosity"]) <= 1.30),
        "q2_within_stop": bool(float(row["q2_abs_max_deg"]) <= 35.0),
        "q2_stop_occupancy": bool(
            float(row["q2_above_nominal_limit_fraction"]) <= 0.0025
        ),
        "q2_p95_below_stop": bool(float(row["q2_abs_p95_deg"]) <= 27.0),
        "q2_speed": bool(float(row["q2_speed_max_rad_s"]) <= 100.0),
        "penetration": bool(float(row["penetration_p95_mm"]) <= 0.18),
        "normal_jump": bool(float(row["contact_normal_jump_nonzero_p95_deg"]) <= 15.0),
    }


def comparison_figure(rows: list[dict[str, object]]) -> None:
    labels = [f"{row['label']}" for row in rows]
    q2_speed = [float(row["q2_speed_max_rad_s"]) for row in rows]
    penetration = [float(row["penetration_p95_mm"]) for row in rows]
    normal_jump = [float(row["contact_normal_jump_nonzero_p95_deg"]) for row in rows]
    speed = [float(row["forward_speed_bl_cycle"]) for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    axes[0, 0].bar(x, q2_speed)
    axes[0, 0].axhline(100.0, color="crimson", linestyle="--")
    axes[0, 0].set_ylabel("max |q2 velocity| [rad/s]")
    axes[0, 1].bar(x, penetration)
    axes[0, 1].axhline(0.18, color="crimson", linestyle="--")
    axes[0, 1].set_ylabel("foot penetration p95 [mm]")
    axes[1, 0].bar(x, normal_jump)
    axes[1, 0].axhline(15.0, color="crimson", linestyle="--")
    axes[1, 0].set_ylabel("contact normal step jump p95 [deg]")
    axes[1, 1].bar(x, speed)
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set_ylabel("forward speed [BL/cycle]")
    for axis in axes.ravel():
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Stage-1 bounded-roll diagnostics (fixed controller; no scalar winner)")
    fig.savefig(FIGURES / "bounded_roll_diagnostics.png", dpi=220)
    plt.close(fig)


def run_sweep(profile: ContactProfile, duration_s: float) -> dict[str, object]:
    manifest_path = DATA / "model_manifest.json"
    if not manifest_path.exists():
        build_all(profile)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["profile"] != profile.name:
        manifest = build_all(profile)
    before = robot_sentinels()
    rows = []
    for limit in Q2_LIMITS_DEG:
        for kind, wave in (("flat", 1.5), ("rough", float(ROUGH_SPEC["wave"]))):
            key = f"{kind}_q2pm{limit:g}"
            model_path = Path(manifest["variants"][key]["model"])
            row = run_diagnostic(
                f"{kind} q2 +/-{limit:g}deg",
                model_path,
                wave,
                limit,
                duration_s=duration_s,
            )
            row["kind"] = kind
            row["gates"] = diagnostic_gates(row)
            row["gate_pass_count"] = int(sum(row["gates"].values()))
            row["all_diagnostic_gates"] = bool(all(row["gates"].values()))
            rows.append(row)

    by_limit = {}
    for limit in Q2_LIMITS_DEG:
        pair = [row for row in rows if math.isclose(float(row["q2_limit_deg"]), limit)]
        by_limit[f"{limit:g}"] = {
            "all_conditions_pass": bool(all(row["all_diagnostic_gates"] for row in pair)),
            "total_gate_passes": int(sum(int(row["gate_pass_count"]) for row in pair)),
            "total_gate_count": int(sum(len(row["gates"]) for row in pair)),
            "max_q2_speed_rad_s": max(float(row["q2_speed_max_rad_s"]) for row in pair),
            "max_penetration_p95_mm": max(float(row["penetration_p95_mm"]) for row in pair),
            "max_normal_rate_p95_deg_ms": max(
                float(row["contact_normal_rate_nonzero_p95_deg_ms"]) for row in pair
            ),
            "minimum_forward_displacement_m": min(
                float(row["forward_displacement_m"]) for row in pair
            ),
        }

    fully_passing = [float(key) for key, value in by_limit.items() if value["all_conditions_pass"]]
    if fully_passing:
        demo_limit = min(fully_passing)
        demo_status = "passes all provisional stage-1 diagnostic gates"
    else:
        ranked = sorted(
            ((float(key), value) for key, value in by_limit.items()),
            key=lambda item: (
                -int(item[1]["total_gate_passes"]),
                float(item[1]["max_q2_speed_rad_s"]),
                float(item[1]["max_penetration_p95_mm"]),
                -float(item[1]["minimum_forward_displacement_m"]),
            ),
        )
        demo_limit = ranked[0][0]
        demo_status = "diagnostic demo only; no q2 limit passed every provisional gate"

    result = {
        "purpose": "fixed-controller plant/contact sensitivity; not controller optimization",
        "profile": profile.name,
        "duration_s": duration_s,
        "provisional_gates": {
            "minimum_root_upright_cos": 0.75,
            "maximum_spine_tortuosity": 1.30,
            "q2_transient_max_deg": 35.0,
            "q2_above_nominal_limit_fraction_max": 0.0025,
            "q2_abs_p95_deg_max": 27.0,
            "q2_speed_max_rad_s": 100.0,
            "penetration_p95_mm_max": 0.18,
            "contact_normal_jump_nonzero_p95_deg_max": 15.0,
            "positive_forward_displacement": True,
        },
        "rows": rows,
        "by_limit": by_limit,
        "demo_q2_limit_deg": demo_limit,
        "demo_status": demo_status,
        "robot_sentinels_before": before,
        "robot_sentinels_after": robot_sentinels(),
    }
    result["robot_sentinels_unchanged"] = (
        result["robot_sentinels_before"] == result["robot_sentinels_after"]
    )
    (DATA / "bounded_roll_sweep.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    comparison_figure(rows)
    return result


def run_passive_sweep(profile: ContactProfile, duration_s: float) -> dict[str, object]:
    manifest = build_passive_all(profile)
    before = robot_sentinels()
    rows = []
    for passive_name in PASSIVE_Q2:
        for kind, wave in (("flat", 1.5), ("rough", float(ROUGH_SPEC["wave"]))):
            model_path = Path(manifest["variants"][f"{kind}_{passive_name}"]["model"])
            row = run_diagnostic(
                f"{kind} passive q2 {passive_name}",
                model_path,
                wave,
                float(manifest["q2_limit_deg"]),
                duration_s=duration_s,
                q2_mode="passive",
                q2_passive_name=passive_name,
            )
            row["kind"] = kind
            row["gates"] = diagnostic_gates(row)
            row["gate_pass_count"] = int(sum(row["gates"].values()))
            row["all_diagnostic_gates"] = bool(all(row["gates"].values()))
            rows.append(row)

    by_passive = {}
    for passive_name in PASSIVE_Q2:
        pair = [row for row in rows if row["q2_passive_name"] == passive_name]
        by_passive[passive_name] = {
            "all_conditions_pass": bool(all(row["all_diagnostic_gates"] for row in pair)),
            "total_gate_passes": int(sum(int(row["gate_pass_count"]) for row in pair)),
            "total_gate_count": int(sum(len(row["gates"]) for row in pair)),
            "max_q2_angle_deg": max(float(row["q2_abs_max_deg"]) for row in pair),
            "max_q2_p95_deg": max(float(row["q2_abs_p95_deg"]) for row in pair),
            "max_q2_speed_rad_s": max(float(row["q2_speed_max_rad_s"]) for row in pair),
            "max_penetration_p95_mm": max(float(row["penetration_p95_mm"]) for row in pair),
            "max_normal_rate_p95_deg_ms": max(
                float(row["contact_normal_rate_nonzero_p95_deg_ms"]) for row in pair
            ),
            "minimum_forward_displacement_m": min(
                float(row["forward_displacement_m"]) for row in pair
            ),
        }

    fully_passing = [
        name for name in PASSIVE_Q2 if by_passive[name]["all_conditions_pass"]
    ]
    if fully_passing:
        # Prefer the least stiff passing physical bracket; this is not a scalar
        # locomotion reward or a claim of a biological optimum.
        demo_name = fully_passing[0]
        demo_status = "passes all provisional stage-1 plant/contact gates"
    else:
        ranked = sorted(
            PASSIVE_Q2,
            key=lambda name: (
                -int(by_passive[name]["total_gate_passes"]),
                float(by_passive[name]["max_q2_speed_rad_s"]),
                float(by_passive[name]["max_penetration_p95_mm"]),
                -float(by_passive[name]["minimum_forward_displacement_m"]),
            ),
        )
        demo_name = ranked[0]
        demo_status = "diagnostic demo only; no passive bracket passed every provisional gate"

    result = {
        "purpose": (
            "explicit passive q2 plant sweep; active q2 servo disabled; "
            "all non-q2 controller parameters frozen"
        ),
        "profile": profile.name,
        "duration_s": duration_s,
        "q2_limit_deg": float(manifest["q2_limit_deg"]),
        "provisional_gates": {
            "minimum_root_upright_cos": 0.75,
            "maximum_spine_tortuosity": 1.30,
            "q2_transient_max_deg": 35.0,
            "q2_above_nominal_limit_fraction_max": 0.0025,
            "q2_abs_p95_deg_max": 27.0,
            "q2_speed_max_rad_s": 100.0,
            "penetration_p95_mm_max": 0.18,
            "contact_normal_jump_nonzero_p95_deg_max": 15.0,
            "positive_forward_displacement": True,
        },
        "rows": rows,
        "by_passive": by_passive,
        "demo_passive_name": demo_name,
        "demo_status": demo_status,
        "normal_metric_correction": (
            "contact.frame is copied before mj_step; predecessor metrics retained mutable views "
            "and are downgraded"
        ),
        "robot_sentinels_before": before,
        "robot_sentinels_after": robot_sentinels(),
    }
    result["robot_sentinels_unchanged"] = (
        result["robot_sentinels_before"] == result["robot_sentinels_after"]
    )
    PASSIVE_SWEEP_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    comparison_figure(rows)
    (FIGURES / "bounded_roll_diagnostics.png").replace(
        FIGURES / "passive_q2_diagnostics_round2.png"
    )
    return result


def run_passive_probe(
    profile: ContactProfile, duration_s: float, passive_names: tuple[str, ...]
) -> dict[str, object]:
    """Run a bounded timestep/constraint convergence check on selected plants."""

    manifest = build_passive_all(profile)
    before = robot_sentinels()
    rows = []
    for passive_name in passive_names:
        for kind, wave in (("flat", 1.5), ("rough", float(ROUGH_SPEC["wave"]))):
            model_path = Path(manifest["variants"][f"{kind}_{passive_name}"]["model"])
            row = run_diagnostic(
                f"{kind} convergence probe {passive_name}",
                model_path,
                wave,
                float(manifest["q2_limit_deg"]),
                duration_s=duration_s,
                q2_mode="passive",
                q2_passive_name=passive_name,
            )
            row["kind"] = kind
            row["gates"] = diagnostic_gates(row)
            row["gate_pass_count"] = int(sum(row["gates"].values()))
            row["all_diagnostic_gates"] = bool(all(row["gates"].values()))
            rows.append(row)
    result = {
        "purpose": (
            "timestep and joint-stop convergence probe after the round-2 fixed-controller sweep"
        ),
        "profile": profile.name,
        "duration_s": duration_s,
        "passive_names": list(passive_names),
        "q2_limit_deg": float(manifest["q2_limit_deg"]),
        "rows": rows,
        "all_rows_pass": bool(all(row["all_diagnostic_gates"] for row in rows)),
        "robot_sentinels_before": before,
        "robot_sentinels_after": robot_sentinels(),
    }
    result["robot_sentinels_unchanged"] = (
        result["robot_sentinels_before"] == result["robot_sentinels_after"]
    )
    suffix = "_".join(passive_names)
    path = DATA / f"passive_probe_{profile.name}_{suffix}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_body_wave_sweep(
    profile: ContactProfile,
    duration_s: float,
    passive_name: str = "restoring_k30_c18",
) -> dict[str, object]:
    """Reduce body-wave curvature with the validated plant held fixed."""

    manifest = build_passive_all(profile)
    before = robot_sentinels()
    rows = []
    for body_scale in BODY_AMPLITUDE_SCALES:
        for kind, wave in (("flat", 1.5), ("rough", float(ROUGH_SPEC["wave"]))):
            model_path = Path(manifest["variants"][f"{kind}_{passive_name}"]["model"])
            row = run_diagnostic(
                f"{kind} body-wave {body_scale:.2f}x",
                model_path,
                wave,
                float(manifest["q2_limit_deg"]),
                duration_s=duration_s,
                q2_mode="passive",
                q2_passive_name=passive_name,
                body_amplitude_scale=body_scale,
            )
            row["kind"] = kind
            row["gates"] = diagnostic_gates(row)
            row["gates"]["locomotion_floor"] = bool(
                float(row["forward_displacement_m"]) >= (0.040 if kind == "flat" else 0.025)
            )
            row["gate_pass_count"] = int(sum(row["gates"].values()))
            row["all_diagnostic_gates"] = bool(all(row["gates"].values()))
            rows.append(row)

    by_scale = {}
    for body_scale in BODY_AMPLITUDE_SCALES:
        pair = [
            row
            for row in rows
            if math.isclose(
                float(row["body_amplitude_scale"]), body_scale, rel_tol=0.0, abs_tol=1e-12
            )
        ]
        by_scale[f"{body_scale:.2f}"] = {
            "all_conditions_pass": bool(all(row["all_diagnostic_gates"] for row in pair)),
            "minimum_forward_displacement_m": min(
                float(row["forward_displacement_m"]) for row in pair
            ),
            "maximum_spine_tortuosity": max(
                float(row["maximum_spine_tortuosity"]) for row in pair
            ),
            "minimum_root_upright_cos": min(
                float(row["minimum_root_upright_cos"]) for row in pair
            ),
            "maximum_q2_deg": max(float(row["q2_abs_max_deg"]) for row in pair),
            "maximum_penetration_p95_mm": max(
                float(row["penetration_p95_mm"]) for row in pair
            ),
        }
    passing = [
        scale
        for scale in BODY_AMPLITUDE_SCALES
        if by_scale[f"{scale:.2f}"]["all_conditions_pass"]
    ]
    if passing:
        # Retain the largest feasible body wave.  This preserves more of the
        # source gait while enforcing the pre-existing tortuosity constraint;
        # it is a feasibility rule, not a scalar locomotion reward.
        selected = max(passing)
        status = "passes all flat/rough 6 s plant, contact, posture and locomotion gates"
    else:
        selected = min(BODY_AMPLITUDE_SCALES)
        status = "diagnostic only; no body-wave scale passed every gate"
    result = {
        "purpose": (
            "fixed-plant body-wave amplitude sweep; leg amplitudes, frequency, phase, gains, "
            "q2 compliance, mass and inertia remain frozen"
        ),
        "profile": profile.name,
        "duration_s": duration_s,
        "passive_name": passive_name,
        "source_body_amplitude_rad": float(base.BODY_AMPLITUDE_RAD),
        "scales": list(BODY_AMPLITUDE_SCALES),
        "locomotion_floor_m": {"flat": 0.040, "rough": 0.025},
        "rows": rows,
        "by_scale": by_scale,
        "selected_body_amplitude_scale": selected,
        "selection_status": status,
        "selection_rule": (
            "largest scale passing every explicit gate, to retain source gait amplitude "
            "without violating tortuosity"
        ),
        "robot_sentinels_before": before,
        "robot_sentinels_after": robot_sentinels(),
    }
    result["robot_sentinels_unchanged"] = (
        result["robot_sentinels_before"] == result["robot_sentinels_after"]
    )
    (DATA / "body_wave_sweep.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def run_head_hold_probe(
    profile: ContactProfile,
    duration_s: float,
    passive_name: str = "restoring_k30_c18",
    body_amplitude_scale: float = 0.55,
) -> dict[str, object]:
    """Test a straight head-to-first-body yaw target without changing leg 0."""

    manifest = build_passive_all(profile)
    before = robot_sentinels()
    rows = []
    for kind, wave in (("flat", 1.5), ("rough", float(ROUGH_SPEC["wave"]))):
        model_path = Path(manifest["variants"][f"{kind}_{passive_name}"]["model"])
        row = run_diagnostic(
            f"{kind} first-body-yaw hold",
            model_path,
            wave,
            float(manifest["q2_limit_deg"]),
            duration_s=duration_s,
            q2_mode="passive",
            q2_passive_name=passive_name,
            body_amplitude_scale=body_amplitude_scale,
            hold_first_body_yaw=True,
        )
        row["kind"] = kind
        row["gates"] = diagnostic_gates(row)
        row["gates"]["locomotion_floor"] = bool(
            float(row["forward_displacement_m"]) >= (0.040 if kind == "flat" else 0.025)
        )
        row["all_diagnostic_gates"] = bool(all(row["gates"].values()))
        rows.append(row)
    result = {
        "purpose": (
            "first body-yaw hold probe; first leg-pair yaw/pitch motion and all plant parameters "
            "remain unchanged"
        ),
        "profile": profile.name,
        "duration_s": duration_s,
        "passive_name": passive_name,
        "body_amplitude_scale": float(body_amplitude_scale),
        "hold_first_body_yaw": True,
        "rows": rows,
        "all_rows_pass": bool(all(row["all_diagnostic_gates"] for row in rows)),
        "robot_sentinels_before": before,
        "robot_sentinels_after": robot_sentinels(),
    }
    result["robot_sentinels_unchanged"] = (
        result["robot_sentinels_before"] == result["robot_sentinels_after"]
    )
    (DATA / "head_first_body_hold_probe.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def render_demos(
    profile: ContactProfile,
    duration_s: float,
    passive_name_override: str | None = None,
    body_amplitude_scale: float = 1.0,
    hold_first_body_yaw: bool = False,
) -> dict[str, object]:
    sweep_path = PASSIVE_SWEEP_PATH
    if passive_name_override is None:
        if not sweep_path.exists():
            run_passive_sweep(profile, min(duration_s, 4.5))
        sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
        limit = float(sweep["q2_limit_deg"])
        passive_name = str(sweep["demo_passive_name"])
        selection_status = str(sweep["demo_status"])
        manifest = json.loads(
            (DATA / "passive_model_manifest.json").read_text(encoding="utf-8")
        )
    else:
        passive_name = passive_name_override
        manifest = build_passive_all(profile)
        limit = float(manifest["q2_limit_deg"])
        selection_status = "explicit convergence-probe candidate"
    flat = Path(manifest["variants"][f"flat_{passive_name}"]["model"])
    rough = Path(manifest["variants"][f"rough_{passive_name}"]["model"])
    body_tag = f"b{body_amplitude_scale:.2f}".replace(".", "p")
    head_tag = "headhold" if hold_first_body_yaw else "headfeedback"
    before = robot_sentinels()
    cases = [
        (
            "flat side",
            flat,
            1.5,
            VIDEOS / f"stage1_flat_q2pm{limit:g}_{passive_name}_{body_tag}_{head_tag}_side.mp4",
            90.0,
            -12.0,
            0.19,
        ),
        (
            "rough side",
            rough,
            float(ROUGH_SPEC["wave"]),
            VIDEOS / f"stage1_l60a12_local_q2pm{limit:g}_{passive_name}_{body_tag}_{head_tag}_side.mp4",
            90.0,
            -12.0,
            0.19,
        ),
        (
            "rough oblique",
            rough,
            float(ROUGH_SPEC["wave"]),
            VIDEOS / f"stage1_l60a12_local_q2pm{limit:g}_{passive_name}_{body_tag}_{head_tag}_oblique.mp4",
            125.0,
            -35.0,
            0.19,
        ),
        (
            "rough top",
            rough,
            float(ROUGH_SPEC["wave"]),
            VIDEOS / f"stage1_l60a12_local_q2pm{limit:g}_{passive_name}_{body_tag}_{head_tag}_top.mp4",
            90.0,
            -89.0,
            0.19,
        ),
    ]
    rows = []
    for label, model_path, wave, video, azimuth, elevation, distance in cases:
        rows.append(
            run_diagnostic(
                label,
                model_path,
                wave,
                limit,
                duration_s=duration_s,
                video=video,
                camera_azimuth=azimuth,
                camera_elevation=elevation,
                camera_distance=distance,
                q2_mode="passive",
                q2_passive_name=passive_name,
                body_amplitude_scale=body_amplitude_scale,
                hold_first_body_yaw=hold_first_body_yaw,
            )
        )
    result = {
        "profile": profile.name,
        "q2_limit_deg": limit,
        "q2_mode": "passive",
        "q2_passive_name": passive_name,
        "q2_passive_stiffness_nm_rad": PASSIVE_Q2[passive_name][0],
        "q2_passive_damping_nms_rad": PASSIVE_Q2[passive_name][1],
        "body_amplitude_scale": float(body_amplitude_scale),
        "body_amplitude_rad": float(base.BODY_AMPLITUDE_RAD) * float(body_amplitude_scale),
        "hold_first_body_yaw": bool(hold_first_body_yaw),
        "selection_status": selection_status,
        "rows": rows,
        "visual_contract": (
            "side/oblique/top use the same trajectory seed and exact collision hfield; "
            "overlay reports instantaneous max |q2| and contacting-foot count"
        ),
        "robot_sentinels_before": before,
        "robot_sentinels_after": robot_sentinels(),
    }
    result["robot_sentinels_unchanged"] = (
        result["robot_sentinels_before"] == result["robot_sentinels_after"]
    )
    (DATA / "demo_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def compact_result(result: dict[str, object]) -> dict[str, object]:
    compact = dict(result)
    if "rows" in compact:
        compact["row_count"] = len(compact["rows"])
        del compact["rows"]
    if "variants" in compact:
        compact["variant_count"] = len(compact["variants"])
        del compact["variants"]
    for key in ("robot_sentinels_before", "robot_sentinels_after"):
        compact.pop(key, None)
    return compact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "calibrate",
            "build",
            "sweep",
            "passive",
            "probe",
            "body",
            "head",
            "demo",
            "all",
        ),
    )
    parser.add_argument("--profile", choices=tuple(CONTACT_PROFILES), default="medium_dt0p5ms")
    parser.add_argument("--duration", type=float, default=4.5)
    parser.add_argument("--passive-name", choices=tuple(PASSIVE_Q2))
    parser.add_argument("--body-scale", type=float, default=1.0)
    parser.add_argument("--hold-first-body-yaw", action="store_true")
    args = parser.parse_args()
    ensure_directories()
    profile = CONTACT_PROFILES[args.profile]
    output = {}
    if args.command in {"calibrate", "all"}:
        output["calibration"] = calibrate_contacts()
    if args.command in {"build", "all"}:
        output["build"] = build_all(profile)
    if args.command in {"sweep", "all"}:
        output["sweep"] = run_sweep(profile, args.duration)
    if args.command in {"passive", "all"}:
        output["passive"] = run_passive_sweep(profile, args.duration)
    if args.command == "probe":
        passive_names = (
            (args.passive_name,)
            if args.passive_name is not None
            else ("restoring_k20_c12", "restoring_k30_c18")
        )
        output["probe"] = run_passive_probe(profile, args.duration, passive_names)
    if args.command == "body":
        output["body"] = run_body_wave_sweep(
            profile,
            args.duration,
            passive_name=args.passive_name or "restoring_k30_c18",
        )
    if args.command == "head":
        output["head"] = run_head_hold_probe(
            profile,
            args.duration,
            passive_name=args.passive_name or "restoring_k30_c18",
            body_amplitude_scale=args.body_scale,
        )
    if args.command in {"demo", "all"}:
        output["demo"] = render_demos(
            profile,
            args.duration,
            passive_name_override=args.passive_name,
            body_amplitude_scale=args.body_scale,
            hold_first_body_yaw=args.hold_first_body_yaw,
        )
    print(json.dumps({key: compact_result(value) for key, value in output.items()}, indent=2))


if __name__ == "__main__":
    main()
