"""Controller optimization for the zero-armature Centipede Model.

This experiment imports the validated geometry/inertia rebuild as a frozen
plant.  It never reads or writes Centipede Robot Step 18/19 outputs.  The
controller search separates locomotion-driving yaw/leg gains from pitch/roll
terrain compliance and keeps every candidate's metric vector.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path
from xml.etree import ElementTree as ET

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from scipy.stats import qmc


ROOT = Path(__file__).resolve().parent
BASE_EXPERIMENT = ROOT.parent / "mujoco_zero_armature_inertia_rebuild_2026-08-25"
BASE_MODEL = BASE_EXPERIMENT / "models" / "centipede_geometry_inertia_armature0.xml"
OUTPUT = ROOT / "outputs"
DATA = OUTPUT / "data"
VIDEOS = OUTPUT / "videos"
TERRAINS = OUTPUT / "terrains"
MODELS = ROOT / "models"

BODY_LENGTH_M = 0.1025
TERRAIN_SEED = 244949
ARENA_SIZE_M = 3.0
HEIGHTFIELD_PIXELS = 512
HEAD_YAW_GAIN = 8.0
HEAD_YAW_TARGET_CLIP_RAD = 0.6
HEAD_BODY_HOLD_RELEASE_START_S = 1.0
HEAD_BODY_HOLD_RELEASE_DURATION_S = 0.25
# Flat-ground lift/propulsion refinement selected with the stability gates
# below.  The clean-RL pitch amplitude remains the source value; these are
# explicit Model-only gait multipliers, not physical-Robot parameters.
LEG_YAW_AMPLITUDE_SCALE = 2.2
LEG_PITCH_AMPLITUDE_SCALE = 1.4
ROUGH_LEG_YAW_AMPLITUDE_SCALE = 0.95
ROUGH_LEG_PITCH_AMPLITUDE_SCALE = 0.95
KNEE_DC_RAD = math.pi / 6.0

TERRAIN_SPECS = {
    "flat": (1000.0, 0.0, 1.5),
    "l90a8": (90.0, 8.0, 2.0),
    "l60a12": (60.0, 12.0, 3.0),
}

# Balanced shared controller selected after feasibility filtering.  It is the
# best retained worst-condition point among the staged candidates: one gain
# vector is used for flat, L90/A8 and L60/A12.  These are multipliers of the
# validated zero-armature geometry-derived controller, not Robot gains.
FINAL_SCALE_MULTIPLIERS = {
    "body_yaw": (2.491080818837389, 0.9626449806432559),
    "body_pitch": (2.4, 2.0),
    "body_roll": (2.0, 2.0),
    "legbase_yaw": (3.9656874288824366, 13.843513937070757),
    "legbase_pitch": (2.957549583794445, 10.87587338715009),
    "legbase_roll": (0.7428381463987926, 5.568526993458692),
    "knee": (1.5, 3.0),
}


def _load_base_pipeline():
    path = BASE_EXPERIMENT / "run_pipeline.py"
    spec = importlib.util.spec_from_file_location("zero_armature_base_pipeline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = _load_base_pipeline()


class FloatWaveController(base.LockedTravelingWaveController):
    """The frozen target generator with fractional spatial-wave support."""

    def __init__(self, model: mujoco.MjModel, index, wave: float):
        super().__init__(model, index, 1)
        self.wave = float(wave)
        self.spatial_phase = (
            2.0
            * math.pi
            * self.wave
            * np.arange(base.N_BODY_YAW, dtype=float)
            / (base.N_BODY_YAW - 1)
        )
        self.head_yaw_reference: float | None = None
        # The more assertive lift/propulsion profile was selected only for the
        # flat-ground w=1.5 contract.  Applying it to rough-terrain w=2/3 was
        # measured to violate tortuosity and qvel gates, so those conditions
        # retain the previously validated clean-RL amplitudes.
        if math.isclose(self.wave, 1.5, rel_tol=0.0, abs_tol=1e-12):
            self.leg_amplitudes[0] *= LEG_YAW_AMPLITUDE_SCALE
            self.leg_amplitudes[1] *= LEG_PITCH_AMPLITUDE_SCALE
        else:
            self.leg_amplitudes[0] *= ROUGH_LEG_YAW_AMPLITUDE_SCALE
            self.leg_amplitudes[1] *= ROUGH_LEG_PITCH_AMPLITUDE_SCALE
        self.leg_dc[3] = KNEE_DC_RAD

    def step(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        t = float(data.time)
        dt = float(self.model.opt.timestep)
        omega = 2.0 * math.pi * base.FREQUENCY_HZ
        target = self.target
        target.fill(0.0)

        body_phase = omega * t - self.spatial_phase
        body_blend = np.asarray(
            [base.dct_blend(t, i, base.N_BODY_YAW) for i in range(base.N_BODY_YAW)], dtype=float
        )
        body_target = body_blend * base.BODY_AMPLITUDE_RAD * np.sin(body_phase)
        yaw = base.head_yaw(data, self.index)
        if self.head_yaw_reference is None:
            self.head_yaw_reference = yaw
        yaw_error = base.unwrap_difference(yaw, self.head_yaw_reference)
        if math.isclose(self.wave, 1.5, rel_tol=0.0, abs_tol=1e-12):
            release_u = float(
                np.clip(
                    (t - HEAD_BODY_HOLD_RELEASE_START_S) / HEAD_BODY_HOLD_RELEASE_DURATION_S,
                    0.0,
                    1.0,
                )
            )
            release_blend = 0.5 - 0.5 * math.cos(math.pi * release_u)
        else:
            release_blend = 1.0
        body_target[0] = release_blend * float(
            np.clip(HEAD_YAW_GAIN * yaw_error, -HEAD_YAW_TARGET_CLIP_RAD, HEAD_YAW_TARGET_CLIP_RAD)
        )
        target[self.index.body_positions] = body_target

        pitch_target = np.zeros(base.N_BODY_PITCH, dtype=float)
        pitch_target[0] = -0.57 * base.dct_blend(t, 0, base.N_BODY_YAW)
        target[self.index.pitch_positions] = pitch_target

        leg_blend = np.asarray(
            [base.dct_blend(t, i, base.N_LEG_PAIRS) for i in range(base.N_LEG_PAIRS)], dtype=float
        )
        phases = body_phase[:, None, None] + self.leg_offsets[None, None, :]
        leg_target = (
            leg_blend[:, None, None]
            * self.side_sign[None, :, None]
            * self.leg_amplitudes[None, None, :]
            * np.sin(phases)
            + self.leg_dc[None, None, :]
        )
        inactive = np.asarray([False, False, True, True])
        leg_target[:, :, inactive] = self.leg_dc[inactive]
        target[self.index.leg_positions.reshape(-1)] = leg_target.reshape(-1)

        if self.previous_target is None:
            target_velocity = np.zeros_like(target)
        else:
            target_velocity = (target - self.previous_target) / dt
        q = data.qpos[self.index.qpos_addresses]
        qdot = data.qvel[self.index.dof_addresses]
        composite_reference = target + (self.index.kv / self.index.kp) * target_velocity
        torque = self.index.kp * (target - q) + self.index.kv * (target_velocity - qdot)
        data.ctrl[:] = 0.0
        data.ctrl[self.index.active_actuator_ids] = composite_reference
        self.previous_target = target.copy()
        return torque, qdot


def float_phase_groups(wave: float) -> list[list[int]]:
    groups: dict[float, list[int]] = {}
    for segment in range(base.N_LEG_PAIRS):
        residue = round((float(wave) * segment) % (base.N_BODY_YAW - 1), 10)
        groups.setdefault(residue, []).append(segment)
    return [groups[key] for key in sorted(groups)]


def float_build_events(config):
    seed_wave = int(round(float(config.wave) * 1000.0))
    rng = np.random.default_rng(int(config.seed) + 100_003 * seed_wave)
    triggered = []
    opportunities = 0
    omega = 2.0 * math.pi * base.FREQUENCY_HZ
    for segment in range(base.N_LEG_PAIRS):
        phase = 2.0 * math.pi * float(config.wave) * segment / (base.N_BODY_YAW - 1)
        phase_offset = (phase / omega) % base.SUPPORT_WINDOW_S
        window_start = config.eval_start_s + phase_offset
        channel_use = 0
        while window_start < config.duration_s - 1e-12:
            opportunities += 1
            u_trigger, u_duration, u_offset = rng.random(3)
            fraction = math.exp(
                math.log(config.minimum_burst_fraction)
                + u_duration * math.log(1.0 / config.minimum_burst_fraction)
            )
            duration = fraction * base.SUPPORT_WINDOW_S
            start = window_start + u_offset * (base.SUPPORT_WINDOW_S - duration)
            if u_trigger < config.q:
                triggered.append(
                    base.ErasureEvent(
                        segment=segment,
                        channel_use=channel_use,
                        window_start_s=window_start,
                        start_s=start,
                        end_s=start + duration,
                        duration_s=duration,
                        duration_fraction=fraction,
                        trigger_uniform=u_trigger,
                    )
                )
            channel_use += 1
            window_start += base.SUPPORT_WINDOW_S
    triggered.sort(key=lambda event: event.start_s)
    return triggered, opportunities


# The previous experiment remains unedited; only this process receives the
# fractional-wave compatibility patch.
base.LockedTravelingWaveController = FloatWaveController
base.phase_groups = float_phase_groups
base.build_events = float_build_events


def spectral_terrain(lambda_mm: float, amplitude_mm: float, seed: int) -> np.ndarray:
    """Frozen clean-RL homogeneous terrain algorithm at reduced raster size."""

    size = HEIGHTFIELD_PIXELS
    amplitude_m = float(amplitude_mm) * 1e-3
    if amplitude_m == 0.0:
        return np.zeros((size, size), dtype=np.float64)
    wavelength_m = float(lambda_mm) * 1e-3
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
    std = float(height.std())
    if std > 1e-12:
        height = height / std * amplitude_m
    pixels_per_cycle = wavelength_m / ARENA_SIZE_M * size
    height = gaussian_filter(height, sigma=max(0.25 * pixels_per_cycle, 0.5))
    height -= height.mean()
    peak = max(abs(float(height.min())), abs(float(height.max())))
    return height * (amplitude_m / peak) if peak > 1e-12 else height


def collision_quantization(source: np.ndarray) -> tuple[np.ndarray, float]:
    low, high = float(source.min()), float(source.max())
    peak = max(abs(low), abs(high))
    z_max = max(2.0 * peak, 0.005)
    if high - low < 1e-15:
        pixels = np.zeros_like(source, dtype=np.uint8)
    else:
        pixels = np.round((source - low) / (high - low) * 255.0).astype(np.uint8)
    collision = pixels.astype(np.float64) / 255.0 * z_max
    return collision, z_max


def height_at(height: np.ndarray, x_m: float, y_m: float) -> float:
    rows, cols = height.shape
    fx = np.clip((x_m + ARENA_SIZE_M / 2) / ARENA_SIZE_M * (cols - 1), 0, cols - 1)
    fy = np.clip((y_m + ARENA_SIZE_M / 2) / ARENA_SIZE_M * (rows - 1), 0, rows - 1)
    x0, y0 = int(np.floor(fx)), int(np.floor(fy))
    x1, y1 = min(x0 + 1, cols - 1), min(y0 + 1, rows - 1)
    tx, ty = fx - x0, fy - y0
    return float(
        (1 - tx) * (1 - ty) * height[y0, x0]
        + tx * (1 - ty) * height[y0, x1]
        + (1 - tx) * ty * height[y1, x0]
        + tx * ty * height[y1, x1]
    )


class TerrainSampler:
    def __init__(self, model: mujoco.MjModel):
        hfield_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain")
        self.enabled = hfield_id >= 0
        if not self.enabled:
            self.nrow = self.ncol = 0
            self.x_half = self.y_half = self.z_top = 0.0
            self.data = np.zeros((1, 1), dtype=float)
            self.offset = np.zeros(3, dtype=float)
            return
        self.nrow = int(model.hfield_nrow[hfield_id])
        self.ncol = int(model.hfield_ncol[hfield_id])
        self.x_half = float(model.hfield_size[hfield_id, 0])
        self.y_half = float(model.hfield_size[hfield_id, 1])
        self.z_top = float(model.hfield_size[hfield_id, 2])
        start = int(model.hfield_adr[hfield_id])
        self.data = model.hfield_data[start : start + self.nrow * self.ncol].reshape(
            self.nrow, self.ncol
        ).copy()
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain_geom")
        self.offset = model.geom_pos[geom_id].copy() if geom_id >= 0 else np.zeros(3)

    def get_height(self, x: float, y: float) -> float:
        if not self.enabled:
            return 0.0
        u = np.clip((x - self.offset[0] + self.x_half) / (2.0 * self.x_half), 0.0, 1.0)
        v = np.clip((y - self.offset[1] + self.y_half) / (2.0 * self.y_half), 0.0, 1.0)
        col = u * (self.ncol - 1)
        row = v * (self.nrow - 1)
        c0, r0 = int(np.floor(col)), int(np.floor(row))
        c1, r1 = min(c0 + 1, self.ncol - 1), min(r0 + 1, self.nrow - 1)
        fc, fr = col - c0, row - r0
        value = (
            self.data[r0, c0] * (1 - fc) * (1 - fr)
            + self.data[r0, c1] * fc * (1 - fr)
            + self.data[r1, c0] * (1 - fc) * fr
            + self.data[r1, c1] * fc * fr
        )
        return float(self.offset[2] + value * self.z_top)

    def slopes(self, x: float, y: float, heading: float, delta: float = 0.003) -> tuple[float, float]:
        forward = np.asarray([math.cos(heading), math.sin(heading)])
        left = np.asarray([-math.sin(heading), math.cos(heading)])
        xy = np.asarray([x, y])
        pitch_slope = (
            self.get_height(*(xy + delta * forward))
            - self.get_height(*(xy - delta * forward))
        ) / (2.0 * delta)
        roll_slope = (
            self.get_height(*(xy + delta * left))
            - self.get_height(*(xy - delta * left))
        ) / (2.0 * delta)
        return float(pitch_slope), float(roll_slope)


def prepare_terrain_models() -> dict[str, dict]:
    TERRAINS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    for name, (lambda_mm, amplitude_mm, wave) in TERRAIN_SPECS.items():
        if name == "flat":
            manifest[name] = {
                "model": str(BASE_MODEL),
                "height_npy": "",
                "lambda_mm": lambda_mm,
                "amplitude_mm": amplitude_mm,
                "wave": wave,
            }
            continue
        source = spectral_terrain(lambda_mm, amplitude_mm, TERRAIN_SEED)
        collision, z_max = collision_quantization(source)
        terrain_dir = TERRAINS / name
        terrain_dir.mkdir(parents=True, exist_ok=True)
        png_path = terrain_dir / "heightfield.png"
        npy_path = terrain_dir / "collision_height.npy"
        Image.fromarray(np.round(collision / z_max * 255.0).astype(np.uint8), mode="L").save(png_path)
        np.save(npy_path, collision)

        tree = ET.parse(BASE_MODEL)
        xml_root = tree.getroot()
        asset = xml_root.find("asset")
        worldbody = xml_root.find("worldbody")
        if asset is None or worldbody is None:
            raise RuntimeError("Base MJCF is missing asset/worldbody")
        ET.SubElement(
            asset,
            "hfield",
            {
                "name": "terrain",
                "file": png_path.as_posix(),
                "size": f"{ARENA_SIZE_M / 2:.9g} {ARENA_SIZE_M / 2:.9g} {z_max:.9g} 0.001",
            },
        )
        floor = worldbody.find("geom[@name='floor']")
        if floor is not None:
            worldbody.remove(floor)
        ET.SubElement(
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
        free_body = next((body for body in xml_root.iter("body") if body.find("freejoint") is not None), None)
        if free_body is None:
            raise RuntimeError("Base MJCF has no free-root body")
        radius = 0.075
        axis = np.linspace(-ARENA_SIZE_M / 2, ARENA_SIZE_M / 2, collision.shape[0])
        mask = np.logical_and(np.abs(axis) <= radius, np.abs(axis[:, None]) <= radius)
        local_max = float(np.max(collision[mask]))
        base_pos = np.fromstring(free_body.get("pos", "0 0 0"), sep=" ")
        base_pos[2] += local_max
        free_body.set("pos", " ".join(f"{value:.12g}" for value in base_pos))
        model_path = MODELS / f"centipede_{name}_armature0.xml"
        tree.write(model_path, encoding="utf-8", xml_declaration=True)
        loaded = mujoco.MjModel.from_xml_path(str(model_path))
        if not np.allclose(loaded.dof_armature, 0.0, atol=1e-16):
            raise RuntimeError(f"{name}: terrain patch changed armature")
        manifest[name] = {
            "model": str(model_path),
            "height_npy": str(npy_path),
            "height_png": str(png_path),
            "lambda_mm": lambda_mm,
            "amplitude_mm": amplitude_mm,
            "wave": wave,
            "z_max_m": z_max,
            "relief_m": float(collision.max() - collision.min()),
            "spawn_local_max_m": local_max,
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        }
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "terrain_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_terrain_manifest() -> dict[str, dict]:
    path = DATA / "terrain_manifest.json"
    if not path.exists():
        return prepare_terrain_models()
    return json.loads(path.read_text(encoding="utf-8"))


def scaled_impedance(scales: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
    result = {}
    for family, (kp, kv) in base.IMPEDANCE.items():
        kp_scale, kv_scale = scales.get(family, (1.0, 1.0))
        result[family] = (float(kp * kp_scale), float(kv * kv_scale))
    return result


def parse_scales(text: str) -> dict[str, tuple[float, float]]:
    result: dict[str, tuple[float, float]] = {}
    if not text:
        return result
    for item in text.split(";"):
        family, values = item.split("=", 1)
        kp, kv = (float(value) for value in values.split(",", 1))
        if family not in base.IMPEDANCE:
            raise ValueError(f"Unknown family {family}")
        result[family] = (kp, kv)
    return result


def run_condition(
    condition: str,
    impedance: dict[str, tuple[float, float]],
    duration_s: float = 4.5,
    eval_start_s: float = 1.5,
    video: Path | None = None,
) -> dict:
    manifest = load_terrain_manifest()
    wave = float(manifest[condition]["wave"])
    config = replace(
        base.TrialConfig(wave=1, q=0.0, seed=0),
        wave=wave,
        duration_s=float(duration_s),
        eval_start_s=float(eval_start_s),
    )
    result = base.run_trial(
        config,
        model_path=Path(manifest[condition]["model"]),
        render_path=video,
        armature_override_kg_m2=0.0,
        impedance=impedance,
    )
    eval_duration = max(duration_s - eval_start_s, 1e-12)
    result["condition"] = condition
    result["speed_bl_cycle"] = result["absolute_forward_displacement_m"] / eval_duration / BODY_LENGTH_M
    result["rough_path_speed_bl_cycle"] = result["path_length_m"] / eval_duration / BODY_LENGTH_M
    result["pass_gate"] = bool(
        result["finite"]
        and result["warning_counts"] == "{}"
        and result["minimum_root_upright_cos"] >= 0.259
        and result["maximum_spine_tortuosity"] <= 1.30
        and result["peak_abs_active_joint_speed_rad_s"] <= 100.0
    )
    return result


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compact(row: dict) -> dict:
    return {
        "condition": row["condition"],
        "pass_gate": row["pass_gate"],
        "speed_bl_cycle": row["speed_bl_cycle"],
        "rough_path_speed_bl_cycle": row["rough_path_speed_bl_cycle"],
        "forward_m": row["absolute_forward_displacement_m"],
        "path_efficiency": row["path_efficiency"],
        "yaw_drift_rad": row["yaw_drift_rad"],
        "minimum_root_upright_cos": row["minimum_root_upright_cos"],
        "maximum_spine_tortuosity": row["maximum_spine_tortuosity"],
        "peak_speed_rad_s": row["peak_abs_active_joint_speed_rad_s"],
        "peak_speed_joint": row["peak_active_joint_speed_name"],
        "warning_counts": row["warning_counts"],
        "finite": row["finite"],
    }


def safe_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    valid = np.logical_and(np.isfinite(a), np.isfinite(b))
    a, b = a[valid], b[valid]
    if a.size < 3 or float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def run_compliance_trace(
    condition: str,
    impedance: dict[str, tuple[float, float]],
    duration_s: float = 6.0,
    eval_start_s: float = 1.5,
) -> dict:
    manifest = load_terrain_manifest()
    model = mujoco.MjModel.from_xml_path(manifest[condition]["model"])
    index = base.ModelIndex(model, impedance=impedance)
    base.configure_native_implicit_pd(model, index, 0.0)
    data = mujoco.MjData(model)
    wave = float(manifest[condition]["wave"])
    controller = FloatWaveController(model, index, wave)
    terrain = TerrainSampler(model)
    mujoco.mj_resetData(model, data)
    knees = index.leg_positions[:, :, 3].reshape(-1)
    data.qpos[index.qpos_addresses[knees]] = math.pi / 6.0
    mujoco.mj_forward(model, data)

    pitch_joint = index.pitch_positions[1:]
    roll_joint = index.roll_positions
    pitch_values: list[float] = []
    roll_values: list[float] = []
    body_pitch_values: list[float] = []
    body_roll_values: list[float] = []
    terrain_pitch_values: list[float] = []
    terrain_roll_values: list[float] = []
    vertical_correlations: list[float] = []
    clearance_residual_rms: list[float] = []
    next_sample = eval_start_s
    dt = float(model.opt.timestep)
    finite = True
    while data.time < duration_s - 0.5 * dt:
        controller.step(data)
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            finite = False
            break
        if data.time + 1e-12 < next_sample:
            continue
        pitch_values.extend(data.qpos[index.qpos_addresses[pitch_joint]].tolist())
        roll_values.extend(data.qpos[index.qpos_addresses[roll_joint]].tolist())
        positions = data.xpos[index.chassis_body_ids]
        body_pitch_frame = []
        body_roll_frame = []
        terrain_pitch_frame = []
        terrain_roll_frame = []
        terrain_height_frame = []
        for body_id, position in zip(index.chassis_body_ids, positions):
            rotation = data.xmat[body_id].reshape(3, 3)
            yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
            pitch = math.atan2(
                -float(rotation[2, 0]),
                math.hypot(float(rotation[2, 1]), float(rotation[2, 2])),
            )
            roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
            pitch_slope, roll_slope = terrain.slopes(float(position[0]), float(position[1]), yaw)
            body_pitch_frame.append(pitch)
            body_roll_frame.append(roll)
            terrain_pitch_frame.append(math.atan(pitch_slope))
            terrain_roll_frame.append(math.atan(roll_slope))
            terrain_height_frame.append(terrain.get_height(float(position[0]), float(position[1])))
        body_pitch_values.extend(body_pitch_frame)
        body_roll_values.extend(body_roll_frame)
        terrain_pitch_values.extend(terrain_pitch_frame)
        terrain_roll_values.extend(terrain_roll_frame)
        body_z = positions[:, 2] - float(np.mean(positions[:, 2]))
        terrain_z = np.asarray(terrain_height_frame) - float(np.mean(terrain_height_frame))
        vertical_correlations.append(safe_correlation(body_z, terrain_z))
        residual = (positions[:, 2] - np.asarray(terrain_height_frame))
        clearance_residual_rms.append(float(np.std(residual)))
        next_sample += 0.01

    pitch_array = np.asarray(pitch_values)
    roll_array = np.asarray(roll_values)
    warnings = {
        mujoco.mjtWarning(warning_id).name: int(item.number)
        for warning_id, item in enumerate(data.warning)
        if int(item.number) > 0
    }
    valid_vertical = np.asarray(vertical_correlations, dtype=float)
    valid_vertical = valid_vertical[np.isfinite(valid_vertical)]
    return {
        "condition": condition,
        "wave": wave,
        "duration_s": duration_s,
        "eval_start_s": eval_start_s,
        "finite": finite,
        "warning_counts": warnings,
        "body_pitch_joint_rms_deg": float(np.degrees(np.sqrt(np.mean(pitch_array**2)))),
        "body_pitch_joint_p95_abs_deg": float(np.degrees(np.quantile(np.abs(pitch_array), 0.95))),
        "body_roll_joint_rms_deg": float(np.degrees(np.sqrt(np.mean(roll_array**2)))),
        "body_roll_joint_p95_abs_deg": float(np.degrees(np.quantile(np.abs(roll_array), 0.95))),
        "body_pitch_vs_local_terrain_slope_corr": safe_correlation(
            np.asarray(body_pitch_values), np.asarray(terrain_pitch_values)
        ),
        "body_roll_vs_local_terrain_slope_corr": safe_correlation(
            np.asarray(body_roll_values), np.asarray(terrain_roll_values)
        ),
        "mean_within_spine_vertical_terrain_corr": (
            float(np.mean(valid_vertical)) if valid_vertical.size else float("nan")
        ),
        "mean_clearance_shape_rms_mm": float(1000.0 * np.mean(clearance_residual_rms)),
    }


def run_flat_lift_diagnostic(
    impedance: dict[str, tuple[float, float]],
    duration_s: float = 6.0,
    eval_start_s: float = 1.5,
    side_video: Path | None = None,
    camera_azimuth: float = 90.0,
    camera_elevation: float = -6.0,
    camera_distance: float = 0.19,
) -> dict:
    """Measure actual foot lift/contact and optionally render a true side view."""

    model = mujoco.MjModel.from_xml_path(str(BASE_MODEL))
    index = base.ModelIndex(model, impedance=impedance)
    base.configure_native_implicit_pd(model, index, 0.0)
    data = mujoco.MjData(model)
    controller = FloatWaveController(model, index, 1.5)
    mujoco.mj_resetData(model, data)
    knees = index.leg_positions[:, :, 3].reshape(-1)
    data.qpos[index.qpos_addresses[knees]] = KNEE_DC_RAD
    mujoco.mj_forward(model, data)

    floor_geom = base.name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    foot_geom_ids = np.asarray(
        [
            base.name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{segment}_{side}_collision")
            for segment in range(base.N_LEG_PAIRS)
            for side in range(2)
        ],
        dtype=int,
    )
    foot_body_ids = np.asarray(
        [
            base.name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"foot_{segment}_{side}")
            for segment in range(base.N_LEG_PAIRS)
            for side in range(2)
        ],
        dtype=int,
    )
    foot_geom_to_position = {int(geom): position for position, geom in enumerate(foot_geom_ids)}
    nonfoot_leg_geom_ids = {
        geom_id
        for geom_id in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").startswith(
            "link_leg_"
        )
    }
    foot_radii = model.geom_size[foot_geom_ids, 0].copy()
    contact_steps = np.zeros(len(foot_geom_ids), dtype=int)
    clearance_samples: list[list[float]] = [[] for _ in foot_geom_ids]
    all_feet_grounded_steps = 0
    lifted_0p5mm_fraction_sum = 0.0
    chassis_contact_steps = 0
    nonfoot_leg_contact_steps = 0
    nonfoot_leg_contact_fraction_sum = 0.0
    maximum_nonfoot_leg_contact_fraction = 0.0
    sample_count = 0
    body_first_target: list[float] = []
    body_first_q: list[float] = []
    leg0_left_pitch_target: list[float] = []
    leg0_right_pitch_target: list[float] = []
    leg0_left_pitch_q: list[float] = []
    leg0_right_pitch_q: list[float] = []
    warmup_head_yaw_error: list[float] = []
    settle_head_yaw_error: list[float] = []
    activation_head_yaw_error: list[float] = []
    settle_first_body_q: list[float] = []
    activation_first_body_q: list[float] = []
    reference_yaw: float | None = None
    measurement_start: np.ndarray | None = None
    measurement_end: np.ndarray | None = None
    previous_center: np.ndarray | None = None
    path_length = 0.0
    yaw_start: float | None = None
    yaw_end: float | None = None
    min_upright = float("inf")
    max_tortuosity = 0.0
    peak_speed = 0.0
    finite = True
    next_sample = 0.0
    dt = float(model.opt.timestep)

    renderer = None
    writer = None
    camera = None
    next_frame = 0.0
    if side_video is not None:
        os.environ.setdefault("MUJOCO_GL", "wgl")
        side_video.parent.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(model, height=720, width=1280)
        writer = imageio.get_writer(
            side_video, fps=30, codec="libx264", quality=8, macro_block_size=None
        )
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance = float(camera_distance)
        camera.azimuth = float(camera_azimuth)
        camera.elevation = float(camera_elevation)

    try:
        while data.time < duration_s - 0.5 * dt:
            controller.step(data)
            if reference_yaw is None and controller.head_yaw_reference is not None:
                reference_yaw = controller.head_yaw_reference
            mujoco.mj_step(model, data)
            peak_speed = max(peak_speed, float(np.max(np.abs(data.qvel[index.dof_addresses]))))
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                finite = False
                break

            if data.time <= 1.5 and reference_yaw is not None:
                warmup_head_yaw_error.append(
                    abs(base.unwrap_difference(base.head_yaw(data, index), reference_yaw))
                )
                first_body_q_value = float(
                    data.qpos[index.qpos_addresses[int(index.body_positions[0])]]
                )
                if data.time <= 1.0:
                    settle_head_yaw_error.append(warmup_head_yaw_error[-1])
                    settle_first_body_q.append(first_body_q_value)
                if data.time <= 1.125:
                    activation_head_yaw_error.append(warmup_head_yaw_error[-1])
                    activation_first_body_q.append(first_body_q_value)

            if data.time + 1e-12 >= next_sample:
                center = base.chassis_centroid(data, index)
                posture = base.posture_metrics(data, index)
                min_upright = min(min_upright, posture["root_upright_cos"])
                max_tortuosity = max(max_tortuosity, posture["spine_tortuosity"])

                body_position = int(index.body_positions[0])
                left_position = int(index.leg_positions[0, 0, 1])
                right_position = int(index.leg_positions[0, 1, 1])
                body_first_target.append(float(controller.target[body_position]))
                body_first_q.append(float(data.qpos[index.qpos_addresses[body_position]]))
                leg0_left_pitch_target.append(float(controller.target[left_position]))
                leg0_right_pitch_target.append(float(controller.target[right_position]))
                leg0_left_pitch_q.append(float(data.qpos[index.qpos_addresses[left_position]]))
                leg0_right_pitch_q.append(float(data.qpos[index.qpos_addresses[right_position]]))

                if data.time >= eval_start_s:
                    if measurement_start is None:
                        measurement_start = center.copy()
                        yaw_start = base.head_yaw(data, index)
                    if previous_center is not None:
                        path_length += float(np.linalg.norm(center[:2] - previous_center[:2]))
                    previous_center = center.copy()
                    measurement_end = center.copy()
                    yaw_end = base.head_yaw(data, index)

                    contacted = np.zeros(len(foot_geom_ids), dtype=bool)
                    chassis_contact = False
                    nonfoot_leg_contact = False
                    contacted_nonfoot_leg_geoms: set[int] = set()
                    for contact_id in range(data.ncon):
                        contact = data.contact[contact_id]
                        if int(contact.geom1) == floor_geom:
                            other = int(contact.geom2)
                        elif int(contact.geom2) == floor_geom:
                            other = int(contact.geom1)
                        else:
                            continue
                        if other in foot_geom_to_position:
                            contacted[foot_geom_to_position[other]] = True
                        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or ""
                        if name.startswith("link_leg_"):
                            nonfoot_leg_contact = True
                            contacted_nonfoot_leg_geoms.add(other)
                        elif name.startswith("link_"):
                            chassis_contact = True
                    contact_steps += contacted.astype(int)
                    chassis_contact_steps += int(chassis_contact)
                    nonfoot_leg_contact_steps += int(nonfoot_leg_contact)
                    nonfoot_fraction = len(contacted_nonfoot_leg_geoms) / max(
                        len(nonfoot_leg_geom_ids), 1
                    )
                    nonfoot_leg_contact_fraction_sum += nonfoot_fraction
                    maximum_nonfoot_leg_contact_fraction = max(
                        maximum_nonfoot_leg_contact_fraction, nonfoot_fraction
                    )
                    foot_clearance = data.xpos[foot_body_ids, 2] - foot_radii
                    for position, value in enumerate(foot_clearance):
                        clearance_samples[position].append(float(value))
                    all_feet_grounded_steps += int(np.all(contacted))
                    lifted_0p5mm_fraction_sum += float(np.mean(foot_clearance > 0.0005))
                    sample_count += 1
                next_sample += 0.01

            if renderer is not None and writer is not None and data.time + 1e-12 >= next_frame:
                center = base.chassis_centroid(data, index)
                camera.lookat[:] = (float(center[0]), float(center[1]), 0.011)
                renderer.update_scene(data, camera=camera)
                canvas = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(canvas)
                label = (
                    f"az={camera_azimuth:.0f} flat w=1.5 pitch_amp={base.LEG_PITCH_AMPLITUDE_RAD * LEG_PITCH_AMPLITUDE_SCALE:.3f}rad "
                    "head/first-body hold; leg0 pitch active; armature=0"
                )
                draw.rectangle((18, 18, 920, 58), fill=(28, 28, 28))
                draw.text((30, 29), label, fill=(255, 255, 255))
                writer.append_data(np.asarray(canvas))
                next_frame += 1.0 / 30.0
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()

    clearance_arrays = [np.asarray(values, dtype=float) for values in clearance_samples]
    apex = np.asarray(
        [np.quantile(values, 0.95) if values.size else float("nan") for values in clearance_arrays]
    )
    contact_duty = contact_steps / max(sample_count, 1)
    if measurement_start is None or measurement_end is None or yaw_start is None or yaw_end is None:
        finite = False
        directed = path_efficiency = yaw_drift = float("nan")
    else:
        directed = base.FORWARD_SIGN * float(measurement_end[0] - measurement_start[0])
        path_efficiency = directed / max(path_length, 1e-12)
        yaw_drift = base.unwrap_difference(yaw_end, yaw_start)
    eval_duration = max(duration_s - eval_start_s, 1e-12)
    warning_counts = {
        mujoco.mjtWarning(warning_id).name: int(item.number)
        for warning_id, item in enumerate(data.warning)
        if int(item.number) > 0
    }
    return {
        "duration_s": duration_s,
        "eval_start_s": eval_start_s,
        "leg_yaw_amplitude_scale": LEG_YAW_AMPLITUDE_SCALE,
        "leg_pitch_amplitude_scale": LEG_PITCH_AMPLITUDE_SCALE,
        "leg_pitch_amplitude_rad": base.LEG_PITCH_AMPLITUDE_RAD * LEG_PITCH_AMPLITUDE_SCALE,
        "knee_dc_rad": KNEE_DC_RAD,
        "warmup_settle_s": 1.0,
        "warmup_dct_duration_s": 0.125,
        "first_body_yaw_hold_release_start_s": HEAD_BODY_HOLD_RELEASE_START_S,
        "first_body_yaw_hold_release_duration_s": HEAD_BODY_HOLD_RELEASE_DURATION_S,
        "first_body_wave_enabled": False,
        "first_body_world_yaw_hold_enabled": True,
        "first_leg_pitch_wave_enabled": True,
        "warmup_head_yaw_error_max_deg": float(np.degrees(max(warmup_head_yaw_error, default=float("nan")))),
        "settle_head_yaw_error_max_deg": float(
            np.degrees(max(settle_head_yaw_error, default=float("nan")))
        ),
        "activation_head_yaw_error_max_deg": float(
            np.degrees(max(activation_head_yaw_error, default=float("nan")))
        ),
        "settle_first_body_q_max_abs_deg": float(
            np.degrees(max((abs(value) for value in settle_first_body_q), default=float("nan")))
        ),
        "activation_first_body_q_max_abs_deg": float(
            np.degrees(max((abs(value) for value in activation_first_body_q), default=float("nan")))
        ),
        "first_body_target_max_abs_deg": float(np.degrees(np.max(np.abs(body_first_target)))),
        "first_body_q_max_abs_deg": float(np.degrees(np.max(np.abs(body_first_q)))),
        "first_leg_left_pitch_target_range_deg": float(np.degrees(np.ptp(leg0_left_pitch_target))),
        "first_leg_right_pitch_target_range_deg": float(np.degrees(np.ptp(leg0_right_pitch_target))),
        "first_leg_left_pitch_q_range_deg": float(np.degrees(np.ptp(leg0_left_pitch_q))),
        "first_leg_right_pitch_q_range_deg": float(np.degrees(np.ptp(leg0_right_pitch_q))),
        "mean_foot_contact_duty": float(np.mean(contact_duty)),
        "minimum_foot_contact_duty": float(np.min(contact_duty)),
        "maximum_foot_contact_duty": float(np.max(contact_duty)),
        "median_foot_apex_clearance_mm": float(1000.0 * np.nanmedian(apex)),
        "p10_foot_apex_clearance_mm": float(1000.0 * np.nanquantile(apex, 0.10)),
        "p90_foot_apex_clearance_mm": float(1000.0 * np.nanquantile(apex, 0.90)),
        "mean_fraction_feet_above_0p5mm": lifted_0p5mm_fraction_sum / max(sample_count, 1),
        "all_feet_grounded_fraction": all_feet_grounded_steps / max(sample_count, 1),
        "chassis_ground_contact_fraction": chassis_contact_steps / max(sample_count, 1),
        "nonfoot_leg_ground_contact_fraction": nonfoot_leg_contact_steps / max(sample_count, 1),
        "mean_fraction_nonfoot_leg_geoms_in_contact": (
            nonfoot_leg_contact_fraction_sum / max(sample_count, 1)
        ),
        "maximum_fraction_nonfoot_leg_geoms_in_contact": maximum_nonfoot_leg_contact_fraction,
        "forward_displacement_m": directed,
        "forward_speed_bl_cycle": directed / eval_duration / BODY_LENGTH_M,
        "path_efficiency": path_efficiency,
        "yaw_drift_rad": yaw_drift,
        "minimum_root_upright_cos": min_upright,
        "maximum_spine_tortuosity": max_tortuosity,
        "peak_active_joint_speed_rad_s": peak_speed,
        "warning_counts": warning_counts,
        "finite": finite,
        "side_video": str(side_video) if side_video is not None else "",
        "camera_azimuth_deg": float(camera_azimuth),
        "camera_elevation_deg": float(camera_elevation),
        "camera_distance_m": float(camera_distance),
    }


def final_impedance() -> dict[str, tuple[float, float]]:
    return scaled_impedance(FINAL_SCALE_MULTIPLIERS)


def run_final_random_condition(duration: float, seed: int, video: Path | None = None) -> dict:
    row = base.run_trial(
        replace(
            base.TrialConfig(wave=3, q=0.3, seed=int(seed)),
            duration_s=float(duration),
            eval_start_s=1.5,
        ),
        model_path=BASE_MODEL,
        render_path=video,
        armature_override_kg_m2=0.0,
        impedance=final_impedance(),
    )
    eval_duration = max(duration - 1.5, 1e-12)
    row["condition"] = "flat_random_contact_w3_q030"
    row["speed_bl_cycle"] = row["absolute_forward_displacement_m"] / eval_duration / BODY_LENGTH_M
    row["rough_path_speed_bl_cycle"] = row["path_length_m"] / eval_duration / BODY_LENGTH_M
    row["pass_gate"] = bool(
        row["finite"]
        and row["warning_counts"] == "{}"
        and row["minimum_root_upright_cos"] >= 0.259
        and row["maximum_spine_tortuosity"] <= 1.30
        and row["peak_abs_active_joint_speed_rad_s"] <= 100.0
    )
    return row


def run_final_validation(duration: float, render: bool) -> dict:
    prepare_terrain_models()
    impedance = final_impedance()
    locomotion = []
    compliance = []
    for condition in TERRAIN_SPECS:
        video = VIDEOS / f"optimized_{condition}.mp4" if render else None
        locomotion.append(run_condition(condition, impedance, duration, 1.5, video))
        compliance.append(run_compliance_trace(condition, impedance, duration, 1.5))
        print(json.dumps(compact(locomotion[-1]), sort_keys=True), flush=True)
        print(json.dumps(compliance[-1], sort_keys=True), flush=True)
    random_contact = []
    for seed in range(3):
        video = VIDEOS / "optimized_random_contact_w3_q030_seed0.mp4" if render and seed == 0 else None
        row = run_final_random_condition(duration, seed, video)
        random_contact.append(row)
        print(json.dumps(compact(row), sort_keys=True), flush=True)
    DATA.mkdir(parents=True, exist_ok=True)
    write_rows(DATA / "final_locomotion_validation.csv", locomotion)
    (DATA / "final_compliance_validation.json").write_text(
        json.dumps(compliance, indent=2), encoding="utf-8"
    )
    write_rows(DATA / "final_random_contact_validation.csv", random_contact)
    result = {
        "controller": impedance,
        "scale_multipliers": FINAL_SCALE_MULTIPLIERS,
        "head_yaw_reference": {
            "gain": HEAD_YAW_GAIN,
            "target_clip_rad": HEAD_YAW_TARGET_CLIP_RAD,
        },
        "duration_s": duration,
        "all_locomotion_gates_pass": all(bool(row["pass_gate"]) for row in locomotion),
        "all_random_contact_gates_pass": all(bool(row["pass_gate"]) for row in random_contact),
        "locomotion": [compact(row) for row in locomotion],
        "compliance": compliance,
        "random_contact": [compact(row) for row in random_contact],
    }
    (DATA / "final_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def render_final_demos(duration: float) -> list[dict]:
    prepare_terrain_models()
    impedance = final_impedance()
    rows = []
    for condition in TERRAIN_SPECS:
        row = run_condition(
            condition,
            impedance,
            duration_s=duration,
            eval_start_s=1.5,
            video=VIDEOS / f"optimized_{condition}.mp4",
        )
        rows.append(row)
        print(json.dumps(compact(row), sort_keys=True), flush=True)
    random_row = base.run_trial(
        replace(
            base.TrialConfig(wave=3, q=0.3, seed=0),
            duration_s=float(duration),
            eval_start_s=1.5,
        ),
        model_path=BASE_MODEL,
        render_path=VIDEOS / "optimized_random_contact_w3_q030_seed0.mp4",
        armature_override_kg_m2=0.0,
        impedance=impedance,
    )
    random_row["condition"] = "flat_random_contact_w3_q030"
    random_row["speed_bl_cycle"] = (
        random_row["absolute_forward_displacement_m"] / max(duration - 1.5, 1e-12) / BODY_LENGTH_M
    )
    random_row["rough_path_speed_bl_cycle"] = (
        random_row["path_length_m"] / max(duration - 1.5, 1e-12) / BODY_LENGTH_M
    )
    random_row["pass_gate"] = bool(
        random_row["finite"]
        and random_row["warning_counts"] == "{}"
        and random_row["minimum_root_upright_cos"] >= 0.259
        and random_row["maximum_spine_tortuosity"] <= 1.30
        and random_row["peak_abs_active_joint_speed_rad_s"] <= 100.0
    )
    rows.append(random_row)
    print(json.dumps(compact(random_row), sort_keys=True), flush=True)
    write_rows(DATA / "demo_manifest.csv", rows)
    return rows


def run_manual(scales_text: str, conditions: list[str], duration: float, video: bool) -> list[dict]:
    impedance = scaled_impedance(parse_scales(scales_text))
    rows = []
    for condition in conditions:
        path = VIDEOS / f"manual_{condition}.mp4" if video else None
        row = run_condition(condition, impedance, duration_s=duration, video=path)
        rows.append(row)
        print(json.dumps(compact(row), sort_keys=True), flush=True)
    return rows


def load_candidate(candidate_id: str) -> dict[str, tuple[float, float]]:
    path = DATA / "sobol_flat_screen.csv"
    with path.open("r", newline="", encoding="utf-8") as handle:
        row = next((item for item in csv.DictReader(handle) if item["candidate_id"] == candidate_id), None)
    if row is None:
        raise ValueError(f"Candidate {candidate_id} is not present in {path}")
    decoded = json.loads(row["impedance_vector"])
    return {family: (float(values[0]), float(values[1])) for family, values in decoded.items()}


def candidate_from_vector(values: np.ndarray, candidate_id: int) -> tuple[str, dict, dict]:
    families = (
        "body_yaw",
        "body_pitch",
        "body_roll",
        "legbase_yaw",
        "legbase_pitch",
        "legbase_roll",
        "knee",
    )
    # Broad but finite log ranges. Head pitch is frozen per the RL contract.
    bounds = {
        "body_yaw": ((0.75, 6.0), (0.75, 12.0)),
        "body_pitch": ((0.35, 2.5), (0.5, 6.0)),
        "body_roll": ((0.35, 2.5), (0.5, 6.0)),
        "legbase_yaw": ((0.75, 8.0), (0.75, 16.0)),
        "legbase_pitch": ((0.75, 8.0), (0.75, 16.0)),
        "legbase_roll": ((0.5, 3.0), (0.75, 12.0)),
        "knee": ((0.5, 4.0), (0.75, 12.0)),
    }
    scales = {}
    metadata = {"candidate_id": f"S{candidate_id:03d}"}
    for index, family in enumerate(families):
        kp_range, kv_range = bounds[family]
        kp_scale = math.exp(math.log(kp_range[0]) + values[2 * index] * math.log(kp_range[1] / kp_range[0]))
        kv_scale = math.exp(math.log(kv_range[0]) + values[2 * index + 1] * math.log(kv_range[1] / kv_range[0]))
        scales[family] = (kp_scale, kv_scale)
        metadata[f"{family}_kp_scale"] = kp_scale
        metadata[f"{family}_kv_scale"] = kv_scale
    return metadata["candidate_id"], scaled_impedance(scales), metadata


def _screen_one(args_tuple):
    candidate_id, impedance, metadata = args_tuple
    row = run_condition("flat", impedance, duration_s=4.5, eval_start_s=1.5)
    row.update(metadata)
    row["impedance_vector"] = json.dumps(impedance, sort_keys=True)
    return row


def run_sobol_screen(count: int, workers: int) -> list[dict]:
    if count <= 0 or count & (count - 1):
        raise ValueError("Sobol count must be a positive power of two")
    prepare_terrain_models()
    sample = qmc.Sobol(d=14, scramble=True, seed=20260826).random_base2(int(math.log2(count)))
    candidates = [candidate_from_vector(vector, index) for index, vector in enumerate(sample)]
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(_screen_one, candidate): candidate[0] for candidate in candidates}
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            summary = compact(row)
            print(f"[{completed:03d}/{count:03d}] {row['candidate_id']} {json.dumps(summary, sort_keys=True)}", flush=True)
    rows.sort(key=lambda row: row["candidate_id"])
    write_rows(DATA / "sobol_flat_screen.csv", rows)
    return rows


def main() -> None:
    global HEAD_YAW_GAIN, HEAD_YAW_TARGET_CLIP_RAD
    global LEG_YAW_AMPLITUDE_SCALE, LEG_PITCH_AMPLITUDE_SCALE, KNEE_DC_RAD
    global ROUGH_LEG_YAW_AMPLITUDE_SCALE, ROUGH_LEG_PITCH_AMPLITUDE_SCALE
    global HEAD_BODY_HOLD_RELEASE_START_S, HEAD_BODY_HOLD_RELEASE_DURATION_S
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    manual = sub.add_parser("manual")
    manual.add_argument("--scales", default="")
    manual.add_argument("--candidate", default="")
    manual.add_argument("--conditions", default="flat,l90a8,l60a12")
    manual.add_argument("--duration", type=float, default=4.5)
    manual.add_argument("--video", action="store_true")
    manual.add_argument("--head-yaw-gain", type=float, default=HEAD_YAW_GAIN)
    manual.add_argument("--head-yaw-clip", type=float, default=HEAD_YAW_TARGET_CLIP_RAD)
    screen = sub.add_parser("screen")
    screen.add_argument("--count", type=int, default=64)
    screen.add_argument("--workers", type=int, default=4)
    validate = sub.add_parser("validate")
    validate.add_argument("--duration", type=float, default=9.6)
    validate.add_argument("--render", action="store_true")
    demos = sub.add_parser("demos")
    demos.add_argument("--duration", type=float, default=9.6)
    final_condition = sub.add_parser("final-condition")
    final_condition.add_argument("--condition", choices=tuple(TERRAIN_SPECS), required=True)
    final_condition.add_argument("--duration", type=float, default=9.6)
    final_condition.add_argument(
        "--rough-leg-yaw-scale", type=float, default=ROUGH_LEG_YAW_AMPLITUDE_SCALE
    )
    final_condition.add_argument(
        "--rough-leg-pitch-scale", type=float, default=ROUGH_LEG_PITCH_AMPLITUDE_SCALE
    )
    final_condition.add_argument("--output", type=Path)
    final_random = sub.add_parser("final-random")
    final_random.add_argument("--seed", type=int, required=True)
    final_random.add_argument("--duration", type=float, default=9.6)
    final_random.add_argument("--output", type=Path)
    lift = sub.add_parser("lift-diagnostic")
    lift.add_argument("--duration", type=float, default=6.0)
    lift.add_argument("--leg-yaw-scale", type=float, default=LEG_YAW_AMPLITUDE_SCALE)
    lift.add_argument("--leg-pitch-scale", type=float, default=LEG_PITCH_AMPLITUDE_SCALE)
    lift.add_argument("--knee-dc-deg", type=float, default=30.0)
    lift.add_argument(
        "--head-body-release-start", type=float, default=HEAD_BODY_HOLD_RELEASE_START_S
    )
    lift.add_argument(
        "--head-body-release-duration", type=float, default=HEAD_BODY_HOLD_RELEASE_DURATION_S
    )
    lift.add_argument("--side-video", type=Path)
    lift.add_argument("--camera-azimuth", type=float, default=90.0)
    lift.add_argument("--camera-elevation", type=float, default=-6.0)
    lift.add_argument("--camera-distance", type=float, default=0.19)
    lift.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.command == "prepare":
        print(json.dumps(prepare_terrain_models(), indent=2))
    elif args.command == "manual":
        HEAD_YAW_GAIN = float(args.head_yaw_gain)
        HEAD_YAW_TARGET_CLIP_RAD = float(args.head_yaw_clip)
        conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
        if args.candidate:
            impedance = load_candidate(args.candidate)
            for condition in conditions:
                path = VIDEOS / f"{args.candidate}_{condition}.mp4" if args.video else None
                print(json.dumps(compact(run_condition(condition, impedance, args.duration, 1.5, path)), sort_keys=True), flush=True)
        else:
            run_manual(args.scales, conditions, args.duration, args.video)
    elif args.command == "screen":
        run_sobol_screen(args.count, args.workers)
    elif args.command == "validate":
        print(json.dumps(run_final_validation(args.duration, args.render), indent=2))
    elif args.command == "demos":
        render_final_demos(args.duration)
    elif args.command == "final-condition":
        ROUGH_LEG_YAW_AMPLITUDE_SCALE = float(args.rough_leg_yaw_scale)
        ROUGH_LEG_PITCH_AMPLITUDE_SCALE = float(args.rough_leg_pitch_scale)
        result = run_condition(
            args.condition,
            final_impedance(),
            duration_s=float(args.duration),
            eval_start_s=1.5,
        )
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(compact(result), sort_keys=True), flush=True)
    elif args.command == "final-random":
        result = run_final_random_condition(float(args.duration), int(args.seed))
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(compact(result), sort_keys=True), flush=True)
    elif args.command == "lift-diagnostic":
        LEG_YAW_AMPLITUDE_SCALE = float(args.leg_yaw_scale)
        LEG_PITCH_AMPLITUDE_SCALE = float(args.leg_pitch_scale)
        KNEE_DC_RAD = math.radians(float(args.knee_dc_deg))
        HEAD_BODY_HOLD_RELEASE_START_S = float(args.head_body_release_start)
        HEAD_BODY_HOLD_RELEASE_DURATION_S = float(args.head_body_release_duration)
        result = run_flat_lift_diagnostic(
            final_impedance(),
            duration_s=float(args.duration),
            eval_start_s=1.5,
            side_video=args.side_video,
            camera_azimuth=float(args.camera_azimuth),
            camera_elevation=float(args.camera_elevation),
            camera_distance=float(args.camera_distance),
        )
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
