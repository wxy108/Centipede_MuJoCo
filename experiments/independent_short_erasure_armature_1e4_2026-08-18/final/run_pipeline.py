"""Independent short-contact-erasure experiments for the FARMS centipede.

The morphology, impedance gains, gait amplitudes, frequency, terrain, and
evaluation horizon are fixed.  Only the integer spatial wave number and the
independent event-trigger probability q change.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import csv
import heapq
import hashlib
import io
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import yaml
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "farms" / "centipede_reliability_refined.xml"
ORIGINAL_MODEL_PATH = Path(
    r"C:\Users\wxy22\Documents\Centipede_MUJOCO-main\models\farms\centipede.xml"
)
DATA_DIR = ROOT / "outputs" / "data"
FIGURE_DIR = ROOT / "outputs" / "figures"
VIDEO_DIR = ROOT / "outputs" / "videos"
CONTROLLER_DIR = ROOT / "controllers" / "farms"
CONTROLLER_CONFIG = ROOT / "configs" / "farms_controller.yaml"

if str(CONTROLLER_DIR) not in sys.path:
    sys.path.insert(0, str(CONTROLLER_DIR))
from impedance_controller import ImpedanceTravelingWaveController

N_LEG_PAIRS = 19
N_BODY_YAW = 19
N_BODY_PITCH = 20
N_BODY_ROLL = 20
N_LEG_DOF = 4
FREQUENCY_HZ = 1.0
GAIT_PERIOD_S = 1.0
SUPPORT_WINDOW_S = 0.5
# The native FARMS controller advances along world -x in this MJCF.  Reported
# forward displacement is therefore -(x_final-x_initial), in metres.
FORWARD_SIGN = -1.0

ROOT_TILT_LIMIT_DEG = 45.0
SEGMENT_TILT_LIMIT_DEG = 75.0
SPINE_JOINT_LIMIT_DEG = 35.0
DYNAMIC_PENETRATION_LIMIT_M = 3.5e-3

# Locked at the clean Isaac-Lab wave=2.5 operating point.  These values are
# held fixed for every wave in this experiment to avoid a gait-map confound.
BODY_AMPLITUDE_RAD = 0.4 * 1.3702376747306109
LEG_YAW_AMPLITUDE_RAD = 0.6 * 0.5453011208974309
LEG_PITCH_AMPLITUDE_RAD = 0.4 * 1.15

IMPEDANCE = {
    "body_yaw": (0.00996598695234778, 6.998080672480617e-08),
    "head_pitch": (0.03041804430974008, 0.00768829695690118),
    "body_pitch": (0.0006, 0.0009),
    "body_roll": (0.07536, 0.0036062445840513925),
    "legbase_yaw": (0.004293438984912552, 4.249695808879143e-06),
    "legbase_pitch": (0.000728032484289019, 9.855345633124885e-07),
    "legbase_roll": (0.00024, 2.94e-08),
    "knee": (0.00586, 0.000433),
}

DEFAULT_WAVES = (1, 2, 3, 4)
DEFAULT_Q = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4)


@dataclass(frozen=True)
class TrialConfig:
    wave: int
    q: float
    seed: int
    duration_s: float = 8.0
    eval_start_s: float = 2.0
    minimum_burst_fraction: float = 0.02


@dataclass(frozen=True)
class ErasureEvent:
    segment: int
    channel_use: int
    window_start_s: float
    start_s: float
    end_s: float
    duration_s: float
    duration_fraction: float
    trigger_uniform: float


def name2id(model: mujoco.MjModel, object_type, name: str) -> int:
    value = mujoco.mj_name2id(model, object_type, name)
    if value < 0:
        raise ValueError(f"Missing {name}")
    return int(value)


class ModelIndex:
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        actuator_for_joint: dict[int, int] = {}
        for actuator_id in range(model.nu):
            joint_id = int(model.actuator_trnid[actuator_id, 0])
            actuator_for_joint[joint_id] = actuator_id

        self.active_joint_ids: list[int] = []
        self.active_actuator_ids: list[int] = []
        self.qpos_addresses: list[int] = []
        self.dof_addresses: list[int] = []
        self.kp: list[float] = []
        self.kv: list[float] = []
        self.position_by_name: dict[str, int] = {}

        def add(joint_name: str, family: str) -> int:
            joint_id = name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id not in actuator_for_joint:
                raise ValueError(f"No actuator for {joint_name}")
            position = len(self.active_joint_ids)
            self.active_joint_ids.append(joint_id)
            self.active_actuator_ids.append(actuator_for_joint[joint_id])
            self.qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
            self.dof_addresses.append(int(model.jnt_dofadr[joint_id]))
            gain = IMPEDANCE[family]
            self.kp.append(gain[0])
            self.kv.append(gain[1])
            self.position_by_name[joint_name] = position
            return position

        self.body_positions = np.asarray(
            [add(f"joint_body_{i}", "body_yaw") for i in range(1, N_BODY_YAW + 1)],
            dtype=int,
        )
        self.pitch_positions = np.asarray(
            [
                add(
                    f"joint_pitch_body_{i}",
                    "head_pitch" if i == 0 else "body_pitch",
                )
                for i in range(N_BODY_PITCH)
            ],
            dtype=int,
        )
        self.roll_positions = np.asarray(
            [add(f"joint_roll_body_{i}", "body_roll") for i in range(N_BODY_ROLL)],
            dtype=int,
        )
        self.leg_positions = np.empty((N_LEG_PAIRS, 2, N_LEG_DOF), dtype=int)
        leg_families = ("legbase_yaw", "legbase_pitch", "legbase_roll", "knee")
        for segment in range(N_LEG_PAIRS):
            for side_index, side in enumerate(("L", "R")):
                for dof, family in enumerate(leg_families):
                    self.leg_positions[segment, side_index, dof] = add(
                        f"joint_leg_{segment}_{side}_{dof}", family
                    )
        if len(self.active_joint_ids) != 211:
            raise RuntimeError(f"Expected 211 active joints, found {len(self.active_joint_ids)}")

        self.active_actuator_ids = np.asarray(self.active_actuator_ids, dtype=int)
        self.qpos_addresses = np.asarray(self.qpos_addresses, dtype=int)
        self.dof_addresses = np.asarray(self.dof_addresses, dtype=int)
        self.kp = np.asarray(self.kp, dtype=float)
        self.kv = np.asarray(self.kv, dtype=float)
        self.chassis_body_ids = np.asarray(
            [name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"link_body_{i}") for i in range(21)],
            dtype=int,
        )
        self.head_body_id = int(self.chassis_body_ids[0])

        self.segment_collision_geoms: list[np.ndarray] = []
        self.segment_visual_geoms: list[np.ndarray] = []
        for segment in range(N_LEG_PAIRS):
            collision: list[int] = []
            visual: list[int] = []
            leg_pattern = re.compile(rf"^link_leg_{segment}_[LR]_[0-3]_(collision|visual)$")
            foot_pattern = re.compile(rf"^foot_{segment}_[01]_(collision|visual)$")
            for geom_id in range(model.ngeom):
                geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
                match = leg_pattern.fullmatch(geom_name) or foot_pattern.fullmatch(geom_name)
                if not match:
                    continue
                (collision if match.group(1) == "collision" else visual).append(geom_id)
            if len(collision) != 10 or len(visual) != 10:
                raise RuntimeError(
                    f"Segment {segment}: expected 10 collision and 10 visual geoms, "
                    f"found {len(collision)} and {len(visual)}"
                )
            self.segment_collision_geoms.append(np.asarray(collision, dtype=int))
            self.segment_visual_geoms.append(np.asarray(visual, dtype=int))


def dct_blend(t: float, index: int, count: int, settle: float = 1.0, ramp: float = 0.25) -> float:
    if ramp <= 0.0:
        return float(t >= settle)
    duration = 0.5 * ramp
    start = settle + index * duration / max(count - 1, 1)
    elapsed = max(t - start, 0.0)
    if elapsed >= duration:
        return 1.0
    if elapsed <= 0.0:
        return 0.0
    return 0.5 * (1.0 - math.cos(math.pi * elapsed / duration))


class LockedTravelingWaveController:
    def __init__(self, model: mujoco.MjModel, index: ModelIndex, wave: int):
        self.model = model
        self.index = index
        self.wave = int(wave)
        self.previous_target: np.ndarray | None = None
        self.target = np.zeros(211, dtype=float)
        self.leg_amplitudes = np.asarray(
            [LEG_YAW_AMPLITUDE_RAD, LEG_PITCH_AMPLITUDE_RAD, 0.0, 0.0], dtype=float
        )
        self.leg_offsets = np.asarray([-math.pi / 2.0, math.pi, 0.0, math.pi], dtype=float)
        self.leg_dc = np.asarray([0.0, 0.0, 0.0, math.pi / 6.0], dtype=float)
        self.side_sign = np.asarray([1.0, -1.0], dtype=float)
        self.spatial_phase = (
            2.0 * math.pi * self.wave * np.arange(N_BODY_YAW, dtype=float) / (N_BODY_YAW - 1)
        )

    def step(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        t = float(data.time)
        dt = float(self.model.opt.timestep)
        omega = 2.0 * math.pi * FREQUENCY_HZ
        target = self.target
        target.fill(0.0)

        body_phase = omega * t - self.spatial_phase
        body_blend = np.asarray(
            [dct_blend(t, i, N_BODY_YAW) for i in range(N_BODY_YAW)], dtype=float
        )
        body_target = body_blend * BODY_AMPLITUDE_RAD * np.sin(body_phase)
        body_target[0] = 0.0
        target[self.index.body_positions] = body_target

        pitch_target = np.zeros(N_BODY_PITCH, dtype=float)
        pitch_target[0] = -0.57 * dct_blend(t, 0, N_BODY_YAW)
        target[self.index.pitch_positions] = pitch_target

        leg_blend = np.asarray(
            [dct_blend(t, i, N_LEG_PAIRS) for i in range(N_LEG_PAIRS)], dtype=float
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
        torque = self.index.kp * (target - q) + self.index.kv * (target_velocity - qdot)
        data.ctrl[:] = 0.0
        data.ctrl[self.index.active_actuator_ids] = torque
        self.previous_target = target.copy()
        return torque, qdot


def make_farms_controller(
    model: mujoco.MjModel, wave: int
) -> ImpedanceTravelingWaveController:
    """Build the validated local FARMS controller and vary only wave number."""
    with contextlib.redirect_stdout(io.StringIO()):
        controller = ImpedanceTravelingWaveController(model, str(CONTROLLER_CONFIG))
    controller.n_wave = float(wave)
    controller.cpg_delta = (
        2.0 * math.pi * controller.n_wave * controller.speed
        / max(N_BODY_YAW - 1, 1)
    )
    return controller


def phase_groups(wave: int) -> list[list[int]]:
    groups: dict[int, list[int]] = {}
    for segment in range(N_LEG_PAIRS):
        residue = (int(wave) * segment) % (N_BODY_YAW - 1)
        groups.setdefault(residue, []).append(segment)
    return [groups[key] for key in sorted(groups)]


def expected_burst_fraction(minimum_fraction: float) -> float:
    if not 0.0 < minimum_fraction <= 1.0:
        raise ValueError("minimum_fraction must lie in (0, 1]")
    if minimum_fraction == 1.0:
        return 1.0
    return (1.0 - minimum_fraction) / math.log(1.0 / minimum_fraction)


def predicted_any_group_blackout(wave: int, q: float, minimum_fraction: float = 0.02) -> float:
    """Independent random-time snapshot approximation.

    A segment-pair is erased for marginal time fraction nu=q E[A].  Treating
    segment states as independent at a random snapshot gives nu**m_g for a
    complete blackout of phase group g.
    """
    nu = q * expected_burst_fraction(minimum_fraction)
    survival = 1.0
    for group in phase_groups(wave):
        survival *= 1.0 - nu ** len(group)
    return 1.0 - survival


def build_events(config: TrialConfig) -> tuple[list[ErasureEvent], int]:
    rng = np.random.default_rng(config.seed + 100_003 * config.wave)
    triggered: list[ErasureEvent] = []
    opportunities = 0
    omega = 2.0 * math.pi * FREQUENCY_HZ
    for segment in range(N_LEG_PAIRS):
        phase = 2.0 * math.pi * config.wave * segment / (N_BODY_YAW - 1)
        phase_offset = (phase / omega) % SUPPORT_WINDOW_S
        window_start = config.eval_start_s + phase_offset
        channel_use = 0
        while window_start < config.duration_s - 1e-12:
            opportunities += 1
            u_trigger, u_duration, u_offset = rng.random(3)
            fraction = math.exp(
                math.log(config.minimum_burst_fraction)
                + u_duration * math.log(1.0 / config.minimum_burst_fraction)
            )
            duration = fraction * SUPPORT_WINDOW_S
            start = window_start + u_offset * (SUPPORT_WINDOW_S - duration)
            if u_trigger < config.q:
                triggered.append(
                    ErasureEvent(
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
            window_start += SUPPORT_WINDOW_S
    triggered.sort(key=lambda event: event.start_s)
    return triggered, opportunities


class ErasureManager:
    RED = np.asarray([1.0, 0.03, 0.03, 1.0], dtype=float)

    def __init__(self, model: mujoco.MjModel, index: ModelIndex, events: list[ErasureEvent]):
        self.model = model
        self.index = index
        self.events = events
        self.pointer = 0
        self.ends: list[tuple[float, int]] = []
        self.counts = np.zeros(N_LEG_PAIRS, dtype=int)
        self.original_contype = model.geom_contype.copy()
        self.original_conaffinity = model.geom_conaffinity.copy()
        self.original_rgba = model.geom_rgba.copy()

    def _activate(self, segment: int) -> None:
        if self.counts[segment] == 0:
            collision = self.index.segment_collision_geoms[segment]
            visible = self.index.segment_visual_geoms[segment]
            self.model.geom_contype[collision] = 0
            self.model.geom_conaffinity[collision] = 0
            self.model.geom_rgba[visible] = self.RED
            self.model.geom_rgba[collision] = self.RED
        self.counts[segment] += 1

    def _deactivate(self, segment: int) -> None:
        self.counts[segment] -= 1
        if self.counts[segment] < 0:
            raise RuntimeError("Negative erasure overlap count")
        if self.counts[segment] == 0:
            collision = self.index.segment_collision_geoms[segment]
            visible = self.index.segment_visual_geoms[segment]
            all_geoms = np.concatenate((collision, visible))
            self.model.geom_contype[collision] = self.original_contype[collision]
            self.model.geom_conaffinity[collision] = self.original_conaffinity[collision]
            self.model.geom_rgba[all_geoms] = self.original_rgba[all_geoms]

    def update(self, t: float) -> np.ndarray:
        while self.ends and self.ends[0][0] <= t + 1e-12:
            _, segment = heapq.heappop(self.ends)
            self._deactivate(segment)
        while self.pointer < len(self.events) and self.events[self.pointer].start_s <= t + 1e-12:
            event = self.events[self.pointer]
            self._activate(event.segment)
            heapq.heappush(self.ends, (event.end_s, event.segment))
            self.pointer += 1
        return self.counts > 0

    def restore(self) -> None:
        self.model.geom_contype[:] = self.original_contype
        self.model.geom_conaffinity[:] = self.original_conaffinity
        self.model.geom_rgba[:] = self.original_rgba
        self.counts.fill(0)
        self.ends.clear()


def chassis_centroid(data: mujoco.MjData, index: ModelIndex) -> np.ndarray:
    return np.mean(data.xpos[index.chassis_body_ids], axis=0)


def head_yaw(data: mujoco.MjData, index: ModelIndex) -> float:
    rotation = data.xmat[index.head_body_id].reshape(3, 3)
    return math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))


def unwrap_difference(final: float, initial: float) -> float:
    return (final - initial + math.pi) % (2.0 * math.pi) - math.pi


def euler_roll_pitch(rotation: np.ndarray) -> tuple[float, float]:
    roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
    pitch = math.atan2(
        -float(rotation[2, 0]),
        math.hypot(float(rotation[2, 1]), float(rotation[2, 2])),
    )
    return roll, pitch


def tilt_deg(rotation: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip(rotation[2, 2], -1.0, 1.0))))


def contact_summary(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float | int]:
    floor_id = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    floor_distances: list[float] = []
    self_distances: list[float] = []
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        target = (
            floor_distances
            if floor_id in (int(contact.geom1), int(contact.geom2))
            else self_distances
        )
        target.append(float(contact.dist))
    return {
        "floor_contact_count": len(floor_distances),
        "self_contact_count": len(self_distances),
        "minimum_floor_distance_m": min(floor_distances, default=float("nan")),
        "minimum_self_distance_m": min(self_distances, default=float("nan")),
    }


def forced_single_event(
    segment: int = 9, start_s: float = 3.0, duration_s: float = 0.10
) -> list[ErasureEvent]:
    if not 0 <= segment < N_LEG_PAIRS:
        raise ValueError(f"segment must lie in [0,{N_LEG_PAIRS - 1}]")
    if not 0.0 < duration_s <= SUPPORT_WINDOW_S:
        raise ValueError("forced duration must lie in (0, support_window]")
    return [
        ErasureEvent(
            segment=segment,
            channel_use=-1,
            window_start_s=start_s,
            start_s=start_s,
            end_s=start_s + duration_s,
            duration_s=duration_s,
            duration_fraction=duration_s / SUPPORT_WINDOW_S,
            trigger_uniform=0.0,
        )
    ]


def run_trial(
    config: TrialConfig,
    model_path: Path = MODEL_PATH,
    render_path: Path | None = None,
    render_fps: int = 30,
    events_override: list[ErasureEvent] | None = None,
) -> dict[str, float | int | str | bool]:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    index = ModelIndex(model)
    controller = make_farms_controller(model, config.wave)
    if events_override is None:
        events, opportunities = build_events(config)
    else:
        events = sorted(events_override, key=lambda event: event.start_s)
        opportunities = N_LEG_PAIRS * max(
            int(round((config.duration_s - config.eval_start_s) / SUPPORT_WINDOW_S)), 0
        )
    erasure = ErasureManager(model, index, events)
    groups = phase_groups(config.wave)
    group_blackout_time = np.zeros(len(groups), dtype=float)
    any_group_blackout_time = 0.0
    segment_erased_time = np.zeros(N_LEG_PAIRS, dtype=float)

    renderer = None
    writer = None
    camera = None
    next_frame_t = 0.0
    if render_path is not None:
        os.environ.setdefault("MUJOCO_GL", "wgl")
        render_path.parent.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(model, height=720, width=1280)
        writer = imageio.get_writer(
            render_path, fps=render_fps, codec="libx264", quality=8, macro_block_size=None
        )
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance = 0.19
        camera.azimuth = 132.0
        camera.elevation = -30.0

    mujoco.mj_resetData(model, data)
    # Match the frozen controller's nonzero knee DC target at reset.  Starting
    # every knee at zero would create an artificial all-leg torque impulse at
    # t=0 that is unrelated to locomotion or contact erasure.
    knee_positions = index.leg_positions[:, :, 3].reshape(-1)
    data.qpos[index.qpos_addresses[knee_positions]] = math.pi / 6.0
    mujoco.mj_forward(model, data)
    dt = float(model.opt.timestep)
    measurement_start: np.ndarray | None = None
    measurement_end: np.ndarray | None = None
    yaw_start: float | None = None
    yaw_end: float | None = None
    previous_sample: np.ndarray | None = None
    path_length = 0.0
    mechanical_work = 0.0
    min_chassis_z = float("inf")
    finite = True
    max_abs_root_roll_deg = 0.0
    max_abs_root_pitch_deg = 0.0
    max_segment_tilt_deg = 0.0
    max_body_pitch_joint_deg = 0.0
    max_body_roll_joint_deg = 0.0
    minimum_dynamic_floor_distance_m = float("inf")
    maximum_dynamic_self_contact_count = 0
    first_failure_time_s = float("nan")
    failure_kind = ""
    next_metric_t = config.eval_start_s

    try:
        while data.time < config.duration_s - 0.5 * dt:
            t = float(data.time)
            active = erasure.update(t)
            controller.step(model, data, t)
            torque = data.ctrl[index.active_actuator_ids]
            qdot = data.qvel[index.dof_addresses]
            if t >= config.eval_start_s:
                mechanical_work += float(np.sum(np.abs(torque * qdot))) * dt
                segment_erased_time += active.astype(float) * dt
                blackout = np.asarray([bool(np.all(active[group])) for group in groups])
                group_blackout_time += blackout.astype(float) * dt
                if np.any(blackout):
                    any_group_blackout_time += dt
            time_before_step = float(data.time)
            mujoco.mj_step(model, data)

            if (
                float(data.time) <= time_before_step
                or not np.isfinite(data.qpos).all()
                or not np.isfinite(data.qvel).all()
            ):
                finite = False
                first_failure_time_s = float(data.time)
                failure_kind = "nonfinite_or_time_stall"
                break

            if data.time + 1e-12 >= next_metric_t:
                center = chassis_centroid(data, index)
                min_chassis_z = min(min_chassis_z, float(np.min(data.xpos[index.chassis_body_ids, 2])))
                rotations = data.xmat[index.chassis_body_ids].reshape(-1, 3, 3)
                root_roll, root_pitch = euler_roll_pitch(rotations[0])
                root_roll_deg = math.degrees(root_roll)
                root_pitch_deg = math.degrees(root_pitch)
                segment_tilts = np.asarray([tilt_deg(rotation) for rotation in rotations])
                nonhead_pitch = data.qpos[index.qpos_addresses[index.pitch_positions[1:]]]
                spine_roll = data.qpos[index.qpos_addresses[index.roll_positions]]
                contacts = contact_summary(model, data)
                floor_distance = float(contacts["minimum_floor_distance_m"])
                max_abs_root_roll_deg = max(max_abs_root_roll_deg, abs(root_roll_deg))
                max_abs_root_pitch_deg = max(max_abs_root_pitch_deg, abs(root_pitch_deg))
                max_segment_tilt_deg = max(max_segment_tilt_deg, float(np.max(segment_tilts)))
                max_body_pitch_joint_deg = max(
                    max_body_pitch_joint_deg,
                    float(np.max(np.abs(np.degrees(nonhead_pitch)))),
                )
                max_body_roll_joint_deg = max(
                    max_body_roll_joint_deg,
                    float(np.max(np.abs(np.degrees(spine_roll)))),
                )
                if math.isfinite(floor_distance):
                    minimum_dynamic_floor_distance_m = min(
                        minimum_dynamic_floor_distance_m, floor_distance
                    )
                maximum_dynamic_self_contact_count = max(
                    maximum_dynamic_self_contact_count,
                    int(contacts["self_contact_count"]),
                )
                if math.isnan(first_failure_time_s):
                    checks = (
                        (abs(root_roll_deg) > ROOT_TILT_LIMIT_DEG, "root_roll"),
                        (abs(root_pitch_deg) > ROOT_TILT_LIMIT_DEG, "root_pitch"),
                        (float(np.max(segment_tilts)) > SEGMENT_TILT_LIMIT_DEG, "segment_tilt"),
                        (float(np.max(np.abs(np.degrees(nonhead_pitch)))) > SPINE_JOINT_LIMIT_DEG, "body_pitch_joint"),
                        (float(np.max(np.abs(np.degrees(spine_roll)))) > SPINE_JOINT_LIMIT_DEG, "body_roll_joint"),
                        (min_chassis_z <= 0.0, "chassis_below_floor"),
                        (
                            math.isfinite(floor_distance)
                            and floor_distance < -DYNAMIC_PENETRATION_LIMIT_M,
                            "floor_penetration",
                        ),
                        (int(contacts["self_contact_count"]) > 0, "self_contact"),
                    )
                    for failed, kind in checks:
                        if failed:
                            first_failure_time_s = float(data.time)
                            failure_kind = kind
                            break
                if measurement_start is None:
                    measurement_start = center.copy()
                    yaw_start = head_yaw(data, index)
                if previous_sample is not None:
                    path_length += float(np.linalg.norm(center[:2] - previous_sample[:2]))
                previous_sample = center.copy()
                measurement_end = center.copy()
                yaw_end = head_yaw(data, index)
                next_metric_t += 0.01

            if renderer is not None and writer is not None and data.time + 1e-12 >= next_frame_t:
                center = chassis_centroid(data, index)
                camera.lookat[:] = (float(center[0]), float(center[1]), 0.015)
                renderer.update_scene(data, camera=camera)
                frame = renderer.render()
                canvas = Image.fromarray(frame)
                draw = ImageDraw.Draw(canvas)
                active_ids = np.flatnonzero(active).tolist()
                noise_label = (
                    "single-event diagnostic"
                    if events_override is not None
                    else f"q={config.q:.2f}"
                )
                label = (
                    f"w={config.wave}   {noise_label}   "
                    f"erased segments: {active_ids if active_ids else 'none'}"
                )
                draw.rectangle((20, 18, 650, 58), fill=(22, 27, 34))
                indicator = (235, 52, 52) if active_ids else (95, 108, 122)
                draw.ellipse((32, 29, 48, 45), fill=indicator)
                draw.text((60, 29), label, fill=(245, 247, 250))
                writer.append_data(np.asarray(canvas))
                next_frame_t += 1.0 / render_fps
    finally:
        erasure.restore()
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()

    if measurement_start is None or measurement_end is None or yaw_start is None or yaw_end is None:
        finite = False
        measurement_start = np.full(3, np.nan)
        measurement_end = np.full(3, np.nan)
        yaw_start = float("nan")
        yaw_end = float("nan")

    directed_displacement = FORWARD_SIGN * float(measurement_end[0] - measurement_start[0])
    lateral_displacement = float(measurement_end[1] - measurement_start[1])
    yaw_drift = unwrap_difference(yaw_end, yaw_start)
    evaluation_duration = max(config.duration_s - config.eval_start_s, 1e-12)
    realized_erased_load = float(segment_erased_time.sum()) / (
        N_LEG_PAIRS * evaluation_duration
    )
    expected_load = config.q * expected_burst_fraction(config.minimum_burst_fraction)
    sizes = [len(group) for group in groups]
    if not math.isfinite(minimum_dynamic_floor_distance_m):
        minimum_dynamic_floor_distance_m = float("nan")
    stable = bool(
        finite
        and max_abs_root_roll_deg <= ROOT_TILT_LIMIT_DEG
        and max_abs_root_pitch_deg <= ROOT_TILT_LIMIT_DEG
        and max_segment_tilt_deg <= SEGMENT_TILT_LIMIT_DEG
        and max_body_pitch_joint_deg <= SPINE_JOINT_LIMIT_DEG
        and max_body_roll_joint_deg <= SPINE_JOINT_LIMIT_DEG
        and min_chassis_z > 0.0
        and (
            not math.isfinite(minimum_dynamic_floor_distance_m)
            or minimum_dynamic_floor_distance_m >= -DYNAMIC_PENETRATION_LIMIT_M
        )
        and maximum_dynamic_self_contact_count == 0
    )

    return {
        "model": str(model_path),
        "wave": config.wave,
        "phase_group_count": len(groups),
        "m_min": min(sizes),
        "m_max": max(sizes),
        "m_mean": N_LEG_PAIRS / len(groups),
        "q": config.q,
        "seed": config.seed,
        "duration_s": config.duration_s,
        "eval_start_s": config.eval_start_s,
        "support_window_s": SUPPORT_WINDOW_S,
        "minimum_burst_fraction": config.minimum_burst_fraction,
        "expected_burst_fraction": expected_burst_fraction(config.minimum_burst_fraction),
        "expected_erased_load": expected_load,
        "channel_opportunities": opportunities,
        "triggered_events": len(events),
        "realized_trigger_rate": len(events) / max(opportunities, 1),
        "realized_erased_load": realized_erased_load,
        "mean_burst_duration_s": float(np.mean([event.duration_s for event in events])) if events else 0.0,
        "max_burst_duration_s": float(np.max([event.duration_s for event in events])) if events else 0.0,
        "absolute_forward_displacement_m": directed_displacement,
        "lateral_displacement_m": lateral_displacement,
        "path_length_m": path_length,
        "path_efficiency": directed_displacement / path_length if path_length > 1e-12 else 0.0,
        "yaw_drift_rad": yaw_drift,
        "mechanical_work_j": mechanical_work,
        "min_chassis_z_m": min_chassis_z,
        "max_abs_root_roll_deg": max_abs_root_roll_deg,
        "max_abs_root_pitch_deg": max_abs_root_pitch_deg,
        "max_segment_tilt_deg": max_segment_tilt_deg,
        "max_body_pitch_joint_deg": max_body_pitch_joint_deg,
        "max_body_roll_joint_deg": max_body_roll_joint_deg,
        "minimum_dynamic_floor_distance_m": minimum_dynamic_floor_distance_m,
        "maximum_dynamic_self_contact_count": maximum_dynamic_self_contact_count,
        "first_failure_time_s": first_failure_time_s,
        "failure_kind": failure_kind,
        "stable": stable,
        "any_group_blackout_fraction": any_group_blackout_time / evaluation_duration,
        "mean_group_blackout_fraction": float(np.mean(group_blackout_time)) / evaluation_duration,
        "max_group_blackout_fraction": float(np.max(group_blackout_time)) / evaluation_duration,
        "finite": finite,
        "video": str(render_path) if render_path is not None else "",
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize(raw_rows: list[dict]) -> list[dict]:
    baselines = {
        int(row["wave"]): float(row["absolute_forward_displacement_m"])
        for row in raw_rows
        if float(row["q"]) == 0.0
    }
    summary: list[dict] = []
    for wave in sorted({int(row["wave"]) for row in raw_rows}):
        for q in sorted({float(row["q"]) for row in raw_rows if int(row["wave"]) == wave}):
            rows = [row for row in raw_rows if int(row["wave"]) == wave and float(row["q"]) == q]
            displacement = np.asarray(
                [float(row["absolute_forward_displacement_m"]) for row in rows], dtype=float
            )
            baseline = baselines[wave]
            retention = displacement / baseline if abs(baseline) > 1e-12 else np.full_like(displacement, np.nan)
            stable = np.asarray(
                [str(row["stable"]).lower() == "true" for row in rows], dtype=bool
            )
            distance_success_02 = np.abs(displacement - baseline) <= 0.02 * abs(baseline)
            distance_success_05 = np.abs(displacement - baseline) <= 0.05 * abs(baseline)
            distance_success = np.abs(displacement - baseline) <= 0.10 * abs(baseline)
            reliable_02 = stable & distance_success_02
            reliable_05 = stable & distance_success_05
            reliable = stable & distance_success
            template = rows[0]
            summary.append(
                {
                    "wave": wave,
                    "phase_group_count": int(template["phase_group_count"]),
                    "m_min": int(template["m_min"]),
                    "m_max": int(template["m_max"]),
                    "m_mean": float(template["m_mean"]),
                    "q": q,
                    "n": len(rows),
                    "baseline_displacement_m": baseline,
                    "mean_displacement_m": float(np.mean(displacement)),
                    "sd_displacement_m": float(np.std(displacement, ddof=1)) if len(rows) > 1 else 0.0,
                    "mean_retention": float(np.nanmean(retention)),
                    "sd_retention": float(np.nanstd(retention, ddof=1)) if len(rows) > 1 else 0.0,
                    "stable_fraction": float(np.mean(stable)),
                    "reliability_02pct": float(np.mean(reliable_02)),
                    "reliability_05pct": float(np.mean(reliable_05)),
                    "distance_success_10pct": float(np.mean(distance_success)),
                    "reliability_10pct": float(np.mean(reliable)),
                    "mean_realized_erased_load": float(
                        np.mean([float(row["realized_erased_load"]) for row in rows])
                    ),
                    "mean_any_group_blackout_fraction": float(
                        np.mean([float(row["any_group_blackout_fraction"]) for row in rows])
                    ),
                    "predicted_any_group_blackout_fraction": predicted_any_group_blackout(wave, q),
                    "mean_work_j": float(np.mean([float(row["mechanical_work_j"]) for row in rows])),
                }
            )
    return summary


def style_axis(axis, xlabel: str, ylabel: str) -> None:
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=0.2, linewidth=0.7)


def make_figures(summary_rows: list[dict], raw_rows: list[dict]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    colors = plt.get_cmap("viridis")(np.linspace(0.12, 0.9, len(DEFAULT_WAVES)))
    labels = {}
    for row in summary_rows:
        labels[int(row["wave"])] = (
            f"w={int(row['wave'])}, G={int(row['phase_group_count'])}, "
            f"m={int(row['m_min'])}–{int(row['m_max'])}"
        )

    def curves(y_key: str, sd_key: str | None, ylabel: str, filename: str, ylim=None):
        fig, axis = plt.subplots(figsize=(6.3, 4.2), constrained_layout=True)
        for color, wave in zip(colors, DEFAULT_WAVES):
            rows = sorted((row for row in summary_rows if int(row["wave"]) == wave), key=lambda r: float(r["q"]))
            x = np.asarray([float(row["q"]) for row in rows])
            y = np.asarray([float(row[y_key]) for row in rows])
            if sd_key is None:
                axis.plot(x, y, "o-", color=color, linewidth=1.8, markersize=4, label=labels[wave])
            else:
                sd = np.asarray([float(row[sd_key]) for row in rows])
                axis.errorbar(
                    x, y, yerr=sd, fmt="o-", color=color, capsize=2.5,
                    linewidth=1.6, markersize=4, label=labels[wave]
                )
        style_axis(axis, "Event trigger probability q", ylabel)
        if ylim is not None:
            axis.set_ylim(*ylim)
        axis.legend(frameon=False, fontsize=8)
        fig.savefig(FIGURE_DIR / filename, dpi=220)
        plt.close(fig)

    curves(
        "mean_displacement_m", "sd_displacement_m", "Forward displacement (m)",
        "absolute_displacement_vs_q.png"
    )
    curves("mean_retention", "sd_retention", "Displacement / matched q=0 baseline", "retention_vs_q.png")
    curves("stable_fraction", None, "Stable-trial fraction", "stability_vs_q.png", (-0.03, 1.03))
    curves("reliability_02pct", None, "Reliability (±2% of matched baseline)", "reliability_02pct_vs_q.png", (-0.03, 1.03))
    curves("reliability_05pct", None, "Reliability (±5% of matched baseline)", "reliability_05pct_vs_q.png", (-0.03, 1.03))
    curves("reliability_10pct", None, "Reliability (±10% of matched baseline)", "reliability_vs_q.png", (-0.03, 1.03))
    curves(
        "mean_any_group_blackout_fraction", None, "Time with any phase group fully erased",
        "group_blackout_vs_q.png", (-0.002, None)
    )

    fig, axis = plt.subplots(figsize=(6.3, 4.2), constrained_layout=True)
    for color, wave in zip(colors, DEFAULT_WAVES):
        rows = [row for row in raw_rows if int(row["wave"]) == wave]
        axis.scatter(
            [float(row["realized_erased_load"]) for row in rows],
            [float(row["absolute_forward_displacement_m"]) for row in rows],
            s=18, alpha=0.65, color=color, label=labels[wave]
        )
    style_axis(axis, "Realized erased-time fraction", "Forward displacement (m)")
    axis.legend(frameon=False, fontsize=8)
    fig.savefig(FIGURE_DIR / "displacement_vs_realized_erased_load.png", dpi=220)
    plt.close(fig)

    structure = []
    for wave in DEFAULT_WAVES:
        groups = phase_groups(wave)
        structure.append((wave, len(groups), min(map(len, groups)), max(map(len, groups)), N_LEG_PAIRS / len(groups)))
    fig, axis = plt.subplots(figsize=(6.3, 3.8), constrained_layout=True)
    axis.plot([row[0] for row in structure], [row[4] for row in structure], "o-", color="#1f6f8b")
    for wave, groups, minimum, maximum, mean in structure:
        axis.annotate(f"G={groups}\n{minimum}–{maximum}", (wave, mean), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    axis.set_xticks([row[0] for row in structure])
    style_axis(axis, "Spatial wave number w", "Mean homologous copies 19/G")
    fig.savefig(FIGURE_DIR / "wave_number_vs_redundancy.png", dpi=220)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(5.2, 4.5), constrained_layout=True)
    for color, wave in zip(colors, DEFAULT_WAVES):
        rows = [row for row in summary_rows if int(row["wave"]) == wave and float(row["q"]) > 0.0]
        axis.scatter(
            [float(row["predicted_any_group_blackout_fraction"]) for row in rows],
            [float(row["mean_any_group_blackout_fraction"]) for row in rows],
            s=34, color=color, label=labels[wave], alpha=0.85,
        )
    limit = max(
        [float(row["predicted_any_group_blackout_fraction"]) for row in summary_rows]
        + [float(row["mean_any_group_blackout_fraction"]) for row in summary_rows]
    )
    axis.plot([0.0, limit], [0.0, limit], "--", color="#666666", linewidth=1.2)
    style_axis(axis, "Independent snapshot prediction", "Measured blackout-time fraction")
    axis.set_xlim(-0.015 * limit, 1.03 * limit)
    axis.set_ylim(-0.015 * limit, 1.03 * limit)
    axis.legend(frameon=False, fontsize=8)
    fig.savefig(FIGURE_DIR / "group_blackout_theory_vs_measured.png", dpi=220)
    plt.close(fig)


def run_ab_audit() -> list[dict]:
    rows = []
    for label, path in (("original", ORIGINAL_MODEL_PATH), ("refined", MODEL_PATH)):
        result = run_trial(TrialConfig(wave=3, q=0.0, seed=0, duration_s=6.0, eval_start_s=2.0), path)
        result["variant"] = label
        rows.append(result)
        print(
            f"A/B {label}: D={result['absolute_forward_displacement_m']:.4f} m, "
            f"zmin={result['min_chassis_z_m']:.4f} m, finite={result['finite']}"
        )
    write_csv(DATA_DIR / "model_ab_audit.csv", rows)
    fig, axis = plt.subplots(figsize=(4.5, 3.7), constrained_layout=True)
    axis.bar(
        [row["variant"] for row in rows],
        [float(row["absolute_forward_displacement_m"]) for row in rows],
        color=["#999999", "#2878b5"],
    )
    style_axis(axis, "Model", "Forward displacement (m), 2–6 s")
    fig.savefig(FIGURE_DIR / "original_vs_refined_baseline.png", dpi=220)
    plt.close(fig)
    return rows


def run_sweep(
    waves: Iterable[int], q_values: Iterable[float], seeds: int, workers: int = 4
) -> tuple[list[dict], list[dict]]:
    raw: list[dict] = []
    started = time.time()
    cells = [(wave, q) for wave in waves for q in q_values]
    trial_count = sum(1 if q == 0.0 else seeds for _, q in cells)
    completed = 0
    configs: list[TrialConfig] = []
    for wave, q in cells:
        cell_seeds = [0] if q == 0.0 else list(range(seeds))
        for seed in cell_seeds:
            configs.append(TrialConfig(wave=wave, q=q, seed=seed))

    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(run_trial, config): config for config in configs}
        for future in concurrent.futures.as_completed(futures):
            config = futures[future]
            result = future.result()
            raw.append(result)
            completed += 1
            elapsed = time.time() - started
            print(
                f"[{completed:03d}/{trial_count:03d}] w={config.wave} q={config.q:.2f} "
                f"seed={config.seed} "
                f"D={result['absolute_forward_displacement_m']:.4f} m "
                f"load={result['realized_erased_load']:.3f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
    write_csv(DATA_DIR / "short_erasure_trials.csv", raw)
    summary = summarize(raw)
    write_csv(DATA_DIR / "short_erasure_summary.csv", summary)
    make_figures(summary, raw)
    return raw, summary


def render_demos() -> list[dict]:
    demos = [
        (3, 0.0, 0, "baseline_w3_q000.mp4"),
        (1, 0.3, 2, "short_erasure_w1_q030.mp4"),
        (2, 0.3, 2, "short_erasure_w2_q030.mp4"),
        (3, 0.3, 2, "short_erasure_w3_q030.mp4"),
        (4, 0.3, 2, "short_erasure_w4_q030.mp4"),
    ]
    rows = []
    forced_path = VIDEO_DIR / "single_erasure_w3_segment09_100ms.mp4"
    print(f"Rendering {forced_path.name} ...", flush=True)
    forced = run_trial(
        TrialConfig(wave=3, q=0.0, seed=0),
        render_path=forced_path,
        events_override=forced_single_event(),
    )
    forced["demo_kind"] = "forced_single_100ms_erasure"
    rows.append(forced)
    for wave, q, seed, filename in demos:
        path = VIDEO_DIR / filename
        print(f"Rendering {filename} ...", flush=True)
        rows.append(run_trial(TrialConfig(wave=wave, q=q, seed=seed), render_path=path))
    write_csv(DATA_DIR / "demo_manifest.csv", rows)
    return rows


def write_manifest() -> None:
    groups = {}
    for wave in DEFAULT_WAVES:
        value = phase_groups(wave)
        groups[str(wave)] = {
            "G": len(value),
            "groups": value,
            "sizes": [len(group) for group in value],
            "m_mean": N_LEG_PAIRS / len(value),
        }
    controller_config = yaml.safe_load(CONTROLLER_CONFIG.read_text(encoding="utf-8"))
    manifest = {
        "model": str(MODEL_PATH),
        "model_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        "model_refinement": {
            "source": str(ORIGINAL_MODEL_PATH),
            "timestep_s": 0.0005,
            "integrator": "RK4",
            "numerical_armature_kg_m2": 1.0e-4,
            "active_joint_passive_stiffness": 0.0,
            "active_joint_passive_damping": 0.0,
            "passive_leg_roll_and_foot_terms": "preserved",
        },
        "morphology": {"leg_pairs": 19, "chassis_links_for_metric": 21, "active_joints": 211},
        "fixed_controller": {
            "implementation": str(CONTROLLER_DIR / "impedance_controller.py"),
            "config": str(CONTROLLER_CONFIG),
            "config_sha256": hashlib.sha256(CONTROLLER_CONFIG.read_bytes()).hexdigest(),
            "config_snapshot": controller_config,
            "varied_parameter": "body_wave.wave_number only",
        },
        "channel": {
            "independent_only": True,
            "support_window_s": SUPPORT_WINDOW_S,
            "minimum_burst_fraction": 0.02,
            "duration_distribution": "log-uniform on [0.02, 1] times support window",
            "expected_duration_fraction": expected_burst_fraction(0.02),
            "start_distribution": "uniform conditional on duration within each support window",
            "spatial_object": "both left and right complete leg subtrees of one segment-pair",
        },
        "phase_groups": groups,
        "protocol": {
            "waves": list(DEFAULT_WAVES),
            "q_values": list(DEFAULT_Q),
            "nonzero_q_seeds": 8,
            "trial_duration_s": 8.0,
            "evaluation_start_s": 2.0,
            "channel_uses_per_segment": 12,
            "channel_opportunities_per_trial": 228,
        },
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def parse_list(text: str, conversion):
    return tuple(conversion(value.strip()) for value in text.split(",") if value.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    trial = sub.add_parser("trial")
    trial.add_argument("--wave", type=int, default=3)
    trial.add_argument("--q", type=float, default=0.0)
    trial.add_argument("--seed", type=int, default=0)
    trial.add_argument("--duration", type=float, default=6.0)
    trial.add_argument("--eval-start", type=float, default=2.0)
    trial.add_argument("--video", type=Path)
    smoke = sub.add_parser("smoke-erasure")
    smoke.add_argument("--wave", type=int, default=3)
    smoke.add_argument("--segment", type=int, default=9)
    smoke.add_argument("--start", type=float, default=3.0)
    smoke.add_argument("--burst-ms", type=float, default=100.0)
    smoke.add_argument("--duration", type=float, default=6.0)
    smoke.add_argument("--video", type=Path, default=VIDEO_DIR / "single_erasure_smoke.mp4")
    sweep = sub.add_parser("sweep")
    sweep.add_argument("--waves", default=",".join(map(str, DEFAULT_WAVES)))
    sweep.add_argument("--q", default=",".join(map(str, DEFAULT_Q)))
    sweep.add_argument("--seeds", type=int, default=8)
    sweep.add_argument("--workers", type=int, default=4)
    sub.add_parser("ab-audit")
    sub.add_parser("demos")
    sub.add_parser("replot")
    sub.add_parser("all")
    args = parser.parse_args()

    for directory in (DATA_DIR, FIGURE_DIR, VIDEO_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    write_manifest()

    if args.command == "trial":
        result = run_trial(
            TrialConfig(
                wave=args.wave,
                q=args.q,
                seed=args.seed,
                duration_s=args.duration,
                eval_start_s=args.eval_start,
            ),
            render_path=args.video,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "smoke-erasure":
        result = run_trial(
            TrialConfig(
                wave=args.wave,
                q=0.0,
                seed=0,
                duration_s=args.duration,
                eval_start_s=2.0,
            ),
            render_path=args.video,
            events_override=forced_single_event(
                segment=args.segment,
                start_s=args.start,
                duration_s=args.burst_ms / 1000.0,
            ),
        )
        print(json.dumps(result, indent=2))
    elif args.command == "sweep":
        run_sweep(
            parse_list(args.waves, int), parse_list(args.q, float), args.seeds, args.workers
        )
    elif args.command == "ab-audit":
        run_ab_audit()
    elif args.command == "demos":
        render_demos()
    elif args.command == "replot":
        raw = read_csv(DATA_DIR / "short_erasure_trials.csv")
        summary = summarize(raw)
        write_csv(DATA_DIR / "short_erasure_summary.csv", summary)
        make_figures(summary, raw)
    elif args.command == "all":
        run_ab_audit()
        run_sweep(DEFAULT_WAVES, DEFAULT_Q, 8, 4)
        render_demos()


if __name__ == "__main__":
    main()
