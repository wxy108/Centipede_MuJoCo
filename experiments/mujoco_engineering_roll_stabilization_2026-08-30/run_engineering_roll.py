"""Engineering body-roll stabilization for the abstract Centipede Model.

The accepted locked-head plant is immutable input.  This experiment keeps
mass, inertia, armature, terrain, passive body/tip compliance, body/leg wave
number, gait amplitudes, and relative phase frozen.  It adds only explicit,
bounded feedback paths:

1. projected-gravity/roll-rate feedback is converted to a left/right
   differential leg-pitch reference, so foot-ground reactions can generate
   the external restoring moment missing from a free-root passive spine; and
2. an optional torque-limited zero-target body-roll servo can add local active
   muscle-like impedance without changing the MJCF plant.

Centipede Robot Step 18/19 files are read only as hash sentinels.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


# MuJoCo's optional plugin DLL scan can block on this Windows host.  These
# models use no plugin instances, so only optional plugin handles are skipped.
_ORIGINAL_CDLL = ctypes.CDLL


def _cdll_without_optional_mujoco_plugins(path, *args, **kwargs):
    normalized = os.path.normcase(str(path))
    if "site-packages\\mujoco\\plugin" in normalized:
        return None
    return _ORIGINAL_CDLL(path, *args, **kwargs)


ctypes.CDLL = _cdll_without_optional_mujoco_plugins
try:
    import mujoco
finally:
    ctypes.CDLL = _ORIGINAL_CDLL


ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT.parents[1]
TIP_SCRIPT = (
    MODEL_ROOT
    / "experiments"
    / "mujoco_template_tip_compliance_2026-08-27"
    / "run_tip_compliance.py"
)
LOCKED_ROOT = (
    MODEL_ROOT / "experiments" / "mujoco_template_locked_head_2026-08-27"
)
MODELS = {
    "flat": LOCKED_ROOT / "models" / "centipede_flat_locked_head_m29p0.xml",
    "rough": LOCKED_ROOT / "models" / "centipede_rough_locked_head_m29p0.xml",
}
OUTPUTS = ROOT / "outputs"
DATA = OUTPUTS / "data"
FIGURES = OUTPUTS / "figures"
VIDEOS = OUTPUTS / "videos"

SEGMENTS = 19
EVAL_START_S = 1.5
FILTER_TAU_S = 0.10
RATE_FILTER_TAU_S = 0.06
FEEDBACK_START_S = 0.70
FEEDBACK_RAMP_S = 0.80
ROLL_TORQUE_LIMIT_NM = 4.0e-5


# The first screen brackets the feedback sign and the minimum useful attitude
# gain.  Kd has units of seconds because the correction is a position target.
# Active roll Kp/Kv are additional controller paths; the frozen passive MJCF
# spring/damper remains present and unchanged.
CANDIDATES: dict[str, dict[str, float]] = {
    "baseline": {"angle_gain": 0.0, "rate_gain_s": 0.0, "clip_rad": 0.0,
                 "roll_kp": 0.0, "roll_kv": 0.0},
    "pg_n100": {"angle_gain": -1.0, "rate_gain_s": -0.06, "clip_rad": 0.30,
                "roll_kp": 0.0, "roll_kv": 0.0},
    "pg_p050": {"angle_gain": 0.5, "rate_gain_s": 0.03, "clip_rad": 0.20,
                "roll_kp": 0.0, "roll_kv": 0.0},
    "pg_p100": {"angle_gain": 1.0, "rate_gain_s": 0.06, "clip_rad": 0.30,
                "roll_kp": 0.0, "roll_kv": 0.0},
    "pg_p150": {"angle_gain": 1.5, "rate_gain_s": 0.08, "clip_rad": 0.35,
                "roll_kp": 0.0, "roll_kv": 0.0},
    "pg_p200": {"angle_gain": 2.0, "rate_gain_s": 0.10, "clip_rad": 0.40,
                "roll_kp": 0.0, "roll_kv": 0.0},
    "pg_p250": {"angle_gain": 2.5, "rate_gain_s": 0.12, "clip_rad": 0.42,
                "roll_kp": 0.0, "roll_kv": 0.0},
    "pg_p300": {"angle_gain": 3.0, "rate_gain_s": 0.14, "clip_rad": 0.48,
                "roll_kp": 0.0, "roll_kv": 0.0},
    "pg_p350": {"angle_gain": 3.5, "rate_gain_s": 0.16, "clip_rad": 0.52,
                "roll_kp": 0.0, "roll_kv": 0.0},
    "pg_p250_l050": {"angle_gain": 2.5, "rate_gain_s": 0.12,
                     "local_gain": 0.5, "clip_rad": 0.48,
                     "roll_kp": 0.0, "roll_kv": 0.0},
    "pg_p300_l050": {"angle_gain": 3.0, "rate_gain_s": 0.14,
                     "local_gain": 0.5, "clip_rad": 0.52,
                     "roll_kp": 0.0, "roll_kv": 0.0},
    "pg_p300_l100": {"angle_gain": 3.0, "rate_gain_s": 0.14,
                     "local_gain": 1.0, "clip_rad": 0.52,
                     "roll_kp": 0.0, "roll_kv": 0.0},
    "pg_p300_l125": {"angle_gain": 3.0, "rate_gain_s": 0.14,
                     "local_gain": 1.25, "clip_rad": 0.52,
                     "roll_kp": 0.0, "roll_kv": 0.0},
    "pg_p275_l125": {"angle_gain": 2.75, "rate_gain_s": 0.13,
                     "local_gain": 1.25, "clip_rad": 0.50,
                     "roll_kp": 0.0, "roll_kv": 0.0},
    "pg_p300d008_l125": {"angle_gain": 3.0, "rate_gain_s": 0.08,
                         "local_gain": 1.25, "clip_rad": 0.50,
                         "roll_kp": 0.0, "roll_kv": 0.0},
    "pg_p300_l150": {"angle_gain": 3.0, "rate_gain_s": 0.14,
                     "local_gain": 1.5, "clip_rad": 0.52,
                     "roll_kp": 0.0, "roll_kv": 0.0},
    "pg_p350_l150_cg20": {"angle_gain": 3.5, "rate_gain_s": 0.16,
                          "local_gain": 1.5, "clip_rad": 0.52,
                          "contact_gate_min": 0.20,
                          "roll_kp": 0.0, "roll_kv": 0.0},
    "pg_p340_l145_cg20": {"angle_gain": 3.4, "rate_gain_s": 0.155,
                          "local_gain": 1.45, "clip_rad": 0.52,
                          "contact_gate_min": 0.20,
                          "roll_kp": 0.0, "roll_kv": 0.0},
    "pg_p330_l140_cg20": {"angle_gain": 3.3, "rate_gain_s": 0.15,
                          "local_gain": 1.4, "clip_rad": 0.52,
                          "contact_gate_min": 0.20,
                          "roll_kp": 0.0, "roll_kv": 0.0},
    "pg_p350_l150_cg35": {"angle_gain": 3.5, "rate_gain_s": 0.16,
                          "local_gain": 1.5, "clip_rad": 0.52,
                          "contact_gate_min": 0.35,
                          "roll_kp": 0.0, "roll_kv": 0.0},
    "pg_p400_l175_cg20": {"angle_gain": 4.0, "rate_gain_s": 0.18,
                          "local_gain": 1.75, "clip_rad": 0.55,
                          "contact_gate_min": 0.20,
                          "roll_kp": 0.0, "roll_kv": 0.0},
    "pg_p350_l150_cg20_tg25": {"angle_gain": 3.5, "rate_gain_s": 0.16,
                               "local_gain": 1.5, "clip_rad": 0.52,
                               "contact_gate_min": 0.20,
                               "tip_guard_start_deg": 25.0,
                               "roll_kp": 0.0, "roll_kv": 0.0},
    "pg_p400_l175_cg20_tg25": {"angle_gain": 4.0, "rate_gain_s": 0.18,
                               "local_gain": 1.75, "clip_rad": 0.55,
                               "contact_gate_min": 0.20,
                               "tip_guard_start_deg": 25.0,
                               "roll_kp": 0.0, "roll_kv": 0.0},
    "pg_p150_r05": {"angle_gain": 1.5, "rate_gain_s": 0.08, "clip_rad": 0.35,
                    "roll_kp": 3.5e-4, "roll_kv": 1.5e-6},
    "pg_p200_r10": {"angle_gain": 2.0, "rate_gain_s": 0.10, "clip_rad": 0.40,
                    "roll_kp": 7.0e-4, "roll_kv": 2.2e-6},
    "pg_p250_l075_r20": {"angle_gain": 2.5, "rate_gain_s": 0.12,
                         "local_gain": 0.75, "clip_rad": 0.48,
                         "roll_kp": 1.4e-3, "roll_kv": 4.0e-6},
    "pg_p250_l075_r40": {"angle_gain": 2.5, "rate_gain_s": 0.12,
                         "local_gain": 0.75, "clip_rad": 0.48,
                         "roll_kp": 2.8e-3, "roll_kv": 6.0e-6},
    "pg_p275_l075_r40": {"angle_gain": 2.75, "rate_gain_s": 0.13,
                         "local_gain": 0.75, "clip_rad": 0.50,
                         "roll_kp": 2.8e-3, "roll_kv": 6.0e-6},
    "pg_p275_l125_r40": {"angle_gain": 2.75, "rate_gain_s": 0.13,
                         "local_gain": 1.25, "clip_rad": 0.52,
                         "roll_kp": 2.8e-3, "roll_kv": 6.0e-6},
    "pg_p300_l125_r40": {"angle_gain": 3.0, "rate_gain_s": 0.14,
                         "local_gain": 1.25, "clip_rad": 0.52,
                         "roll_kp": 2.8e-3, "roll_kv": 6.0e-6},
    "pg_p300_l125_r30": {"angle_gain": 3.0, "rate_gain_s": 0.14,
                         "local_gain": 1.25, "clip_rad": 0.52,
                         "roll_kp": 2.8e-3, "roll_kv": 6.0e-6,
                         "roll_torque_limit": 3.0e-5},
    "pg_p325_l130_r30": {"angle_gain": 3.25, "rate_gain_s": 0.145,
                         "local_gain": 1.3, "clip_rad": 0.52,
                         "roll_kp": 2.8e-3, "roll_kv": 6.0e-6,
                         "roll_torque_limit": 3.0e-5},
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tip = load_module(TIP_SCRIPT, "centipede_engineering_roll_tip")


def ensure_directories() -> None:
    for path in (DATA, FIGURES, VIDEOS):
        path.mkdir(parents=True, exist_ok=True)


def percentile(values, q: float) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.quantile(array, q)) if array.size else float("nan")


def circular_mean_rad(angles_rad: np.ndarray) -> float:
    angles_rad = np.asarray(angles_rad, dtype=float)
    return float(math.atan2(np.mean(np.sin(angles_rad)), np.mean(np.cos(angles_rad))))


def smooth_ramp(time_s: float) -> float:
    x = np.clip((time_s - FEEDBACK_START_S) / FEEDBACK_RAMP_S, 0.0, 1.0)
    return float(x * x * (3.0 - 2.0 * x))


def roll_actuator_ids(model: mujoco.MjModel, index) -> np.ndarray:
    actuator_for_joint = {
        int(model.actuator_trnid[actuator_id, 0]): actuator_id
        for actuator_id in range(model.nu)
    }
    ids = np.asarray(
        [actuator_for_joint[int(joint_id)] for joint_id in index.roll_joint_ids],
        dtype=int,
    )
    if len(ids) != SEGMENTS or len(np.unique(ids)) != SEGMENTS:
        raise RuntimeError(f"Expected {SEGMENTS} unique body-roll actuators")
    return ids


def run_candidate(
    candidate: str,
    kind: str,
    duration_s: float,
    video: Path | None = None,
    camera_azimuth: float = 90.0,
    camera_elevation: float = -12.0,
) -> dict[str, object]:
    spec = CANDIDATES[candidate]
    model_path = MODELS[kind]
    original_configure = tip.configure_model
    original_controller = tip.LeanController
    original_signed_roll = tip.signed_world_roll_deg

    coherent_abs_deg: list[float] = []
    front_abs_deg: list[float] = []
    frame_world_span_deg: list[float] = []
    correction_time: list[float] = []
    correction_abs_deg: list[float] = []
    roll_torque_time: list[float] = []
    roll_torque_abs_nm: list[float] = []
    roll_ids_holder: dict[str, np.ndarray] = {}

    def configure_stabilized(model, index):
        original_configure(model, index)
        actuator_ids = roll_actuator_ids(model, index)
        roll_ids_holder["ids"] = actuator_ids
        model.actuator_gainprm[actuator_ids] = 0.0
        model.actuator_biasprm[actuator_ids] = 0.0
        model.actuator_gaintype[actuator_ids] = int(mujoco.mjtGain.mjGAIN_FIXED)
        model.actuator_biastype[actuator_ids] = int(mujoco.mjtBias.mjBIAS_NONE)
        model.actuator_gainprm[actuator_ids, 0] = 1.0

    class EngineeringRollController(original_controller):
        def __init__(self, model, index, wave):
            super().__init__(model, index, wave)
            self.filtered_roll = 0.0
            self.filtered_rate = 0.0
            self.previous_filtered_roll: float | None = None
            self.filtered_segment_roll = np.zeros(SEGMENTS, dtype=float)
            self.ground_geom_id = tip.stage1.ground_geom_id(model)
            self.foot_geom_ids = np.asarray(
                [
                    [
                        tip.base.name2id(
                            model,
                            mujoco.mjtObj.mjOBJ_GEOM,
                            f"foot_{segment}_{side}_collision",
                        )
                        for side in range(2)
                    ]
                    for segment in range(SEGMENTS)
                ],
                dtype=int,
            )
            self.foot_geom_lookup = {
                int(geom_id): (segment, side)
                for segment in range(SEGMENTS)
                for side, geom_id in enumerate(self.foot_geom_ids[segment])
            }
            self.filtered_contact_weight = np.zeros((SEGMENTS, 2), dtype=float)

        def step(self, data):
            previous_corrected = (
                None if self.previous_target is None else self.previous_target.copy()
            )
            torque = super().step(data)
            dt = float(self.model.opt.timestep)
            time_s = float(data.time)

            # Mean projected-gravity roll across the articulated chassis is a
            # low-noise common-mode attitude estimate.  Local roll remains a
            # separate measured outcome and is not erased from the plant.
            world_deg = original_signed_roll(data, self.index.chassis_body_ids)
            coherent_roll = circular_mean_rad(np.radians(world_deg))
            alpha = dt / (FILTER_TAU_S + dt)
            self.filtered_roll += alpha * (coherent_roll - self.filtered_roll)
            if self.previous_filtered_roll is None:
                raw_rate = 0.0
            else:
                raw_rate = (self.filtered_roll - self.previous_filtered_roll) / dt
            self.previous_filtered_roll = self.filtered_roll
            rate_alpha = dt / (RATE_FILTER_TAU_S + dt)
            self.filtered_rate += rate_alpha * (raw_rate - self.filtered_rate)
            segment_roll = np.radians(world_deg[1 : 1 + SEGMENTS])
            self.filtered_segment_roll += alpha * (
                segment_roll - self.filtered_segment_roll
            )

            blend = smooth_ramp(time_s)
            common_correction = blend * (
                float(spec["angle_gain"]) * self.filtered_roll
                + float(spec["rate_gain_s"]) * self.filtered_rate
            )
            corrections = common_correction + blend * float(
                spec.get("local_gain", 0.0)
            ) * (self.filtered_segment_roll - self.filtered_roll)
            corrections = np.clip(
                corrections, -float(spec["clip_rad"]), float(spec["clip_rad"])
            )

            contact_gate_min = spec.get("contact_gate_min")
            if contact_gate_min is None:
                contact_weight = np.ones((SEGMENTS, 2), dtype=float)
            else:
                desired = np.full(
                    (SEGMENTS, 2), float(contact_gate_min), dtype=float
                )
                for contact_id in range(data.ncon):
                    contact = data.contact[contact_id]
                    geom1, geom2 = int(contact.geom1), int(contact.geom2)
                    foot_geom = None
                    if geom1 == self.ground_geom_id and geom2 in self.foot_geom_lookup:
                        foot_geom = geom2
                    elif geom2 == self.ground_geom_id and geom1 in self.foot_geom_lookup:
                        foot_geom = geom1
                    if foot_geom is not None:
                        desired[self.foot_geom_lookup[foot_geom]] = 1.0
                contact_alpha = dt / (0.035 + dt)
                self.filtered_contact_weight += contact_alpha * (
                    desired - self.filtered_contact_weight
                )
                contact_weight = self.filtered_contact_weight

            pitch_positions = self.index.leg_positions[:, :, 1].reshape(-1)
            corrected = self.target.copy()
            leg_corrections = (
                corrections[:, None]
                * np.broadcast_to(self.side_sign[None, :], (SEGMENTS, 2))
                * contact_weight
            )
            tip_guard_start_deg = spec.get("tip_guard_start_deg")
            if tip_guard_start_deg is not None:
                tip_abs_deg = np.degrees(
                    np.abs(data.qpos[self.index.tip_qpos_addresses])
                ).reshape(SEGMENTS, 2)
                guard_x = np.clip(
                    (tip_abs_deg - float(tip_guard_start_deg))
                    / (35.0 - float(tip_guard_start_deg)),
                    0.0,
                    1.0,
                )
                guard_blend = guard_x * guard_x * (3.0 - 2.0 * guard_x)
                leg_corrections *= 1.0 - 0.80 * guard_blend
            corrected[pitch_positions] += leg_corrections.reshape(-1)
            if previous_corrected is None:
                target_velocity = np.zeros_like(corrected)
            else:
                target_velocity = (corrected - previous_corrected) / dt

            q = data.qpos[self.index.qpos_addresses[pitch_positions]]
            qdot = data.qvel[self.index.dof_addresses[pitch_positions]]
            kp = self.index.kp[pitch_positions]
            kv = self.index.kv[pitch_positions]
            composite = corrected[pitch_positions] + (kv / kp) * target_velocity[pitch_positions]
            active_ids = self.index.active_actuator_ids[pitch_positions]
            data.ctrl[active_ids] = composite
            torque[pitch_positions] = kp * (corrected[pitch_positions] - q) + kv * (
                target_velocity[pitch_positions] - qdot
            )

            roll_q = data.qpos[self.index.roll_qpos_addresses]
            roll_qd = data.qvel[self.index.roll_dof_addresses]
            roll_tau = -blend * (
                float(spec["roll_kp"]) * roll_q + float(spec["roll_kv"]) * roll_qd
            )
            torque_limit = float(spec.get("roll_torque_limit", ROLL_TORQUE_LIMIT_NM))
            roll_tau = np.clip(
                roll_tau, -torque_limit, torque_limit
            )
            data.ctrl[roll_ids_holder["ids"]] = roll_tau

            self.target[:] = corrected
            self.previous_target = corrected.copy()
            correction_time.extend([time_s] * (SEGMENTS * 2))
            correction_abs_deg.extend(np.degrees(np.abs(leg_corrections)).reshape(-1).tolist())
            roll_torque_time.extend([time_s] * len(roll_tau))
            roll_torque_abs_nm.extend(np.abs(roll_tau).tolist())
            return torque

    def signed_roll_capture(data, body_ids):
        world_deg = original_signed_roll(data, body_ids)
        if len(body_ids) == 21 and float(data.time) >= min(EVAL_START_S, 0.25 * duration_s):
            radians = np.radians(world_deg)
            coherent_abs_deg.append(abs(math.degrees(circular_mean_rad(radians))))
            front_abs_deg.append(abs(math.degrees(circular_mean_rad(radians[:7]))))
            frame_world_span_deg.append(float(np.ptp(world_deg)))
        return world_deg

    tip.configure_model = configure_stabilized
    tip.LeanController = EngineeringRollController
    tip.signed_world_roll_deg = signed_roll_capture
    try:
        row = tip.run_case(
            candidate,
            kind,
            model_path,
            duration_s,
            video=video,
            camera_azimuth=camera_azimuth,
            camera_elevation=camera_elevation,
        )
    finally:
        tip.configure_model = original_configure
        tip.LeanController = original_controller
        tip.signed_world_roll_deg = original_signed_roll

    correction_eval = [
        value
        for time_s, value in zip(correction_time, correction_abs_deg)
        if time_s >= min(EVAL_START_S, 0.25 * duration_s)
    ]
    torque_eval = [
        value
        for time_s, value in zip(roll_torque_time, roll_torque_abs_nm)
        if time_s >= min(EVAL_START_S, 0.25 * duration_s)
    ]
    row.update(
        {
            "engineering_feedback": True,
            "projected_gravity_source": "circular mean of all chassis world-roll estimates",
            "projected_gravity_angle_gain": float(spec["angle_gain"]),
            "projected_gravity_rate_gain_s": float(spec["rate_gain_s"]),
            "projected_gravity_local_gain": float(spec.get("local_gain", 0.0)),
            "contact_gate_minimum_weight": spec.get("contact_gate_min"),
            "contact_gate_filter_tau_s": 0.035 if spec.get("contact_gate_min") is not None else None,
            "tip_authority_guard_start_deg": spec.get("tip_guard_start_deg"),
            "tip_authority_guard_full_deg": 35.0 if spec.get("tip_guard_start_deg") is not None else None,
            "tip_authority_guard_minimum_weight": 0.20 if spec.get("tip_guard_start_deg") is not None else None,
            "projected_gravity_filter_tau_s": FILTER_TAU_S,
            "projected_gravity_rate_filter_tau_s": RATE_FILTER_TAU_S,
            "feedback_start_s": FEEDBACK_START_S,
            "feedback_ramp_s": FEEDBACK_RAMP_S,
            "leg_pitch_correction_clip_deg": math.degrees(float(spec["clip_rad"])),
            "leg_pitch_correction_abs_p95_deg": percentile(correction_eval, 0.95),
            "leg_pitch_correction_abs_max_deg": max(correction_eval, default=0.0),
            "additional_body_roll_kp_nm_rad": float(spec["roll_kp"]),
            "additional_body_roll_kv_nms_rad": float(spec["roll_kv"]),
            "body_roll_torque_limit_nm": float(
                spec.get("roll_torque_limit", ROLL_TORQUE_LIMIT_NM)
            ),
            "body_roll_active_torque_abs_p95_nm": percentile(torque_eval, 0.95),
            "body_roll_active_torque_abs_max_nm": max(torque_eval, default=0.0),
            "coherent_chassis_roll_abs_p95_deg": percentile(coherent_abs_deg, 0.95),
            "coherent_chassis_roll_abs_max_deg": max(coherent_abs_deg, default=float("nan")),
            "front7_coherent_roll_abs_p95_deg": percentile(front_abs_deg, 0.95),
            "chassis_world_roll_span_p95_deg": percentile(frame_world_span_deg, 0.95),
            "frozen_plant_model_sha256": tip.sha256(model_path),
        }
    )
    return row


def selection_gates(row: dict[str, object], baseline: dict[str, object]) -> dict[str, bool]:
    return {
        "finite_warning_free": bool(row["finite"]) and not row["warning_counts"],
        "armature_zero": int(row["armature_nonzero_count"]) == 0,
        "wave_architecture_frozen": bool(row["wave_numbers_coupled"])
        and math.isclose(float(row["relative_leg_phase_deg"]), 0.0, abs_tol=1e-12),
        "topology_frozen": int(row["nq"]) == int(baseline["nq"])
        and int(row["nv"]) == int(baseline["nv"])
        and int(row["nu"]) == int(baseline["nu"]),
        "forward_retention": float(row["forward_displacement_m"])
        >= 0.90 * float(baseline["forward_displacement_m"]),
        "path_efficiency_retention": float(row["path_efficiency"])
        >= 0.92 * float(baseline["path_efficiency"]),
        "tip_compliance_not_suppressed": float(row["tip_roll_abs_p95_deg"])
        >= float(baseline["tip_roll_abs_p95_deg"]) - 3.0,
        "tip_motion_bounded_away_from_stop": float(row["tip_roll_abs_p95_deg"])
        <= float(row["tip_roll_limit_deg"]) - 5.0,
        "tip_stop_not_increased": float(row["tip_roll_stop_fraction"])
        <= float(baseline["tip_roll_stop_fraction"]) + 0.01,
        "contact_penetration_bounded": float(row["contact_penetration_p95_mm"])
        <= max(0.08, float(baseline["contact_penetration_p95_mm"]) + 0.02),
        "upright": float(row["minimum_root_upright_cos"])
        >= max(0.75, float(baseline["minimum_root_upright_cos"]) - 0.02),
        "spine_shape": float(row["maximum_spine_tortuosity"]) <= 1.30,
        "correction_bounded": float(row["leg_pitch_correction_abs_p95_deg"]) <= 20.0,
        "roll_torque_bounded": float(row["body_roll_active_torque_abs_max_nm"])
        <= float(row["body_roll_torque_limit_nm"]) + 1e-12,
        "coherent_roll_halved": float(row["coherent_chassis_roll_abs_p95_deg"])
        <= 0.50 * float(baseline["coherent_chassis_roll_abs_p95_deg"]),
        "front_roll_bounded": float(row["front7_coherent_roll_abs_p95_deg"]) <= 8.0,
    }


def run_screen(duration_s: float, names: list[str], kinds: list[str]) -> dict[str, object]:
    ensure_directories()
    before = tip.robot_sentinels()
    if "baseline" not in names:
        names = ["baseline", *names]
    rows = [
        run_candidate(name, kind, duration_s)
        for name in names
        for kind in kinds
    ]
    baselines = {row["kind"]: row for row in rows if row["candidate"] == "baseline"}
    for row in rows:
        baseline = baselines[str(row["kind"])]
        row["forward_retention"] = float(row["forward_displacement_m"]) / max(
            float(baseline["forward_displacement_m"]), 1e-12
        )
        row["coherent_roll_retention"] = float(
            row["coherent_chassis_roll_abs_p95_deg"]
        ) / max(float(baseline["coherent_chassis_roll_abs_p95_deg"]), 1e-12)
        row["engineering_gates"] = selection_gates(row, baseline)
        row["all_engineering_gates"] = bool(all(row["engineering_gates"].values()))

    passing = []
    for name in names:
        if name == "baseline":
            continue
        candidate_rows = [row for row in rows if row["candidate"] == name]
        if len(candidate_rows) == len(kinds) and all(
            row["all_engineering_gates"] for row in candidate_rows
        ):
            passing.append(name)
    selected = min(
        passing,
        key=lambda name: (
            abs(CANDIDATES[name]["angle_gain"]),
            abs(CANDIDATES[name]["rate_gain_s"]),
            CANDIDATES[name]["roll_kp"],
        ),
    ) if passing else None
    after = tip.robot_sentinels()
    payload = {
        "purpose": "feasibility-vector screen; no scalar reward",
        "duration_s": duration_s,
        "models": {kind: str(path) for kind, path in MODELS.items()},
        "frozen_parameters": {
            "mass_inertia_armature": True,
            "passive_body_roll": True,
            "passive_tip_roll": True,
            "body_leg_wave_coupled": True,
            "relative_leg_phase_deg": 0.0,
            "gait_amplitudes": True,
            "head_link1_topology": "fixed_-29_deg",
        },
        "candidate_specs": {name: CANDIDATES[name] for name in names},
        "rows": rows,
        "passing_candidates": passing,
        "selected_candidate": selected,
        "selection_rule": "all gates first, then minimum feedback/impedance intervention",
        "robot_sentinels_before": before,
        "robot_sentinels_after": after,
        "robot_sentinels_unchanged": before == after,
    }
    (DATA / "engineering_roll_screen.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def render_demos(duration_s: float, candidate: str | None) -> dict[str, object]:
    ensure_directories()
    screen_path = DATA / "engineering_roll_screen.json"
    if candidate is None:
        if not screen_path.exists():
            raise RuntimeError("Run the flat/rough screen before rendering demos")
        candidate = json.loads(screen_path.read_text(encoding="utf-8"))[
            "selected_candidate"
        ]
    if not candidate:
        raise RuntimeError("No candidate passed the engineering gate set")
    before = tip.robot_sentinels()
    views = [
        ("front", 180.0, -10.0),
        ("side", 90.0, -12.0),
        ("top", 90.0, -89.0),
    ]
    rows = []
    video_paths: list[Path] = []
    for kind in ("flat", "rough"):
        for view, azimuth, elevation in views:
            video = VIDEOS / f"engineering_roll_{candidate}_{kind}_{view}.mp4"
            rows.append(
                run_candidate(
                    candidate,
                    kind,
                    duration_s,
                    video=video,
                    camera_azimuth=azimuth,
                    camera_elevation=elevation,
                )
            )
            video_paths.append(video)

    cells = []
    for video in video_paths:
        reader = imageio.get_reader(str(video))
        frame_index = min(90, max(0, int(reader.count_frames()) - 1))
        cells.append(Image.fromarray(reader.get_data(frame_index)).resize((480, 270)))
        reader.close()
    sheet = Image.new("RGB", (1440, 580), "white")
    draw = ImageDraw.Draw(sheet)
    for index, cell in enumerate(cells):
        x = (index % 3) * 480
        y = (index // 3) * 290
        sheet.paste(cell, (x, y))
        draw.text((x + 8, y + 272), video_paths[index].name, fill="black")
    sheet_path = FIGURES / f"engineering_roll_{candidate}_contact_sheet.png"
    sheet.save(sheet_path)

    after = tip.robot_sentinels()
    payload = {
        "candidate": candidate,
        "duration_s": duration_s,
        "views": [name for name, _, _ in views],
        "rows": rows,
        "videos": [str(path) for path in video_paths],
        "contact_sheet": str(sheet_path),
        "robot_sentinels_before": before,
        "robot_sentinels_after": after,
        "robot_sentinels_unchanged": before == after,
    }
    (DATA / "engineering_roll_demo_audit.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def compact(payload: dict[str, object]) -> dict[str, object]:
    output = dict(payload)
    if "rows" in output:
        output["row_count"] = len(output.pop("rows"))
    for key in ("robot_sentinels_before", "robot_sentinels_after"):
        output.pop(key, None)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("screen", "demo", "all"))
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--candidates", nargs="*", choices=tuple(CANDIDATES))
    parser.add_argument("--kinds", nargs="*", choices=("flat", "rough"))
    parser.add_argument("--candidate", choices=tuple(CANDIDATES))
    args = parser.parse_args()
    output = {}
    if args.command in ("screen", "all"):
        output["screen"] = run_screen(
            args.duration,
            list(args.candidates or CANDIDATES),
            list(args.kinds or ("flat", "rough")),
        )
    if args.command in ("demo", "all"):
        output["demo"] = render_demos(args.duration, args.candidate)
    print(json.dumps({key: compact(value) for key, value in output.items()}, indent=2))


if __name__ == "__main__":
    main()
