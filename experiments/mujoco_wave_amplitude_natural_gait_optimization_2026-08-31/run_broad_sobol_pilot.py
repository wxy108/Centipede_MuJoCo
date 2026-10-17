"""Broad, anchor-calibrated amplitude pilot for the frozen Centipede Model.

This is a feasibility-landscape pilot, not a final continuous optimum claim.
Only body yaw, leg yaw, and leg pitch amplitudes vary.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import importlib.util
import json
import math
import multiprocessing
import sys
import time
from itertools import product
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Codex's bundled Python is isolated with site loading disabled.  Its own
# signed packages are compatible with this experiment and avoid the Windows
# policy block on the historical project-local binary wheels.
TRUSTED_SITE = Path(sys.executable).resolve().parent / "Lib" / "site-packages"
if TRUSTED_SITE.exists():
    sys.path.insert(0, str(TRUSTED_SITE))

import numpy as np

# MuJoCo 3.3.7 is stored in the project-local package directory.  Append that
# directory only after the signed bundled NumPy is imported, otherwise the
# blocked project-local NumPy wheel shadows it on Windows.
PROJECT_PACKAGES = PROJECT_ROOT / ".python_packages"
if PROJECT_PACKAGES.exists():
    sys.path.append(str(PROJECT_PACKAGES))

import mujoco


ROOT = Path(__file__).resolve().parent
REJECTED_ROOT = (
    PROJECT_ROOT / "experiments" / "mujoco_wave_amplitude_global_optimization_2026-08-31"
)
OPTIMIZER = REJECTED_ROOT / "run_wave_amplitude_global.py"
FINAL_SELECTION = REJECTED_ROOT / "outputs" / "data" / "final_selection_summary.json"
DATA = ROOT / "outputs" / "data"
JSONL = DATA / "broad_sobol_pilot_trials.jsonl"
CSV = DATA / "broad_sobol_pilot_trials.csv"
SUMMARY = DATA / "broad_sobol_pilot_summary.json"
SOBOL_BANK = DATA / "sobol_point_bank_power7.json"

WAVES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0)
LOWER = np.asarray([0.0, 0.0, 0.0])
UPPER = np.asarray([36.0, 90.0, 60.0])
SAMPLE_DT_S = 0.01


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


opt = load_module(OPTIMIZER, "centipede_natural_pilot_base")


def pctl(values, q: float) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.quantile(array, q)) if array.size else float("nan")


def sinusoid_residual(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    omega = 2.0 * math.pi * opt.base.FREQUENCY_HZ
    design = np.column_stack((np.sin(omega * times), np.cos(omega * times), np.ones_like(times)))
    output = []
    for column in values.T:
        beta, *_ = np.linalg.lstsq(design, column, rcond=None)
        fitted = design @ beta
        centered = float(np.sqrt(np.mean((column - np.mean(column)) ** 2)))
        resid = float(np.sqrt(np.mean((column - fitted) ** 2)))
        output.append(resid / max(centered, 1e-12))
    return np.asarray(output)


def reversal_rate(times: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    window = max(3, int(round(0.05 / SAMPLE_DT_S)))
    kernel = np.ones(window) / window
    cycles = max((float(times[-1]) - float(times[0])) * opt.base.FREQUENCY_HZ, 1e-12)
    output = []
    for column in velocities.T:
        smooth = np.convolve(column, kernel, mode="same")
        threshold = 0.05 * pctl(np.abs(smooth), 0.95)
        signs = np.sign(smooth)
        signs[np.abs(smooth) < threshold] = 0.0
        signs = signs[signs != 0.0]
        changes = int(np.count_nonzero(signs[1:] != signs[:-1])) if signs.size else 0
        output.append(changes / cycles)
    return np.asarray(output)


class ChunkedFastEvaluator(opt.FastEvaluator):
    """Broad-screen evaluator with slower CPG updates but unchanged physics dt.

    This is a multi-fidelity screening approximation only.  MuJoCo still
    integrates at the XML timestep; the 1 Hz open-loop target is held for a
    small number of physics steps.  Finalists must be certified with stride 1.
    """

    def __init__(self, control_stride: int, proof_early_exit: bool = False):
        super().__init__()
        self.control_stride = int(control_stride)
        self.proof_early_exit = bool(proof_early_exit)
        if self.control_stride < 1:
            raise ValueError("control_stride must be positive")

    def evaluate(self, task):
        wave, body_deg, leg_yaw_deg, leg_pitch_deg, duration_s, eval_start_s = task[:6]
        erasure_q = float(task[6]) if len(task) > 6 else 0.0
        erasure_seed = int(task[7]) if len(task) > 7 else 0
        model, data, index = self.model, self.data, self.index
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        controller = opt.ThreeAmplitudeController(
            model, index, wave, body_deg, leg_yaw_deg, leg_pitch_deg
        )
        dt = float(model.opt.timestep)
        events, erasure_opportunities = opt.build_erasure_events(
            int(wave), erasure_q, erasure_seed, eval_start_s, duration_s
        )
        erasure = opt.ContactErasure(model, self.segment_collision_geoms, events)
        erased_segment_time = np.zeros(19, dtype=float)
        sample_stride = max(1, int(round(0.01 / dt)))
        if sample_stride % self.control_stride:
            raise ValueError("control_stride must divide the 10 ms metric stride")
        start = end = previous = None
        previous_z = None
        previous_sample_t = None
        path_length = 0.0
        min_upright = float("inf")
        max_tortuosity = 0.0
        active_torque_max = 0.0
        active_speed_max = 0.0
        tip_speed_max = 0.0
        coherent_roll_abs = []
        front_roll_abs = []
        tip_abs_deg = []
        tip_stop = []
        contact_counts = []
        penetrations_mm = []
        center_z = []
        vertical_speed_abs = []
        chassis_ground_samples = 0
        evaluated_samples = 0
        no_contact_run = 0
        max_no_contact_run = 0
        finite = True
        early_terminated_by = None
        simulated_steps = 0
        maximum_possible_samples = int(math.ceil((duration_s - eval_start_s) / 0.01)) + 1
        try:
            while float(data.time) < duration_s - 0.5 * dt:
                remaining = int(round((duration_s - float(data.time)) / dt))
                chunk = max(1, min(self.control_stride, remaining))
                erased = erasure.update(float(data.time))
                if float(data.time) >= eval_start_s:
                    erased_segment_time += erased.astype(float) * (chunk * dt)
                model.opt.timestep = dt * chunk
                active_torque_max = max(active_torque_max, controller.step(data))
                model.opt.timestep = dt
                if self.proof_early_exit and active_torque_max > 2.0e-4:
                    early_terminated_by = "active_torque_abs_max_nm"
                    break
                mujoco.mj_step(model, data, chunk)
                simulated_steps += chunk
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    finite = False
                    break
                metric_phase = (
                    (simulated_steps - 1) % sample_stride
                    if self.control_stride == 1
                    else simulated_steps % sample_stride
                )
                if float(data.time) < eval_start_s or metric_phase:
                    continue
                evaluated_samples += 1
                center = opt.base.chassis_centroid(data, index)
                posture = opt.base.posture_metrics(data, index)
                min_upright = min(min_upright, float(posture["root_upright_cos"]))
                max_tortuosity = max(max_tortuosity, float(posture["spine_tortuosity"]))
                if start is None:
                    start = center.copy()
                if previous is not None:
                    path_length += float(np.linalg.norm(center[:2] - previous[:2]))
                previous = center.copy()
                end = center.copy()
                center_z.append(float(center[2]))
                if previous_z is not None and previous_sample_t is not None:
                    vertical_speed_abs.append(
                        abs(float(center[2]) - previous_z)
                        / max(float(data.time) - previous_sample_t, 1e-12)
                    )
                previous_z = float(center[2])
                previous_sample_t = float(data.time)
                active_speed_max = max(
                    active_speed_max,
                    float(np.max(np.abs(data.qvel[index.dof_addresses]))),
                )
                tip_speed_max = max(
                    tip_speed_max,
                    float(np.max(np.abs(data.qvel[index.tip_dof_addresses]))),
                )
                tip_now = np.degrees(np.abs(data.qpos[index.tip_qpos_addresses]))
                tip_abs_deg.extend(tip_now.tolist())
                tip_stop.extend((tip_now >= 37.75).tolist())
                world = opt.tip.signed_world_roll_deg(data, index.chassis_body_ids)
                radians = np.radians(world)
                coherent_roll_abs.append(
                    abs(math.degrees(opt.circular_mean_rad(radians)))
                )
                front_roll_abs.append(
                    abs(math.degrees(opt.circular_mean_rad(radians[:7])))
                )
                contacting = set()
                chassis_ground = False
                for contact_id in range(data.ncon):
                    contact = data.contact[contact_id]
                    geom1, geom2 = int(contact.geom1), int(contact.geom2)
                    other = geom2 if geom1 == self.ground else (geom1 if geom2 == self.ground else None)
                    if other is None:
                        continue
                    if other in self.foot_lookup:
                        contacting.add(other)
                        penetrations_mm.append(max(0.0, -1000.0 * float(contact.dist)))
                    if other in self.chassis_geoms:
                        chassis_ground = True
                contact_counts.append(float(len(contacting)))
                if contacting:
                    no_contact_run = 0
                else:
                    no_contact_run += 1
                    max_no_contact_run = max(max_no_contact_run, no_contact_run)
                chassis_ground_samples += int(chassis_ground)
                if self.proof_early_exit:
                    irreversible = None
                    if min_upright < math.cos(math.radians(50.0)):
                        irreversible = "minimum_root_upright_cos"
                    elif max_tortuosity > 1.20:
                        irreversible = "maximum_spine_tortuosity"
                    elif max(center_z) - min(center_z) > 0.004:
                        irreversible = "chassis_z_peak_to_peak_mm"
                    elif active_speed_max > 100.0:
                        irreversible = "active_joint_speed_abs_max_rad_s"
                    elif max_no_contact_run * sample_stride * dt > 0.08:
                        irreversible = "max_no_foot_contact_duration_s"
                    elif chassis_ground_samples > 0.08 * maximum_possible_samples:
                        irreversible = "chassis_ground_contact_fraction"
                    elif sum(tip_stop) > 0.03 * maximum_possible_samples * len(index.tip_qpos_addresses):
                        irreversible = "tip_stop_fraction"
                    if irreversible is not None:
                        early_terminated_by = irreversible
                        break
        finally:
            model.opt.timestep = dt
            erasure.restore()

        displacement = (
            opt.base.FORWARD_SIGN * float(end[0] - start[0])
            if start is not None and end is not None
            else float("nan")
        )
        path_efficiency = displacement / max(path_length, 1e-12)
        warnings = opt.tip.stage1.warning_counts(data)
        metrics = {
            "forward_displacement_m": displacement,
            "path_efficiency": path_efficiency,
            "minimum_root_upright_cos": min_upright,
            "maximum_spine_tortuosity": max_tortuosity,
            "coherent_roll_p95_deg": opt.percentile(coherent_roll_abs, 0.95),
            "front7_roll_p95_deg": opt.percentile(front_roll_abs, 0.95),
            "chassis_z_peak_to_peak_mm": 1000.0
            * (max(center_z, default=float("nan")) - min(center_z, default=float("nan"))),
            "vertical_speed_abs_p99_m_s": opt.percentile(vertical_speed_abs, 0.99),
            "active_joint_speed_abs_max_rad_s": active_speed_max,
            "tip_speed_abs_max_rad_s": tip_speed_max,
            "active_torque_abs_max_nm": active_torque_max,
            "mean_contacting_feet": float(np.mean(contact_counts)) if contact_counts else float("nan"),
            "max_no_foot_contact_duration_s": max_no_contact_run * sample_stride * dt,
            "chassis_ground_contact_fraction": chassis_ground_samples / max(evaluated_samples, 1),
            "contact_penetration_p95_mm": opt.percentile(penetrations_mm, 0.95),
            "tip_roll_p95_deg": opt.percentile(tip_abs_deg, 0.95),
            "tip_stop_fraction": float(np.mean(tip_stop)) if tip_stop else float("nan"),
        }
        gates = {
            "finite_warning_free": finite and not warnings,
            "positive_forward_displacement": displacement > 0.0,
            "path_efficiency": path_efficiency >= 0.80,
            "upright": min_upright >= math.cos(math.radians(50.0)),
            "spine_tortuosity": max_tortuosity <= 1.20,
            "coherent_roll": metrics["coherent_roll_p95_deg"] <= 10.0,
            "front_roll": metrics["front7_roll_p95_deg"] <= 12.0,
            "no_jump_height": metrics["chassis_z_peak_to_peak_mm"] <= 4.0,
            "no_jump_speed": metrics["vertical_speed_abs_p99_m_s"] <= 0.10,
            "active_joint_speed": active_speed_max <= 100.0,
            "tip_speed": tip_speed_max <= 1000.0,
            "active_torque": active_torque_max <= 2.0e-4,
            "contacting_feet": 6.0 <= metrics["mean_contacting_feet"] <= 28.0,
            "no_ballistic_gap": metrics["max_no_foot_contact_duration_s"] <= 0.08,
            "no_chassis_drag": metrics["chassis_ground_contact_fraction"] <= 0.05,
            "contact_penetration": metrics["contact_penetration_p95_mm"] <= 0.10,
            "tip_motion_bounded": metrics["tip_roll_p95_deg"] <= 33.0,
            "tip_stop_bounded": metrics["tip_stop_fraction"] <= 0.03,
        }
        return {
            "wave": wave,
            "body_yaw_amplitude_deg": body_deg,
            "leg_yaw_amplitude_deg": leg_yaw_deg,
            "leg_pitch_amplitude_deg": leg_pitch_deg,
            "duration_s": duration_s,
            "eval_start_s": eval_start_s,
            "erasure_q": erasure_q,
            "erasure_seed": erasure_seed,
            "erasure_opportunities": erasure_opportunities,
            "erasure_triggered_events": len(events),
            "realized_erased_load": float(erased_segment_time.sum())
            / max(19.0 * (duration_s - eval_start_s), 1e-12),
            "finite": finite,
            "proof_early_exit": self.proof_early_exit,
            "early_terminated_by": early_terminated_by,
            "warning_counts": warnings,
            **metrics,
            "gates": gates,
            "feasible": bool(all(gates.values())),
        }


class PilotEvaluator:
    def __init__(self, control_stride: int = 1, proof_early_exit: bool = False):
        self.control_stride = int(control_stride)
        self.proof_early_exit = bool(proof_early_exit)
        if self.proof_early_exit:
            self.base = ChunkedFastEvaluator(self.control_stride, True)
        elif self.control_stride == 1:
            self.base = opt.FastEvaluator()
        else:
            self.base = ChunkedFastEvaluator(self.control_stride)
        model = self.base.model
        self.foot_sites = np.asarray(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"site_foot_{seg}_{side}")
                for seg in range(19)
                for side in range(2)
            ],
            dtype=int,
        )
        self.foot_lookup = {int(geom): i for i, geom in enumerate(self.base.foot_geoms)}

    def naturality(self, wave, body, leg_yaw, leg_pitch, duration, eval_start):
        model, data, index = self.base.model, self.base.data, self.base.index
        controller = opt.ThreeAmplitudeController(model, index, wave, body, leg_yaw, leg_pitch)
        dt = float(model.opt.timestep)
        stride = max(1, int(round(SAMPLE_DT_S / dt)))
        times = []
        body_q = []
        leg_q = []
        leg_qvel = []
        leg_target = []
        foot_z = []
        contacts = []
        simulated_steps = 0
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        try:
            while float(data.time) < duration - 0.5 * dt:
                remaining = int(round((duration - float(data.time)) / dt))
                chunk = max(1, min(self.control_stride, remaining))
                model.opt.timestep = dt * chunk
                controller.step(data)
                model.opt.timestep = dt
                mujoco.mj_step(model, data, chunk)
                simulated_steps += chunk
                metric_phase = (
                    (simulated_steps - 1) % stride
                    if self.control_stride == 1
                    else simulated_steps % stride
                )
                if float(data.time) < eval_start or metric_phase:
                    continue
                q = data.qpos[index.qpos_addresses]
                qvel = data.qvel[index.dof_addresses]
                times.append(float(data.time))
                body_q.append(q[index.body_positions].copy())
                leg_q.append(q[index.leg_positions.reshape(-1)].copy())
                leg_qvel.append(qvel[index.leg_positions.reshape(-1)].copy())
                leg_target.append(controller.target[index.leg_positions.reshape(-1)].copy())
                foot_z.append(data.site_xpos[self.foot_sites, 2].copy())
                mask = np.zeros(38, dtype=bool)
                for contact_id in range(data.ncon):
                    contact = data.contact[contact_id]
                    g1, g2 = int(contact.geom1), int(contact.geom2)
                    other = g2 if g1 == self.base.ground else (g1 if g2 == self.base.ground else -1)
                    foot_index = self.foot_lookup.get(other)
                    if foot_index is not None:
                        mask[foot_index] = True
                contacts.append(mask)
        finally:
            model.opt.timestep = dt

        t = np.asarray(times)
        body_q = np.asarray(body_q)
        leg_q = np.asarray(leg_q).reshape(len(t), 19, 2, 2)
        leg_qvel = np.asarray(leg_qvel).reshape(len(t), 19, 2, 2)
        leg_target = np.asarray(leg_target).reshape(len(t), 19, 2, 2)
        foot_z = np.asarray(foot_z)
        contacts = np.asarray(contacts)
        yaw_values = leg_q[..., 0].reshape(len(t), -1)
        pitch_values = leg_q[..., 1].reshape(len(t), -1)
        yaw_vel = leg_qvel[..., 0].reshape(len(t), -1)
        pitch_vel = leg_qvel[..., 1].reshape(len(t), -1)
        duty = np.mean(contacts, axis=0)
        swing = foot_z[~contacts]
        body_resid = sinusoid_residual(t, body_q[:, 1:])
        yaw_resid = sinusoid_residual(t, yaw_values)
        pitch_resid = sinusoid_residual(t, pitch_values)
        yaw_reverse = reversal_rate(t, yaw_vel)
        pitch_reverse = reversal_rate(t, pitch_vel)
        metrics = {
            "naturality_samples": len(t),
            "body_nonsinusoidal_residual_p95": pctl(body_resid, 0.95),
            "leg_yaw_nonsinusoidal_residual_p95": pctl(yaw_resid, 0.95),
            "leg_pitch_nonsinusoidal_residual_p95": pctl(pitch_resid, 0.95),
            "leg_yaw_tracking_error_rms_deg": math.degrees(float(np.sqrt(np.mean((leg_q[..., 0] - leg_target[..., 0]) ** 2)))),
            "leg_pitch_tracking_error_rms_deg": math.degrees(float(np.sqrt(np.mean((leg_q[..., 1] - leg_target[..., 1]) ** 2)))),
            "leg_yaw_speed_p99_rad_s": pctl(np.abs(yaw_vel), 0.99),
            "leg_pitch_speed_p99_rad_s": pctl(np.abs(pitch_vel), 0.99),
            "leg_yaw_velocity_reversals_per_cycle_p95": pctl(yaw_reverse, 0.95),
            "leg_pitch_velocity_reversals_per_cycle_p95": pctl(pitch_reverse, 0.95),
            "contact_duty_mean": float(np.mean(duty)),
            "contact_duty_p05": pctl(duty, 0.05),
            "contact_duty_p95": pctl(duty, 0.95),
            "left_right_contact_duty_abs_diff_mean": float(np.mean(np.abs(duty.reshape(19, 2)[:, 0] - duty.reshape(19, 2)[:, 1]))),
            "swing_foot_height_p95_mm": 1000.0 * pctl(swing, 0.95),
        }
        gates = {
            "body_wave_shape": metrics["body_nonsinusoidal_residual_p95"] <= 0.40,
            "leg_yaw_wave_shape": metrics["leg_yaw_nonsinusoidal_residual_p95"] <= 0.18,
            "leg_pitch_wave_shape": metrics["leg_pitch_nonsinusoidal_residual_p95"] <= 0.40,
            "leg_yaw_speed": metrics["leg_yaw_speed_p99_rad_s"] <= 8.0,
            "leg_pitch_speed": metrics["leg_pitch_speed_p99_rad_s"] <= 7.0,
            "leg_yaw_reversals": metrics["leg_yaw_velocity_reversals_per_cycle_p95"] <= 3.0,
            "leg_pitch_reversals": metrics["leg_pitch_velocity_reversals_per_cycle_p95"] <= 6.25,
            "leg_pitch_tracking": metrics["leg_pitch_tracking_error_rms_deg"] <= 10.0,
            "swing_height": metrics["swing_foot_height_p95_mm"] <= 5.5,
            "contact_duty": 0.15 <= metrics["contact_duty_mean"] <= 0.65,
            "left_right_contact_balance": metrics["left_right_contact_duty_abs_diff_mean"] <= 0.10,
        }
        return metrics, gates

    def evaluate(self, task):
        candidate_id, source, wave, body, leg_yaw, leg_pitch, duration, eval_start = task
        row = self.base.evaluate((wave, body, leg_yaw, leg_pitch, duration, eval_start))
        base_required = {
            name: passed
            for name, passed in row["gates"].items()
            if name not in {"tip_speed", "no_chassis_drag"}
        }
        base_required["accepted_clean_chassis_contact_envelope"] = row["chassis_ground_contact_fraction"] <= 0.08
        provisional = bool(all(base_required.values()))
        if provisional:
            natural_metrics, natural_gates = self.naturality(
                wave, body, leg_yaw, leg_pitch, duration, eval_start
            )
        else:
            natural_metrics = {}
            natural_gates = {"not_run_after_base_failure": False}
        feasible = provisional and all(natural_gates.values())
        return {
            "candidate_id": candidate_id,
            "source": source,
            "control_stride": self.control_stride,
            "proof_early_exit": self.proof_early_exit,
            **{key: value for key, value in row.items() if key not in {"gates", "feasible"}},
            **natural_metrics,
            "base_gates_v2": base_required,
            "naturality_gates": natural_gates,
            "provisional_base_feasible": provisional,
            "feasible_v2_pilot": bool(feasible),
        }


_EVALUATOR = None


def init_worker(control_stride=1, proof_early_exit=False):
    global _EVALUATOR
    _EVALUATOR = PilotEvaluator(control_stride, proof_early_exit)


def worker(task):
    if _EVALUATOR is None:
        init_worker()
    return _EVALUATOR.evaluate(task)


def anchors() -> list[tuple[str, tuple[float, float, float]]]:
    clean = (
        math.degrees(opt.base.BODY_AMPLITUDE_RAD * opt.tip.BODY_AMPLITUDE_SCALE),
        math.degrees(opt.base.LEG_YAW_AMPLITUDE_RAD * opt.tip.ctrl.LEG_YAW_AMPLITUDE_SCALE),
        math.degrees(opt.base.LEG_PITCH_AMPLITUDE_RAD * opt.tip.ctrl.LEG_PITCH_AMPLITUDE_SCALE) * 0.525,
    )
    return [
        ("accepted_clean_amplitudes", clean),
        ("isaac_yaml_amplitudes", tuple(map(math.degrees, (0.4, 0.6, 0.4)))),
        (
            "old_model_erasure_amplitudes",
            tuple(
                map(
                    math.degrees,
                    (
                        opt.base.BODY_AMPLITUDE_RAD,
                        opt.base.LEG_YAW_AMPLITUDE_RAD,
                        opt.base.LEG_PITCH_AMPLITUDE_RAD,
                    ),
                )
            ),
        ),
    ]


def boundary_points() -> list[tuple[str, tuple[float, float, float]]]:
    output = []
    for values in product((0.0, 1.0), repeat=3):
        point = LOWER + np.asarray(values) * (UPPER - LOWER)
        output.append(("box_corner", tuple(float(x) for x in point)))
    center = 0.5 * (LOWER + UPPER)
    for dim, name in enumerate(("body", "leg_yaw", "leg_pitch")):
        for value, side in ((LOWER[dim], "low"), (UPPER[dim], "high")):
            point = center.copy()
            point[dim] = value
            output.append((f"face_{name}_{side}", tuple(float(x) for x in point)))
    return output


def rejected_lookup():
    final = json.loads(FINAL_SELECTION.read_text(encoding="utf-8"))
    output = {}
    for wave, item in final["waves"].items():
        selected = item.get("selected")
        row = selected["grid_row"] if selected else item["diagnostic_best"]
        output[float(wave)] = (
            float(row["body_yaw_amplitude_deg"]),
            float(row["leg_yaw_amplitude_deg"]),
            float(row["leg_pitch_amplitude_deg"]),
        )
    return output


def build_tasks(waves, sobol_power, duration, eval_start):
    bank = json.loads(SOBOL_BANK.read_text(encoding="utf-8"))
    if int(bank["power"]) != int(sobol_power):
        raise ValueError(
            f"Sobol bank power={bank['power']} does not match requested power={sobol_power}"
        )
    rejected = rejected_lookup()
    tasks = []
    seen = set()
    for wave in waves:
        normalized = np.asarray(bank["normalized_points"][f"{wave:g}"], dtype=float)
        samples = LOWER + normalized * (UPPER - LOWER)
        points = [("sobol", tuple(float(x) for x in row)) for row in samples]
        points.extend(anchors())
        points.extend(boundary_points())
        if float(wave) in rejected:
            points.append(("rejected_negative_control", rejected[float(wave)]))
        for point_index, (source, (body, yaw, pitch)) in enumerate(points):
            key = (float(wave), round(body, 8), round(yaw, 8), round(pitch, 8))
            if key in seen:
                continue
            seen.add(key)
            # Keep IDs stable when a single wave is resumed on its own.  The old
            # global len(tasks) suffix collided across separate per-wave runs.
            cid = f"w{wave:g}_{source}_{point_index:06d}"
            tasks.append((cid, source, float(wave), body, yaw, pitch, duration, eval_start))
    return tasks


def read_completed(jsonl_path=JSONL):
    rows = []
    keys = set()
    if not jsonl_path.exists():
        return rows, keys
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(row)
        keys.add(row["candidate_id"])
    return rows, keys


def write_outputs(rows, metadata, csv_path=CSV, summary_path=SUMMARY):
    rows = sorted(rows, key=lambda r: (float(r["wave"]), str(r["candidate_id"])))
    flat = []
    keys = []
    for row in rows:
        record = {}
        for key, value in row.items():
            record[key] = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
        flat.append(record)
        for key in record:
            if key not in keys:
                keys.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(flat)
    summary = {"metadata": metadata, "waves": {}}
    for wave in metadata["waves"]:
        group = [r for r in rows if math.isclose(float(r["wave"]), wave)]
        feasible = [r for r in group if r["feasible_v2_pilot"]]
        feasible.sort(
            key=lambda r: (
                -float(r["forward_displacement_m"]),
                -float(r["path_efficiency"]),
                float(r["body_yaw_amplitude_deg"])
                + float(r["leg_yaw_amplitude_deg"])
                + float(r["leg_pitch_amplitude_deg"]),
            )
        )
        summary["waves"][f"{wave:g}"] = {
            "trials": len(group),
            "base_feasible": sum(bool(r["provisional_base_feasible"]) for r in group),
            "natural_feasible": len(feasible),
            "pilot_best": feasible[0] if feasible else None,
            "top10": feasible[:10],
        }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sobol-power", type=int, default=7, help="2**m Sobol points per wave")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--eval-start", type=float, default=1.5)
    parser.add_argument(
        "--control-stride",
        type=int,
        default=1,
        choices=(1, 2, 4, 5, 10, 20),
        help="CPG update stride in physics steps; stride 1 is required for final certification",
    )
    parser.add_argument(
        "--proof-early-exit",
        action="store_true",
        help="Stop only after a hard-gate failure is mathematically irreversible",
    )
    parser.add_argument("--waves", type=float, nargs="*", default=list(WAVES))
    parser.add_argument(
        "--output-tag",
        default="",
        help="Write an isolated resumable shard instead of the primary output files",
    )
    parser.add_argument("--task-modulo", type=int, default=1)
    parser.add_argument("--task-remainder", type=int, default=0)
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    if args.output_tag:
        safe_tag = "".join(c for c in args.output_tag if c.isalnum() or c in "-_")
        if not safe_tag:
            raise ValueError("output tag contains no safe characters")
        jsonl_path = DATA / f"broad_sobol_pilot_trials_{safe_tag}.jsonl"
        csv_path = DATA / f"broad_sobol_pilot_trials_{safe_tag}.csv"
        summary_path = DATA / f"broad_sobol_pilot_summary_{safe_tag}.json"
    else:
        jsonl_path, csv_path, summary_path = JSONL, CSV, SUMMARY
    waves = tuple(float(x) for x in args.waves)
    all_tasks = build_tasks(waves, args.sobol_power, args.duration, args.eval_start)
    if args.task_modulo < 1 or not 0 <= args.task_remainder < args.task_modulo:
        raise ValueError("task remainder must satisfy 0 <= remainder < modulo")
    tasks = [
        task
        for task_index, task in enumerate(all_tasks)
        if task_index % args.task_modulo == args.task_remainder
    ]
    rows, completed = read_completed(jsonl_path)
    todo = [task for task in tasks if task[0] not in completed]
    print(f"tasks total={len(tasks)} completed={len(completed)} todo={len(todo)}", flush=True)
    started = time.time()
    def consume(iterator, handle):
        for index, row in enumerate(iterator, start=1):
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            rows.append(row)
            if index % 25 == 0 or index == len(todo):
                elapsed = time.time() - started
                feasible = sum(bool(r["feasible_v2_pilot"]) for r in rows)
                print(
                    f"progress {index}/{len(todo)} elapsed={elapsed/60:.1f}min natural_feasible={feasible}",
                    flush=True,
                )

    with jsonl_path.open("a", encoding="utf-8") as handle:
        if args.workers == 1:
            evaluator = PilotEvaluator(args.control_stride, args.proof_early_exit)
            consume((evaluator.evaluate(task) for task in todo), handle)
        else:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=init_worker,
                initargs=(args.control_stride, args.proof_early_exit),
            ) as pool:
                consume(pool.map(worker, todo, chunksize=1), handle)
    metadata = {
        "scope": "Centipede Model only; Robot Step 18/19 untouched",
        "model": str(opt.MODEL),
        "waves": waves,
        "sobol_power": args.sobol_power,
        "sobol_points_per_wave": 2 ** args.sobol_power,
        "domain_deg": {"body_yaw": [0, 36], "leg_yaw": [0, 90], "leg_pitch": [0, 60]},
        "duration_s": args.duration,
        "eval_start_s": args.eval_start,
        "control_stride": args.control_stride,
        "proof_early_exit": args.proof_early_exit,
        "candidate_count": len(tasks),
        "full_candidate_count": len(all_tasks),
        "task_modulo": args.task_modulo,
        "task_remainder": args.task_remainder,
        "continuous_global_claim": False,
        "status": "broad pilot; adaptive refinement required",
    }
    summary = write_outputs(rows, metadata, csv_path, summary_path)
    print(json.dumps({w: {k: v for k, v in item.items() if k != "top10"} for w, item in summary["waves"].items()}, indent=2))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
