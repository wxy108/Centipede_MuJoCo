"""Grid-global three-amplitude optimization for the current Centipede Model.

For every integer spatial wave number 1..9, exhaustively evaluate the same
pre-registered finite grid over only:

  1. body-yaw amplitude,
  2. leg-yaw amplitude,
  3. leg-pitch amplitude.

The objective is forward displacement on the frozen flat locked-head model.
Hard feasibility gates are applied before objective comparison.  There is no
weighted scalar reward.  The returned winner is therefore a reproducible
global optimum on the declared grid, not a claim of a continuous-domain
global optimum for nonsmooth contact dynamics.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import ctypes
import heapq
import importlib.util
import json
import math
import multiprocessing
import os
import sys
import threading
import time
from pathlib import Path

import mujoco
import numpy as np


_ORIGINAL_CDLL = ctypes.CDLL


def _cdll_without_optional_plugins(path, *args, **kwargs):
    if "site-packages\\mujoco\\plugin" in os.path.normcase(str(path)):
        return None
    return _ORIGINAL_CDLL(path, *args, **kwargs)


ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT.parents[1]
TIP_SCRIPT = (
    MODEL_ROOT
    / "experiments"
    / "mujoco_template_tip_compliance_2026-08-27"
    / "run_tip_compliance.py"
)
MODEL = (
    MODEL_ROOT
    / "experiments"
    / "mujoco_template_locked_head_2026-08-27"
    / "models"
    / "centipede_flat_locked_head_m29p0.xml"
)
OUTPUTS = ROOT / "outputs"
DATA = OUTPUTS / "data"

WAVES = tuple(range(1, 10))
BODY_YAW_DEG = (8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0)
LEG_YAW_DEG = (24.0, 30.0, 36.0, 42.0, 48.0, 54.0, 60.0)
LEG_PITCH_DEG = (12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0)
GRID_SIZE_PER_WAVE = len(BODY_YAW_DEG) * len(LEG_YAW_DEG) * len(LEG_PITCH_DEG)
SUPPORT_WINDOW_S = 0.5
MINIMUM_BURST_FRACTION = 0.02


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tip = load_module(TIP_SCRIPT, "centipede_wave_amplitude_tip")
base = tip.base


def circular_mean_rad(angles_rad: np.ndarray) -> float:
    return float(
        math.atan2(np.mean(np.sin(angles_rad)), np.mean(np.cos(angles_rad)))
    )


def percentile(values, q: float) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.quantile(array, q)) if array.size else float("nan")


def build_erasure_events(
    wave: int,
    q: float,
    seed: int,
    eval_start_s: float,
    duration_s: float,
) -> tuple[list[tuple[float, float, int]], int]:
    """Reproduce the prior seed-locked segment-pair erasure protocol."""
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must lie in [0, 1]")
    rng = np.random.default_rng(seed + 100_003 * wave)
    events: list[tuple[float, float, int]] = []
    opportunities = 0
    omega = 2.0 * math.pi * base.FREQUENCY_HZ
    for segment in range(19):
        phase = 2.0 * math.pi * wave * segment / 18.0
        phase_offset = (phase / omega) % SUPPORT_WINDOW_S
        window_start = eval_start_s + phase_offset
        while window_start < duration_s - 1e-12:
            opportunities += 1
            trigger, duration_draw, offset_draw = rng.random(3)
            fraction = math.exp(
                math.log(MINIMUM_BURST_FRACTION)
                + duration_draw * math.log(1.0 / MINIMUM_BURST_FRACTION)
            )
            burst_duration = fraction * SUPPORT_WINDOW_S
            start = window_start + offset_draw * (
                SUPPORT_WINDOW_S - burst_duration
            )
            if trigger < q:
                events.append((start, start + burst_duration, segment))
            window_start += SUPPORT_WINDOW_S
    events.sort()
    return events, opportunities


class ContactErasure:
    """Disable both complete leg collision subtrees of selected segment pairs."""

    def __init__(self, model: mujoco.MjModel, segment_geoms, events):
        self.model = model
        self.segment_geoms = segment_geoms
        self.events = events
        self.pointer = 0
        self.ends: list[tuple[float, int]] = []
        self.counts = np.zeros(19, dtype=int)
        self.original_contype = model.geom_contype.copy()
        self.original_conaffinity = model.geom_conaffinity.copy()

    def _activate(self, segment: int) -> None:
        if self.counts[segment] == 0:
            geoms = self.segment_geoms[segment]
            self.model.geom_contype[geoms] = 0
            self.model.geom_conaffinity[geoms] = 0
        self.counts[segment] += 1

    def _deactivate(self, segment: int) -> None:
        self.counts[segment] -= 1
        if self.counts[segment] < 0:
            raise RuntimeError("negative contact-erasure overlap count")
        if self.counts[segment] == 0:
            geoms = self.segment_geoms[segment]
            self.model.geom_contype[geoms] = self.original_contype[geoms]
            self.model.geom_conaffinity[geoms] = self.original_conaffinity[geoms]

    def update(self, t: float) -> np.ndarray:
        while self.ends and self.ends[0][0] <= t + 1e-12:
            _, segment = heapq.heappop(self.ends)
            self._deactivate(segment)
        while self.pointer < len(self.events) and self.events[self.pointer][0] <= t + 1e-12:
            start, end, segment = self.events[self.pointer]
            self._activate(segment)
            heapq.heappush(self.ends, (end, segment))
            self.pointer += 1
        return self.counts > 0

    def restore(self) -> None:
        self.model.geom_contype[:] = self.original_contype
        self.model.geom_conaffinity[:] = self.original_conaffinity
        self.counts.fill(0)
        self.ends.clear()


class ThreeAmplitudeController:
    def __init__(
        self,
        model: mujoco.MjModel,
        index,
        wave: float,
        body_yaw_deg: float,
        leg_yaw_deg: float,
        leg_pitch_deg: float,
    ):
        self.model = model
        self.index = index
        self.wave = float(wave)
        self.body_amplitude = math.radians(body_yaw_deg)
        self.leg_amplitudes = np.radians(
            np.asarray([leg_yaw_deg, leg_pitch_deg], dtype=float)
        )
        self.previous_target: np.ndarray | None = None
        self.target = np.zeros(len(index.active_joint_ids), dtype=float)
        self.spatial_phase = (
            2.0 * math.pi * self.wave * np.arange(19, dtype=float) / 18.0
        )
        self.leg_offsets = np.asarray([-math.pi / 2.0, math.pi], dtype=float)
        self.side_sign = np.asarray([1.0, -1.0], dtype=float)
        self.blend_start = 1.0 + np.arange(19, dtype=float) * 0.125 / 18.0

    def blend(self, t: float) -> np.ndarray:
        elapsed = np.clip(t - self.blend_start, 0.0, 0.125)
        return 0.5 * (1.0 - np.cos(math.pi * elapsed / 0.125))

    def step(self, data: mujoco.MjData) -> float:
        t = float(data.time)
        dt = float(self.model.opt.timestep)
        omega = 2.0 * math.pi * base.FREQUENCY_HZ
        target = self.target
        target.fill(0.0)
        phase = omega * t - self.spatial_phase
        blend = self.blend(t)
        body_target = blend * self.body_amplitude * np.sin(phase)
        body_target[0] = 0.0
        target[self.index.body_positions] = body_target
        target[self.index.pitch_positions] = 0.0
        phases = phase[:, None, None] + self.leg_offsets[None, None, :]
        leg_target = (
            blend[:, None, None]
            * self.side_sign[None, :, None]
            * self.leg_amplitudes[None, None, :]
            * np.sin(phases)
        )
        target[self.index.leg_positions.reshape(-1)] = leg_target.reshape(-1)
        if self.previous_target is None:
            target_velocity = np.zeros_like(target)
        else:
            target_velocity = (target - self.previous_target) / dt
        q = data.qpos[self.index.qpos_addresses]
        qdot = data.qvel[self.index.dof_addresses]
        composite = target + (self.index.kv / self.index.kp) * target_velocity
        torque = self.index.kp * (target - q) + self.index.kv * (
            target_velocity - qdot
        )
        data.ctrl[:] = 0.0
        data.ctrl[self.index.active_actuator_ids] = composite
        self.previous_target = target.copy()
        return float(np.max(np.abs(torque)))


class FastEvaluator:
    def __init__(self, shared_model=None, shared_index=None):
        if shared_model is None or shared_index is None:
            self.model = mujoco.MjModel.from_xml_path(str(MODEL))
            self.index = tip.LeanIndex(self.model)
            tip.configure_model(self.model, self.index)
        else:
            self.model = shared_model
            self.index = shared_index
        self.data = mujoco.MjData(self.model)
        self.ground = int(tip.stage1.ground_geom_id(self.model))
        self.foot_geoms = np.asarray(
            [
                base.name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    f"foot_{segment}_{side}_collision",
                )
                for segment in range(19)
                for side in range(2)
            ],
            dtype=int,
        )
        self.foot_lookup = set(int(value) for value in self.foot_geoms)
        chassis_bodies = set(int(value) for value in self.index.chassis_body_ids)
        self.chassis_geoms = set(
            geom_id
            for geom_id in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom_id]) in chassis_bodies
        )
        self.segment_collision_geoms = []
        for segment in range(19):
            roots = {
                base.name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    f"link_leg_{segment}_{side}_0",
                )
                for side in ("L", "R")
            }
            descendants = set()
            for body_id in range(self.model.nbody):
                cursor = body_id
                while cursor > 0:
                    if cursor in roots:
                        descendants.add(body_id)
                        break
                    cursor = int(self.model.body_parentid[cursor])
            geoms = np.asarray(
                [
                    geom_id
                    for geom_id in range(self.model.ngeom)
                    if int(self.model.geom_bodyid[geom_id]) in descendants
                    and (
                        int(self.model.geom_contype[geom_id]) != 0
                        or int(self.model.geom_conaffinity[geom_id]) != 0
                    )
                ],
                dtype=int,
            )
            if geoms.size == 0:
                raise RuntimeError(f"no collision geoms for segment pair {segment}")
            self.segment_collision_geoms.append(geoms)

    def evaluate(self, task: tuple[int, float, float, float, float, float]):
        wave, body_deg, leg_yaw_deg, leg_pitch_deg, duration_s, eval_start_s = task[:6]
        erasure_q = float(task[6]) if len(task) > 6 else 0.0
        erasure_seed = int(task[7]) if len(task) > 7 else 0
        model, data, index = self.model, self.data, self.index
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        controller = ThreeAmplitudeController(
            model, index, wave, body_deg, leg_yaw_deg, leg_pitch_deg
        )
        dt = float(model.opt.timestep)
        events, erasure_opportunities = build_erasure_events(
            int(wave), erasure_q, erasure_seed, eval_start_s, duration_s
        )
        erasure = ContactErasure(model, self.segment_collision_geoms, events)
        erased_segment_time = np.zeros(19, dtype=float)
        sample_stride = max(1, int(round(0.01 / dt)))
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
        step_index = 0
        while float(data.time) < duration_s - 0.5 * dt:
            erased = erasure.update(float(data.time))
            if float(data.time) >= eval_start_s:
                erased_segment_time += erased.astype(float) * dt
            active_torque_max = max(active_torque_max, controller.step(data))
            mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                finite = False
                break
            if float(data.time) < eval_start_s or step_index % sample_stride:
                step_index += 1
                continue
            evaluated_samples += 1
            center = base.chassis_centroid(data, index)
            posture = base.posture_metrics(data, index)
            min_upright = min(min_upright, float(posture["root_upright_cos"]))
            max_tortuosity = max(
                max_tortuosity, float(posture["spine_tortuosity"])
            )
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
            world = tip.signed_world_roll_deg(data, index.chassis_body_ids)
            radians = np.radians(world)
            coherent_roll_abs.append(
                abs(math.degrees(circular_mean_rad(radians)))
            )
            front_roll_abs.append(
                abs(math.degrees(circular_mean_rad(radians[:7])))
            )
            contacting = set()
            chassis_ground = False
            for contact_id in range(data.ncon):
                contact = data.contact[contact_id]
                geom1, geom2 = int(contact.geom1), int(contact.geom2)
                other = None
                if geom1 == self.ground:
                    other = geom2
                elif geom2 == self.ground:
                    other = geom1
                if other is None:
                    continue
                if other in self.foot_lookup:
                    contacting.add(other)
                    penetrations_mm.append(
                        max(0.0, -1000.0 * float(contact.dist))
                    )
                if other in self.chassis_geoms:
                    chassis_ground = True
            contact_counts.append(float(len(contacting)))
            if contacting:
                no_contact_run = 0
            else:
                no_contact_run += 1
                max_no_contact_run = max(max_no_contact_run, no_contact_run)
            chassis_ground_samples += int(chassis_ground)
            step_index += 1
        displacement = (
            base.FORWARD_SIGN * float(end[0] - start[0])
            if start is not None and end is not None
            else float("nan")
        )
        erasure.restore()
        path_efficiency = displacement / max(path_length, 1e-12)
        warnings = tip.stage1.warning_counts(data)
        metrics = {
            "forward_displacement_m": displacement,
            "path_efficiency": path_efficiency,
            "minimum_root_upright_cos": min_upright,
            "maximum_spine_tortuosity": max_tortuosity,
            "coherent_roll_p95_deg": percentile(coherent_roll_abs, 0.95),
            "front7_roll_p95_deg": percentile(front_roll_abs, 0.95),
            "chassis_z_peak_to_peak_mm": 1000.0
            * (max(center_z, default=float("nan")) - min(center_z, default=float("nan"))),
            "vertical_speed_abs_p99_m_s": percentile(vertical_speed_abs, 0.99),
            "active_joint_speed_abs_max_rad_s": active_speed_max,
            "tip_speed_abs_max_rad_s": tip_speed_max,
            "active_torque_abs_max_nm": active_torque_max,
            "mean_contacting_feet": float(np.mean(contact_counts))
            if contact_counts
            else float("nan"),
            "max_no_foot_contact_duration_s": max_no_contact_run
            * sample_stride
            * dt,
            "chassis_ground_contact_fraction": chassis_ground_samples
            / max(evaluated_samples, 1),
            "contact_penetration_p95_mm": percentile(penetrations_mm, 0.95),
            "tip_roll_p95_deg": percentile(tip_abs_deg, 0.95),
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
            "warning_counts": warnings,
            **metrics,
            "gates": gates,
            "feasible": bool(all(gates.values())),
        }


_EVALUATOR: FastEvaluator | None = None
_THREAD_MODEL = None
_THREAD_INDEX = None
_THREAD_LOCAL = threading.local()


def worker_init():
    global _EVALUATOR
    _EVALUATOR = FastEvaluator()


def worker_evaluate(task):
    if _EVALUATOR is None:
        worker_init()
    return _EVALUATOR.evaluate(task)


def thread_worker_evaluate(task):
    evaluator = getattr(_THREAD_LOCAL, "evaluator", None)
    if evaluator is None:
        evaluator = FastEvaluator(_THREAD_MODEL, _THREAD_INDEX)
        _THREAD_LOCAL.evaluator = evaluator
    return evaluator.evaluate(task)


def task_key(row_or_task) -> tuple[int, float, float, float]:
    if isinstance(row_or_task, dict):
        return (
            int(row_or_task["wave"]),
            float(row_or_task["body_yaw_amplitude_deg"]),
            float(row_or_task["leg_yaw_amplitude_deg"]),
            float(row_or_task["leg_pitch_amplitude_deg"]),
        )
    return (
        int(row_or_task[0]),
        float(row_or_task[1]),
        float(row_or_task[2]),
        float(row_or_task[3]),
    )


def load_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    scalar_rows = []
    for row in rows:
        scalar = {key: value for key, value in row.items() if key not in ("gates", "warning_counts")}
        scalar["failed_gates"] = ";".join(
            key for key, value in row["gates"].items() if not value
        )
        scalar_rows.append(scalar)
    if not scalar_rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scalar_rows[0]))
        writer.writeheader()
        writer.writerows(scalar_rows)


def summarize(rows: list[dict[str, object]], waves: list[int]) -> dict[str, object]:
    winners = {}
    for wave in waves:
        candidates = [
            row for row in rows if int(row["wave"]) == wave and bool(row["feasible"])
        ]
        candidates.sort(
            key=lambda row: (
                -float(row["forward_displacement_m"]),
                -float(row["path_efficiency"]),
                float(row["body_yaw_amplitude_deg"])
                + float(row["leg_yaw_amplitude_deg"])
                + float(row["leg_pitch_amplitude_deg"]),
            )
        )
        winner = candidates[0] if candidates else None
        if winner is None:
            wave_rows = [row for row in rows if int(row["wave"]) == wave]
            diagnostics = sorted(
                wave_rows,
                key=lambda row: (
                    sum(not bool(value) for value in row["gates"].values()),
                    -float(row["forward_displacement_m"]),
                ),
            )
            winners[str(wave)] = {
                "winner": None,
                "feasible_count": 0,
                "finite_grid_global": False,
                "status": "no_feasible_grid_point",
                "diagnostic_best": diagnostics[0] if diagnostics else None,
            }
            continue
        boundary = {
            "body_yaw": float(winner["body_yaw_amplitude_deg"])
            in (min(BODY_YAW_DEG), max(BODY_YAW_DEG)),
            "leg_yaw": float(winner["leg_yaw_amplitude_deg"])
            in (min(LEG_YAW_DEG), max(LEG_YAW_DEG)),
            "leg_pitch": float(winner["leg_pitch_amplitude_deg"])
            in (min(LEG_PITCH_DEG), max(LEG_PITCH_DEG)),
        }
        winners[str(wave)] = {
            "winner": winner,
            "feasible_count": len(candidates),
            "winner_on_grid_boundary": boundary,
            "finite_grid_global": True,
            "interior_optimum": not any(boundary.values()),
        }
    counts = {
        str(wave): sum(1 for row in rows if int(row["wave"]) == wave)
        for wave in waves
    }
    complete = all(counts[str(wave)] == GRID_SIZE_PER_WAVE for wave in waves)
    return {
        "claim": "global optimum on the pre-registered finite grid only",
        "continuous_global_optimum_claimed": False,
        "objective": "maximize forward_displacement_m after all hard gates",
        "waves": waves,
        "grid": {
            "body_yaw_amplitude_deg": BODY_YAW_DEG,
            "leg_yaw_amplitude_deg": LEG_YAW_DEG,
            "leg_pitch_amplitude_deg": LEG_PITCH_DEG,
            "size_per_wave": GRID_SIZE_PER_WAVE,
        },
        "evaluated_counts": counts,
        "grid_complete": complete,
        "winners": winners,
        "model": str(MODEL),
        "model_sha256": tip.sha256(MODEL),
        "armature_nonzero_count": 0,
    }


def run_search(
    waves: list[int],
    workers: int,
    duration_s: float,
    eval_start_s: float,
    backend: str,
):
    DATA.mkdir(parents=True, exist_ok=True)
    jsonl = DATA / "grid_trials.jsonl"
    rows = load_rows(jsonl)
    completed = {task_key(row) for row in rows}
    tasks = [
        (wave, body, leg_yaw, leg_pitch, duration_s, eval_start_s)
        for wave in waves
        for body in BODY_YAW_DEG
        for leg_yaw in LEG_YAW_DEG
        for leg_pitch in LEG_PITCH_DEG
        if (wave, body, leg_yaw, leg_pitch) not in completed
    ]
    before = tip.robot_sentinels()
    started = time.perf_counter()
    if tasks:
        global _THREAD_MODEL, _THREAD_INDEX
        if backend == "thread":
            _THREAD_MODEL = mujoco.MjModel.from_xml_path(str(MODEL))
            _THREAD_INDEX = tip.LeanIndex(_THREAD_MODEL)
            tip.configure_model(_THREAD_MODEL, _THREAD_INDEX)
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
            evaluate = thread_worker_evaluate
        else:
            context = multiprocessing.get_context("spawn")
            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=workers, mp_context=context, initializer=worker_init
            )
            evaluate = worker_evaluate
        with executor, jsonl.open("a", encoding="utf-8") as handle:
            futures = [executor.submit(evaluate, task) for task in tasks]
            for done, future in enumerate(
                concurrent.futures.as_completed(futures), start=1
            ):
                row = future.result()
                rows.append(row)
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                handle.flush()
                if done % 25 == 0 or done == len(tasks):
                    elapsed = time.perf_counter() - started
                    print(
                        f"completed {done}/{len(tasks)} new trials; "
                        f"{elapsed / done:.3f} s/trial wall average",
                        flush=True,
                    )
    rows.sort(key=task_key)
    write_csv(DATA / "grid_trials.csv", rows)
    summary = summarize(rows, waves)
    after = tip.robot_sentinels()
    summary["robot_sentinels_unchanged"] = before == after
    summary["new_trials"] = len(tasks)
    summary["backend"] = backend
    summary["workers"] = workers
    summary["elapsed_wall_s"] = time.perf_counter() - started
    (DATA / "grid_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    benchmark = sub.add_parser("benchmark")
    benchmark.add_argument("--wave", type=int, default=1)
    benchmark.add_argument("--duration", type=float, default=4.0)
    benchmark.add_argument("--body-yaw", type=float)
    benchmark.add_argument("--leg-yaw", type=float)
    benchmark.add_argument("--leg-pitch", type=float)
    benchmark.add_argument("--erasure-q", type=float, default=0.0)
    benchmark.add_argument("--erasure-seed", type=int, default=0)
    search = sub.add_parser("search")
    search.add_argument("--waves", nargs="+", type=int, default=list(WAVES))
    search.add_argument("--workers", type=int, default=max(1, min(6, os.cpu_count() or 1)))
    search.add_argument("--duration", type=float, default=4.0)
    search.add_argument("--eval-start", type=float, default=1.5)
    search.add_argument("--backend", choices=("process", "thread"), default="process")
    args = parser.parse_args()
    if args.command == "benchmark":
        evaluator = FastEvaluator()
        body = args.body_yaw if args.body_yaw is not None else math.degrees(
            base.BODY_AMPLITUDE_RAD * tip.BODY_AMPLITUDE_SCALE
        )
        leg_yaw = args.leg_yaw if args.leg_yaw is not None else math.degrees(
            base.LEG_YAW_AMPLITUDE_RAD * tip.ctrl.LEG_YAW_AMPLITUDE_SCALE
        )
        leg_pitch = args.leg_pitch if args.leg_pitch is not None else math.degrees(
            base.LEG_PITCH_AMPLITUDE_RAD * tip.ctrl.LEG_PITCH_AMPLITUDE_SCALE
        )
        started = time.perf_counter()
        row = evaluator.evaluate(
            (
                args.wave,
                body,
                leg_yaw,
                leg_pitch,
                args.duration,
                1.5,
                args.erasure_q,
                args.erasure_seed,
            )
        )
        row["elapsed_wall_s"] = time.perf_counter() - started
        print(json.dumps(row, indent=2))
    else:
        run_search(
            args.waves,
            args.workers,
            args.duration,
            args.eval_start,
            args.backend,
        )


if __name__ == "__main__":
    main()
