"""Matched kinematic audit for accepted anchors and rejected amplitude winners.

No model or controller source is edited.  Every case is replayed on the same
frozen locked-head Centipede Model using the same open-loop sinusoidal CPG.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / ".python_packages"))

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parent
MODEL_ROOT = PROJECT_ROOT
REJECTED_ROOT = (
    MODEL_ROOT
    / "experiments"
    / "mujoco_wave_amplitude_global_optimization_2026-08-31"
)
OPTIMIZER = REJECTED_ROOT / "run_wave_amplitude_global.py"
FINAL_SELECTION = REJECTED_ROOT / "outputs" / "data" / "final_selection_summary.json"
OUTPUTS = ROOT / "outputs"
DATA = OUTPUTS / "data"

DURATION_S = 6.0
EVAL_START_S = 1.5
SAMPLE_DT_S = 0.002


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


opt = load_module(OPTIMIZER, "centipede_rejected_amplitude_optimizer")


def historical_cases() -> list[dict[str, float | str]]:
    base = opt.base
    tip = opt.tip
    body_current = math.degrees(base.BODY_AMPLITUDE_RAD * tip.BODY_AMPLITUDE_SCALE)
    leg_yaw_current = math.degrees(
        base.LEG_YAW_AMPLITUDE_RAD * tip.ctrl.LEG_YAW_AMPLITUDE_SCALE
    )
    flat_pitch_baseline = math.degrees(
        base.LEG_PITCH_AMPLITUDE_RAD * tip.ctrl.LEG_PITCH_AMPLITUDE_SCALE
    )
    cases: list[dict[str, float | str]] = [
        {
            "case": "accepted_clean_w1p5",
            "provenance": "2026-08-31 selected clean flat profile",
            "wave": 1.5,
            "body_yaw_deg": body_current,
            "leg_yaw_deg": leg_yaw_current,
            "leg_pitch_deg": flat_pitch_baseline * 0.525,
        },
        {
            "case": "preclean_aggressive_w1p5",
            "provenance": "2026-08-26 flat lift profile before roll cleanup",
            "wave": 1.5,
            "body_yaw_deg": body_current,
            "leg_yaw_deg": leg_yaw_current,
            "leg_pitch_deg": flat_pitch_baseline,
        },
        {
            "case": "isaac_yaml_anchor_w1p5",
            "provenance": "IsaacLab YAML targets transferred to frozen MuJoCo model",
            "wave": 1.5,
            "body_yaw_deg": math.degrees(0.4),
            "leg_yaw_deg": math.degrees(0.6),
            "leg_pitch_deg": math.degrees(0.4),
        },
        {
            "case": "old_erasure_anchor_w3",
            "provenance": "2026-08-18 Model independent-erasure target amplitudes",
            "wave": 3.0,
            "body_yaw_deg": math.degrees(base.BODY_AMPLITUDE_RAD),
            "leg_yaw_deg": math.degrees(base.LEG_YAW_AMPLITUDE_RAD),
            "leg_pitch_deg": math.degrees(base.LEG_PITCH_AMPLITUDE_RAD),
        },
    ]
    final = json.loads(FINAL_SELECTION.read_text(encoding="utf-8"))
    for wave_text, item in final["waves"].items():
        selected = item.get("selected")
        row = selected["grid_row"] if selected else item["diagnostic_best"]
        cases.append(
            {
                "case": f"rejected_grid_w{int(wave_text)}",
                "provenance": "rejected 2026-08-31 displacement-grid selection",
                "wave": float(wave_text),
                "body_yaw_deg": float(row["body_yaw_amplitude_deg"]),
                "leg_yaw_deg": float(row["leg_yaw_amplitude_deg"]),
                "leg_pitch_deg": float(row["leg_pitch_amplitude_deg"]),
            }
        )
    return cases


def pctl(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.quantile(values, q)) if values.size else float("nan")


def sinusoid_metrics(times: np.ndarray, values: np.ndarray, requested_rad: float):
    omega = 2.0 * math.pi * opt.base.FREQUENCY_HZ
    design = np.column_stack(
        (np.sin(omega * times), np.cos(omega * times), np.ones_like(times))
    )
    gains = []
    residuals = []
    for column in values.T:
        beta, *_ = np.linalg.lstsq(design, column, rcond=None)
        fitted = design @ beta
        actual_amp = float(math.hypot(float(beta[0]), float(beta[1])))
        centered_rms = float(np.sqrt(np.mean((column - np.mean(column)) ** 2)))
        residual_rms = float(np.sqrt(np.mean((column - fitted) ** 2)))
        gains.append(actual_amp / max(abs(requested_rad), 1e-12))
        residuals.append(residual_rms / max(centered_rms, 1e-12))
    return np.asarray(gains), np.asarray(residuals)


def robust_zero_crossings(times: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    # 50 ms moving average rejects single-contact-sample chatter while retaining
    # the 1 Hz gait reversal.  A clean sinusoid has two velocity sign changes/cycle.
    window = max(3, int(round(0.05 / SAMPLE_DT_S)))
    kernel = np.ones(window, dtype=float) / window
    duration_cycles = max(
        (float(times[-1]) - float(times[0])) * opt.base.FREQUENCY_HZ, 1e-12
    )
    rates = []
    for column in velocities.T:
        smooth = np.convolve(column, kernel, mode="same")
        threshold = 0.05 * pctl(np.abs(smooth), 0.95)
        signs = np.sign(smooth)
        signs[np.abs(smooth) < threshold] = 0.0
        nonzero = signs[signs != 0.0]
        changes = int(np.count_nonzero(nonzero[1:] != nonzero[:-1])) if nonzero.size else 0
        rates.append(changes / duration_cycles)
    return np.asarray(rates, dtype=float)


def replay(case: dict[str, float | str]) -> dict[str, object]:
    evaluator = opt.FastEvaluator()
    model, data, index = evaluator.model, evaluator.data, evaluator.index
    controller = opt.ThreeAmplitudeController(
        model,
        index,
        float(case["wave"]),
        float(case["body_yaw_deg"]),
        float(case["leg_yaw_deg"]),
        float(case["leg_pitch_deg"]),
    )
    ground = evaluator.ground
    foot_geoms = evaluator.foot_geoms
    foot_lookup = {int(geom): i for i, geom in enumerate(foot_geoms)}
    foot_sites = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"site_foot_{seg}_{side}")
            for seg in range(19)
            for side in range(2)
        ],
        dtype=int,
    )
    dt = float(model.opt.timestep)
    stride = max(1, int(round(SAMPLE_DT_S / dt)))
    times = []
    body_q = []
    body_target = []
    leg_q = []
    leg_qvel = []
    leg_target = []
    foot_z = []
    contact_mask = []
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    step = 0
    while float(data.time) < DURATION_S - 0.5 * dt:
        controller.step(data)
        mujoco.mj_step(model, data)
        if float(data.time) < EVAL_START_S or step % stride:
            step += 1
            continue
        q_active = data.qpos[index.qpos_addresses]
        qvel_active = data.qvel[index.dof_addresses]
        times.append(float(data.time))
        body_q.append(q_active[index.body_positions].copy())
        body_target.append(controller.target[index.body_positions].copy())
        leg_q.append(q_active[index.leg_positions.reshape(-1)].copy())
        leg_qvel.append(qvel_active[index.leg_positions.reshape(-1)].copy())
        leg_target.append(controller.target[index.leg_positions.reshape(-1)].copy())
        foot_z.append(data.site_xpos[foot_sites, 2].copy())
        contacts = np.zeros(38, dtype=bool)
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            other = g2 if g1 == ground else (g1 if g2 == ground else -1)
            foot_index = foot_lookup.get(other)
            if foot_index is not None:
                contacts[foot_index] = True
        contact_mask.append(contacts)
        step += 1

    times_a = np.asarray(times)
    body_q_a = np.asarray(body_q)
    body_target_a = np.asarray(body_target)
    leg_q_a = np.asarray(leg_q).reshape(len(times), 19, 2, 2)
    leg_qvel_a = np.asarray(leg_qvel).reshape(len(times), 19, 2, 2)
    leg_target_a = np.asarray(leg_target).reshape(len(times), 19, 2, 2)
    foot_z_a = np.asarray(foot_z)
    contact_a = np.asarray(contact_mask)

    body_gain, body_resid = sinusoid_metrics(
        times_a, body_q_a[:, 1:], math.radians(float(case["body_yaw_deg"]))
    )
    yaw_gain, yaw_resid = sinusoid_metrics(
        times_a, leg_q_a[..., 0].reshape(len(times), -1), math.radians(float(case["leg_yaw_deg"]))
    )
    pitch_gain, pitch_resid = sinusoid_metrics(
        times_a, leg_q_a[..., 1].reshape(len(times), -1), math.radians(float(case["leg_pitch_deg"]))
    )
    yaw_cross = robust_zero_crossings(
        times_a, leg_qvel_a[..., 0].reshape(len(times), -1)
    )
    pitch_cross = robust_zero_crossings(
        times_a, leg_qvel_a[..., 1].reshape(len(times), -1)
    )
    swing_heights = foot_z_a[~contact_a]
    stance_heights = foot_z_a[contact_a]
    contact_duty = np.mean(contact_a, axis=0)
    tracking_yaw = leg_q_a[..., 0] - leg_target_a[..., 0]
    tracking_pitch = leg_q_a[..., 1] - leg_target_a[..., 1]

    base_metrics = evaluator.evaluate(
        (
            float(case["wave"]),
            float(case["body_yaw_deg"]),
            float(case["leg_yaw_deg"]),
            float(case["leg_pitch_deg"]),
            DURATION_S,
            EVAL_START_S,
        )
    )
    result = dict(case)
    result.update(
        {
            "duration_s": DURATION_S,
            "eval_start_s": EVAL_START_S,
            "samples": len(times),
            "forward_displacement_mm": 1000.0 * float(base_metrics["forward_displacement_m"]),
            "base_feasible_old_gates": bool(base_metrics["feasible"]),
            "body_response_gain_median": float(np.median(body_gain)),
            "body_nonsinusoidal_residual_p95": pctl(body_resid, 0.95),
            "leg_yaw_response_gain_median": float(np.median(yaw_gain)),
            "leg_yaw_response_gain_p95": pctl(yaw_gain, 0.95),
            "leg_yaw_nonsinusoidal_residual_p95": pctl(yaw_resid, 0.95),
            "leg_pitch_response_gain_median": float(np.median(pitch_gain)),
            "leg_pitch_response_gain_p95": pctl(pitch_gain, 0.95),
            "leg_pitch_nonsinusoidal_residual_p95": pctl(pitch_resid, 0.95),
            "leg_yaw_tracking_error_rms_deg": math.degrees(float(np.sqrt(np.mean(tracking_yaw ** 2)))),
            "leg_pitch_tracking_error_rms_deg": math.degrees(float(np.sqrt(np.mean(tracking_pitch ** 2)))),
            "leg_yaw_speed_p99_rad_s": pctl(np.abs(leg_qvel_a[..., 0]), 0.99),
            "leg_pitch_speed_p99_rad_s": pctl(np.abs(leg_qvel_a[..., 1]), 0.99),
            "leg_yaw_velocity_reversals_per_cycle_p95": pctl(yaw_cross, 0.95),
            "leg_pitch_velocity_reversals_per_cycle_p95": pctl(pitch_cross, 0.95),
            "contact_duty_mean": float(np.mean(contact_duty)),
            "contact_duty_p05": pctl(contact_duty, 0.05),
            "contact_duty_p95": pctl(contact_duty, 0.95),
            "left_right_contact_duty_abs_diff_mean": float(
                np.mean(np.abs(contact_duty.reshape(19, 2)[:, 0] - contact_duty.reshape(19, 2)[:, 1]))
            ),
            "swing_foot_height_p50_mm": 1000.0 * pctl(swing_heights, 0.50),
            "swing_foot_height_p95_mm": 1000.0 * pctl(swing_heights, 0.95),
            "stance_foot_height_p50_mm": 1000.0 * pctl(stance_heights, 0.50),
            "old_gate_failures": [name for name, passed in base_metrics["gates"].items() if not passed],
        }
    )
    return result


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in historical_cases():
        print(f"auditing {case['case']}", flush=True)
        rows.append(replay(case))
    json_path = DATA / "gait_naturality_audit.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    flat_rows = []
    for row in rows:
        flat = {}
        for key, value in row.items():
            flat[key] = json.dumps(value) if isinstance(value, (list, dict)) else value
        flat_rows.append(flat)
    csv_path = DATA / "gait_naturality_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    print(json.dumps(rows, indent=2))
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
